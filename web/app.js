// The shop floor, browser side.
//
// The page plays footage (a clip, an uploaded video or the camera), sends about three frames a second to the local
// server, and gets back who is in the picture and what each of them holds. Answers take about a second, so the
// picture is shown that much behind the footage: every outline is then drawn on the frame it was computed from,
// and moved smoothly between two analysed frames.

import { brand, currentShop, mountShell, rememberShop } from "/static/shell.js";
import { TICK, clockTime, dayOf, describe as describeEvent, esc, lineNote, moneyIn, stamp, unfolded } from "/static/words.js";

const $ = (id) => document.getElementById(id);
mountShell("floor");
const PALETTE = ["#3b82f6", "#f59e0b", "#10b981", "#a855f7", "#06b6d4", "#ec4899", "#84cc16", "#f97316"];
const VIEW_WIDTH = 960; // the picture on screen; frames sent for analysis may be wider (catalogue's frame_width)

const S = {
  config: null,
  ws: null,
  session: null,
  running: false,
  ended: false,
  finished: false,
  kind: "sample", // sample | file | camera
  wall0: 0, // the time of day (ms) at second 0 of a live camera's clock
  stream: null,
  fileUrl: null,
  t0: 0,
  stoppedAt: null,
  stoppedWall: 0,
  seq: 0,
  inFlight: 0,
  lastSend: -1,
  frames: [], // {t, bmp} ring buffer of what the camera saw
  states: [], // analysed frames, in time order
  inbox: [], // states and updates waiting for the picture to reach their time
  delay: 2.2,
  latencies: [],
  sentTimes: [],
  modelMs: null,
  carts: [],
  meta: { people: {}, payments: {} },
  events: [],
  zone: [],
  cartsKey: "",
  eventsKey: "",
  only: "", // the person whose activity is listed, "" for everyone
};

const video = $("video");
const canvas = $("view");
const ctx = canvas.getContext("2d");
const grab = document.createElement("canvas");

const colour = (pid) => PALETTE[(pid - 1) % PALETTE.length];
// "$249.00" for the shop's currency, or for the one a payment was made in
const money = (n, code) => moneyIn(n, code || S.config?.catalog.currency);
const ordinal = (n) => `${n}${["th", "st", "nd", "rd"][n % 100 > 10 && n % 100 < 14 ? 0 : Math.min(n % 10, 4) % 4]}`;

// Who a tracked person is: the shopper the gate or the floor checked in, or the registered shopper their face
// resembles too weakly to charge. Only a face nobody has registered stays a number.
function nameOf(pid) {
  const who = S.meta.people?.[pid];
  return who?.profile?.name ?? (who?.name || `Shopper ${pid}`);
}

// ---------------------------------------------------------------- setup

async function loadShop(shop) {
  S.config = await (await fetch(`/api/config?shop=${encodeURIComponent(shop ?? "")}`)).json();
  const name = S.config.shops.find((x) => x.id === S.config.shop)?.name ?? "Shop";
  brand(name, "floor");
  rememberShop(S.config.shop);
  $("shop").innerHTML = S.config.shops.map((x) => `<option value="${esc(x.id)}" ${x.id === S.config.shop ? "selected" : ""}>${esc(x.name)}</option>`).join("");
  fillSources();
}

// The footage list: a live camera, this shop's sample clips, or a video file.
function fillSources() {
  const before = $("source").value;
  $("source").innerHTML =
    `<option value="camera">Live camera</option>` +
    S.config.samples.filter((x) => x.shop === S.config.shop).map((x) => `<option value="sample:${esc(x.file)}">${esc(x.title)}</option>`).join("") +
    `<option value="file">${S.fileName ? `Video file: ${esc(S.fileName)}` : "A video file from this computer…"}</option>`;
  const keep = [...$("source").options].find((o) => o.value === before) ?? [...$("source").options].find((o) => o.value.startsWith("sample:"));
  if (keep) $("source").value = keep.value;
  $("camera-field").hidden = $("source").value !== "camera";
}

// Every camera this computer can see: built in, plugged in, or a phone or another computer sharing its camera.
// A browser only gives their names once it has been allowed to use one, so `ask` turns one on for a moment to get
// that permission. The choice is remembered for next time; the gate keeps its own choice on its own screen.
async function listCameras(ask = false) {
  const pick = $("camera");
  const found = async () => {
    try {
      return (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === "videoinput");
    } catch {
      return [];
    }
  };
  let cameras = await found();
  if (ask && !cameras.some((d) => d.label)) {
    try {
      (await navigator.mediaDevices.getUserMedia({ video: true, audio: false })).getTracks().forEach((t) => t.stop());
      cameras = await found();
    } catch {
      pick.innerHTML = `<option value="">Camera access was not allowed in this browser</option>`;
      return;
    }
  }
  const named = cameras.filter((d) => d.label);
  const before = pick.value || localStorage.getItem("floor-camera") || "";
  pick.innerHTML = named.length
    ? named.map((d) => `<option value="${esc(d.deviceId)}">${esc(d.label)}</option>`).join("")
    : `<option value="">${cameras.length ? "Choose Live camera to list the cameras" : "No camera found"}</option>`;
  if (named.some((d) => d.deviceId === before)) pick.value = before;
}
navigator.mediaDevices?.addEventListener?.("devicechange", () => listCameras());

async function boot() {
  await loadShop(currentShop());
  listCameras(); // names are there already if this browser has used a camera here before
  $("pay-note").textContent = S.config.services.payments ? "Charged at exit" : "Payments off";
  wake();
}

// The camera model runs on a GPU that is switched off when nobody has used it for five minutes, so that it costs
// nothing overnight. Switching it back on takes up to a minute, and Start has to wait for that. To keep Start
// instant during a demo, the page asks for it as it opens and keeps it awake while the page is in use.
const WARM_FOR = 4 * 60 * 1000; // an answer this recent means it is still up
const STAY_WARM_FOR = 20 * 60 * 1000; // after the last click or run, let it go back to sleep
let warmAt = -Infinity;
let usedAt = performance.now();
let waking = null;

function wake(quiet = false) {
  if (waking) return waking;
  const dot = $("model-dot");
  const text = $("model-text");
  if (!quiet) {
    dot.className = "dot";
    text.textContent = "Starting the vision model (up to a minute)";
  }
  waking = (async () => {
    try {
      const r = await fetch("/api/health");
      if (!r.ok) throw new Error((await r.json()).error);
      warmAt = performance.now();
      dot.className = "dot ok";
      text.textContent = "Cameras ready";
      return true;
    } catch (e) {
      dot.className = "dot bad";
      text.textContent = `Vision model unavailable: ${e.message}`;
      return false;
    } finally {
      waking = null;
    }
  })();
  return waking;
}

document.addEventListener("pointerdown", () => (usedAt = performance.now()), true);
setInterval(() => {
  const now = performance.now();
  if (document.hidden || now - warmAt < WARM_FOR - 30000) return;
  if (S.running && !S.ended) return void (warmAt = now); // frames are flowing: it is plainly up
  if (now - usedAt < STAY_WARM_FOR) wake(true);
  else if (now - warmAt > WARM_FOR + 60000 && $("model-dot").className === "dot ok") {
    $("model-dot").className = "dot off";
    $("model-text").textContent = "Vision model idle. Press Start to resume";
  }
}, 30000);

// ---------------------------------------------------------------- clock and capture

// Seconds of footage. For a file it is the video's own time, so a stalled video does not look like an empty shop.
function clock() {
  if (S.stoppedAt !== null) return S.stoppedAt + (performance.now() - S.stoppedWall) / 1000;
  return S.kind === "camera" ? (performance.now() - S.t0) / 1000 : video.currentTime;
}

let lastCapture = 0;
function capture() {
  if (!S.running || S.ended) return;
  const now = performance.now();
  if (now - lastCapture >= 40 && video.readyState >= 2 && video.videoWidth) {
    lastCapture = now;
    const t = clock();
    const w = Math.min(VIEW_WIDTH, video.videoWidth);
    const h = Math.round((w * video.videoHeight) / video.videoWidth);
    createImageBitmap(video, { resizeWidth: w, resizeHeight: h, resizeQuality: "low" })
      .then((bmp) => {
        if (!S.running) return bmp.close();
        S.frames.push({ t, bmp });
        if (S.frames.length > 1 && S.frames.at(-2).t > t) S.frames.sort((a, b) => a.t - b.t);
        while (S.frames.length > 220) S.frames.shift().bmp.close();
      })
      .catch(() => {});
  }
  if (video.requestVideoFrameCallback) video.requestVideoFrameCallback(capture);
  else requestAnimationFrame(capture);
}

function send() {
  if (!S.running || S.ended || !S.ws || S.ws.readyState !== 1 || !S.session) return;
  const t = clock();
  // the endpoint sets the pace: how many frames a second it takes, and a few in flight so none of that is wasted
  const fps = S.config.max_fps || 3;
  if (S.inFlight >= (fps > 3 ? 4 : 2) || t - S.lastSend < 1 / fps || video.readyState < 2 || !video.videoWidth) return;
  S.lastSend = t;
  S.inFlight += 1;
  const seq = S.seq++;
  // frame_width caps the longer side, so a phone clip shot upright is not sent at twice the size of a wide one
  const scale = Math.min(1, (S.config.catalog.frame_width || 960) / Math.max(video.videoWidth, video.videoHeight));
  grab.width = Math.round(video.videoWidth * scale);
  grab.height = Math.round(video.videoHeight * scale);
  grab.getContext("2d").drawImage(video, 0, 0, grab.width, grab.height);
  grab.toBlob(
    (blob) => {
      if (!blob || !S.ws || S.ws.readyState !== 1) return;
      S.ws.send(JSON.stringify({ type: "frame", seq, t }));
      S.ws.send(blob);
      S.sentTimes.push(performance.now());
    },
    "image/jpeg",
    0.8,
  );
}

// ---------------------------------------------------------------- start and stop

async function start() {
  const choice = $("source").value;
  const button = $("start");
  button.disabled = true;
  reset();
  try {
    usedAt = performance.now();
    if (performance.now() - warmAt > WARM_FOR) {
      button.textContent = "Starting…";
      if (!(await wake())) throw new Error("the camera model is not reachable");
    }
    if (choice === "camera") {
      S.kind = "camera";
      const id = $("camera").value;
      S.stream = await navigator.mediaDevices.getUserMedia({ video: { ...(id ? { deviceId: { exact: id } } : {}), width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false });
      video.removeAttribute("src");
      video.srcObject = S.stream;
      if (!id) listCameras(); // allowed just now: the cameras have names from here on
    } else {
      S.kind = choice === "file" ? "file" : "sample";
      if (choice === "file" && !S.fileUrl) throw new Error("choose a video file first");
      video.srcObject = null;
      video.src = choice === "file" ? S.fileUrl : `/samples/${choice.slice(7)}`;
      video.loop = false;
    }
    await openSocket();
    await video.play();
    S.t0 = performance.now();
    S.wall0 = Date.now();
    S.running = true;
    $("events").classList.toggle("clock", S.kind === "camera"); // a live camera: the activity list shows the time of day
    $("viewer").classList.add("live");
    $("viewer-note").textContent = "";
    button.textContent = "Stop";
    capture();
  } catch (e) {
    note(`Could not start: ${e.message}`);
    button.textContent = "Start";
    stopMedia();
  }
  button.disabled = false;
}

function openSocket() {
  return new Promise((resolve, reject) => {
    if (S.ws) S.ws.close();
    // A page opened over https (through a tunnel such as ngrok, as a phone's camera needs) may only open a secure
    // socket: a plain ws:// one is blocked by the browser before it reaches the server.
    const ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`);
    S.ws = ws;
    ws.onopen = () => ws.send(JSON.stringify({ type: "start", shop: S.config.shop, dry: $("dry").checked, live: S.kind === "camera" }));
    ws.onerror = () => {
      fetch("/api/staff/me"); // refused because the staff session ran out? that call answers 401, and shell.js sends the page to sign in
      reject(new Error("no connection to the shop's server"));
    };
    ws.onclose = () => {
      if (S.running && S.ws === ws) note("The connection to the shop's server closed.");
    };
    ws.onmessage = (m) => {
      const msg = JSON.parse(m.data);
      if (msg.type === "ready") {
        S.session = msg.session;
        return resolve();
      }
      if (msg.session !== S.session) return;
      if (msg.type === "state" || msg.type === "lost") S.inFlight = Math.max(0, S.inFlight - 1);
      if (msg.type === "lost") {
        if (msg.error) console.warn("frame lost:", msg.error);
        return;
      }
      if (msg.type === "state") {
        S.latencies.push(clock() - msg.t);
        if (S.latencies.length > 12) S.latencies.shift();
        S.modelMs = msg.ms?.model ?? S.modelMs;
        S.states.push(msg);
      }
      S.inbox.push(msg);
    };
  });
}

// The footage is over (or Stop was pressed): everyone still inside walks out, and the picture drains its backlog.
function end() {
  if (!S.running || S.ended) return;
  S.ended = true;
  S.stoppedAt = clock();
  S.stoppedWall = performance.now();
  stopMedia();
  if (S.ws?.readyState === 1) S.ws.send(JSON.stringify({ type: "end" }));
  $("start").textContent = "Start again";
}

function stopMedia() {
  video.pause();
  if (S.stream) S.stream.getTracks().forEach((t) => t.stop());
  S.stream = null;
}

function reset() {
  stopMedia();
  for (const f of S.frames) f.bmp.close();
  Object.assign(S, {
    running: false, ended: false, finished: false, session: null, stoppedAt: null, seq: 0, inFlight: 0, lastSend: -1,
    frames: [], states: [], inbox: [], delay: 2.2, latencies: [], sentTimes: [], carts: [], events: [], zone: [],
    meta: { people: {}, payments: {} }, cartsKey: "", eventsKey: "", only: "",
  });
  renderCarts();
  renderEvents();
  $("meter").textContent = "";
}

function note(text) {
  $("meter").textContent = text;
}

// ---------------------------------------------------------------- the delayed picture

function lerpBox(a, b, k) {
  return a.map((v, i) => v + (b[i] - v) * k);
}

function movePolys(polys, from, to) {
  const sx = (to[2] - to[0]) / Math.max(1e-6, from[2] - from[0]);
  const sy = (to[3] - to[1]) / Math.max(1e-6, from[3] - from[1]);
  return polys.map((poly) => poly.map(([x, y]) => [to[0] + (x - from[0]) * sx, to[1] + (y - from[1]) * sy]));
}

// An outline changes shape between two analysed frames. To draw the frames in between, the main outline of each
// is turned into the same number of points, evenly spaced and running the same way round, and the points are moved
// from one to the other. Without this the shape jumps three or four times a second.
const RING = 144; // enough points for the outline of a person, arms and all

function ring(o) {
  if (o.ring !== undefined) return o.ring;
  let main = null;
  let most = 0;
  let turn = 0;
  for (const poly of o.polygons) {
    let twice = 0;
    for (let i = 0; i < poly.length; i++) {
      const [x0, y0] = poly[i];
      const [x1, y1] = poly[(i + 1) % poly.length];
      twice += x0 * y1 - x1 * y0;
    }
    if (Math.abs(twice) > most) [main, most, turn] = [poly, Math.abs(twice), Math.sign(twice)];
  }
  if (!main || main.length < 3) return (o.ring = null);
  const pts = turn < 0 ? [...main].reverse() : main;
  const steps = pts.map((p, i) => Math.hypot(pts[(i + 1) % pts.length][0] - p[0], pts[(i + 1) % pts.length][1] - p[1]));
  const round = steps.reduce((a, b) => a + b, 0);
  const out = [];
  let i = 0;
  let walked = 0;
  for (let n = 0; n < RING; n++) {
    const at = (n / RING) * round;
    while (i < pts.length - 1 && walked + steps[i] < at) walked += steps[i++];
    const f = steps[i] ? (at - walked) / steps[i] : 0;
    const q = pts[(i + 1) % pts.length];
    out.push([pts[i][0] + (q[0] - pts[i][0]) * f, pts[i][1] + (q[1] - pts[i][1]) * f]);
  }
  o.main = main;
  return (o.ring = out);
}

// Which point of `b`'s ring goes with the first point of `a`'s: the turn that moves the points least.
function turnFor(a, b) {
  if (a.turnTo === b) return a.turn;
  const [ra, rb] = [ring(a), ring(b)];
  const [ax, ay, bx, by] = [(a.box[0] + a.box[2]) / 2, (a.box[1] + a.box[3]) / 2, (b.box[0] + b.box[2]) / 2, (b.box[1] + b.box[3]) / 2];
  let best = 0;
  let least = Infinity;
  for (let shift = 0; shift < RING; shift++) {
    let sum = 0;
    for (let i = 0; i < RING && sum < least; i += 3) {
      const q = rb[(i + shift) % RING];
      sum += (ra[i][0] - ax - q[0] + bx) ** 2 + (ra[i][1] - ay - q[1] + by) ** 2;
    }
    if (sum < least) [least, best] = [sum, shift];
  }
  a.turnTo = b;
  return (a.turn = best);
}

// Objects of state `a`, eased towards where state `b` has them. Things only in one of the two fade in or out.
function between(listA, listB, k) {
  const out = [];
  const inB = new Map((listB ?? []).map((o) => [o.id, o]));
  for (const o of listA) {
    const next = inB.get(o.id);
    if (next) {
      const box = lerpBox(o.box, next.box, k);
      let polygons;
      if (ring(o) && ring(next)) {
        const shift = turnFor(o, next);
        const near = k < 0.5 ? o : next; // the smaller pieces of an outline (a hand beyond a shelf edge) come from the nearer frame
        polygons = [o.ring.map((p, i) => [p[0] + (next.ring[(i + shift) % RING][0] - p[0]) * k, p[1] + (next.ring[(i + shift) % RING][1] - p[1]) * k])];
        polygons.push(...movePolys(near.polygons.filter((poly) => poly !== near.main), near.box, box));
      } else polygons = movePolys(o.polygons, o.box, box);
      out.push({ ...o, box, polygons, alpha: 1 });
      inB.delete(o.id);
    } else out.push({ ...o, alpha: listB ? 1 - k : 1 });
  }
  for (const o of inB.values()) out.push({ ...o, alpha: k });
  return out;
}

function path(polys, W, H) {
  ctx.beginPath();
  for (const poly of polys) {
    poly.forEach(([x, y], i) => (i ? ctx.lineTo(x * W, y * H) : ctx.moveTo(x * W, y * H)));
    ctx.closePath();
  }
}

// A small label running up a tall item (a book spine), or across a wide one.
function spineLabel(text, box, W, H, px) {
  const bw = (box[2] - box[0]) * W;
  const bh = (box[3] - box[1]) * H;
  const along = Math.max(bw, bh);
  ctx.font = `500 ${10 * px}px "Instrument Sans", ui-sans-serif, system-ui, sans-serif`;
  let label = text;
  while (label.length > 4 && ctx.measureText(label).width > along - 10 * px) label = label.slice(0, -2);
  if (label !== text) label = label.trimEnd() + "…";
  const tw = ctx.measureText(label).width + 8 * px;
  const th = 14 * px;
  ctx.save();
  ctx.translate(((box[0] + box[2]) / 2) * W, ((box[1] + box[3]) / 2) * H);
  if (bh > bw) ctx.rotate(-Math.PI / 2);
  ctx.fillStyle = "rgba(22,21,19,0.78)";
  ctx.beginPath();
  ctx.roundRect(-tw / 2, -th / 2, tw, th, 3 * px);
  ctx.fill();
  ctx.fillStyle = "#fff";
  ctx.textBaseline = "middle";
  ctx.textAlign = "center";
  ctx.fillText(label, 0, px * 0.5);
  ctx.restore();
}

function chip(text, x, y, bg, px) {
  ctx.font = `500 ${13 * px}px "Instrument Sans", ui-sans-serif, system-ui, sans-serif`;
  const w = ctx.measureText(text).width + 14 * px;
  const h = 22 * px;
  x = Math.max(4 * px, Math.min(x, canvas.width - w - 4 * px));
  y = Math.max(4 * px, Math.min(y, canvas.height - h - 4 * px));
  ctx.fillStyle = bg;
  ctx.beginPath();
  ctx.roundRect(x, y, w, h, 4 * px);
  ctx.fill();
  ctx.fillStyle = "#fff";
  ctx.textBaseline = "middle";
  ctx.textAlign = "left";
  ctx.fillText(text, x + 7 * px, y + h / 2 + px * 0.5);
}

// A small label for something in a shopper's hands: a bar in the shopper's colour, a tick once it is sure, the name.
// It is moved down past any label already `placed` there, and returns where it ended up.
function tag(text, sure, x, y, bar, px, placed) {
  ctx.font = `500 ${12 * px}px "Instrument Sans", ui-sans-serif, system-ui, sans-serif`;
  if (text.length > 34) text = text.slice(0, 32).trimEnd() + "…";
  const tick = sure ? 14 * px : 0;
  const w = ctx.measureText(text).width + 17 * px + tick;
  const h = 20 * px;
  x = Math.max(4 * px, Math.min(x, canvas.width - w - 4 * px));
  y = Math.max(4 * px, y);
  const hits = (r) => x < r.x + r.w && r.x < x + w && y < r.y + r.h + 2 * px && r.y < y + h + 2 * px;
  for (let tries = 0; tries < 12 && placed.some(hits); tries++) y += h + 2 * px;
  y = Math.min(y, canvas.height - h - 4 * px);
  ctx.fillStyle = "rgba(22,21,19,0.84)";
  ctx.beginPath();
  ctx.roundRect(x, y, w, h, 4 * px);
  ctx.fill();
  ctx.fillStyle = bar;
  ctx.beginPath();
  ctx.roundRect(x, y, 4 * px, h, [4 * px, 0, 0, 4 * px]);
  ctx.fill();
  if (sure) {
    ctx.strokeStyle = "#8fd9b0";
    ctx.lineWidth = 1.8 * px;
    ctx.lineCap = ctx.lineJoin = "round";
    ctx.beginPath();
    ctx.moveTo(x + 10 * px, y + h / 2 + 0.5 * px);
    ctx.lineTo(x + 13 * px, y + h / 2 + 3.5 * px);
    ctx.lineTo(x + 18.5 * px, y + h / 2 - 3.5 * px);
    ctx.stroke();
    ctx.lineCap = ctx.lineJoin = "miter";
  }
  ctx.fillStyle = sure ? "#fff" : "rgba(255,255,255,0.82)";
  ctx.textBaseline = "middle";
  ctx.textAlign = "left";
  ctx.fillText(text, x + 10 * px + tick, y + h / 2 + px * 0.5);
  const rect = { x, y, w, h };
  placed.push(rect);
  return rect;
}

// What the label says about a bag line: [text, is it sure enough to be billed]. Where the shop has pictures of the
// product, how alike the item in hand looks to them follows the name.
function labelOf(line) {
  const like = line.likeness && line.likeness.product === line.product ? `  ${Math.round(line.likeness.score * 100)} %` : "";
  if (line.product && !line.billable && !line.unread) return [`${line.name} (own item)`, false];
  if (line.unread) return [`${line.name}: identifying`, false];
  if (line.charge) return [line.name + like, true];
  if (line.verified === "disputed" || line.repeat || line.audit) return [`${line.name}: held for review`, false];
  return [`${line.name}: checking${like}`, false];
}

// A live camera carries the date and the time of day of the frame on screen, top right, like any shop camera.
function liveStamp(t, px) {
  if (S.kind !== "camera") return;
  const at = S.wall0 + t * 1000;
  ctx.globalAlpha = 1;
  ctx.font = `500 ${13 * px}px "Instrument Sans", ui-sans-serif, system-ui, sans-serif`;
  const day = dayOf(at);
  const w = ctx.measureText(`${day}  00:00:00`).width + 16 * px; // sized for the widest digits, so it does not twitch
  const h = 24 * px;
  const x = canvas.width - w - 8 * px;
  ctx.fillStyle = "rgba(22,21,19,0.72)";
  ctx.beginPath();
  ctx.roundRect(x, 8 * px, w, h, 4 * px);
  ctx.fill();
  ctx.fillStyle = "#fff";
  ctx.textBaseline = "middle";
  ctx.textAlign = "left";
  ctx.fillText(day, x + 8 * px, 8 * px + h / 2 + px * 0.5);
  ctx.textAlign = "right";
  ctx.fillText(clockTime(at), x + w - 8 * px, 8 * px + h / 2 + px * 0.5);
}

function draw() {
  requestAnimationFrame(draw);
  if (!S.running) return;

  // how far behind the footage the picture must run to have an analysed frame on both sides of it
  if (S.latencies.length >= 2) {
    const worst = [...S.latencies].sort((a, b) => a - b)[Math.floor(S.latencies.length * 0.85)];
    const gap = S.states.length > 1 ? S.states.at(-1).t - S.states.at(-2).t : 0.5;
    const want = Math.min(8, Math.max(1.2, worst + gap + 0.25));
    S.delay = want > S.delay ? want : Math.max(want, S.delay - 0.002);
  }
  const T = clock() - S.delay;

  while (S.inbox.length && S.inbox[0].t <= T) apply(S.inbox.shift());
  if (S.ended && !S.states.some((s) => s.t > T)) while (S.inbox.length) apply(S.inbox.shift());

  let frame = null;
  for (const f of S.frames) {
    if (f.t <= T) frame = f;
    else break;
  }
  while (S.frames.length > 2 && S.frames[1].t < T - 1.5) S.frames.shift().bmp.close();
  if (!frame) return;

  const px = Math.min(2, window.devicePixelRatio || 1);
  const W = Math.round(canvas.clientWidth * px);
  const H = Math.round((W * frame.bmp.height) / frame.bmp.width);
  if (canvas.width !== W || canvas.height !== H) {
    canvas.width = W;
    canvas.height = H;
    $("viewer").style.aspectRatio = `${frame.bmp.width} / ${frame.bmp.height}`;
  }
  ctx.drawImage(frame.bmp, 0, 0, W, H);

  let i = -1;
  for (let k = 0; k < S.states.length; k++) if (S.states[k].t <= T) i = k;
  while (i > 1) {
    S.states.shift();
    i -= 1;
  }
  if (i < 0) return liveStamp(frame.t, px);
  const a = S.states[i];
  const b = S.states[i + 1];
  const k = b ? Math.min(1, Math.max(0, (T - a.t) / Math.max(1e-3, b.t - a.t))) : 0;
  if (!b && T - a.t > 1.5 && !S.ended) return liveStamp(frame.t, px); // nothing fresh to show: better no outline than a stale one

  if ($("show-zone").checked && S.zone.length) {
    ctx.setLineDash([6 * px, 6 * px]);
    ctx.lineWidth = 1.5 * px;
    ctx.strokeStyle = "rgba(255,255,255,0.7)";
    path(S.zone, W, H);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  const items = between(a.items, b?.items, k);
  if ($("show-stock").checked) {
    ctx.lineWidth = 1.25 * px;
    for (const item of items) {
      if (item.state !== "rest") continue;
      ctx.globalAlpha = 0.75 * item.alpha;
      ctx.strokeStyle = "#fff";
      if (item.unread) ctx.setLineDash([4 * px, 4 * px]); // one of a family (a book) whose title is not read yet
      path(item.polygons, W, H);
      ctx.stroke();
      ctx.setLineDash([]);
    }
    // what each resting item was read as, written along it
    ctx.globalAlpha = 1;
    for (const item of items) {
      if (item.state !== "rest" || item.unread || !item.read) continue;
      spineLabel(item.name, item.box, W, H, px);
    }
  }

  const people = between(a.people, b?.people, k);
  for (const p of people) {
    ctx.globalAlpha = p.alpha;
    path(p.polygons, W, H);
    ctx.fillStyle = colour(p.id) + "30";
    ctx.fill();
    ctx.lineWidth = 2 * px;
    ctx.strokeStyle = colour(p.id);
    ctx.stroke();
  }

  if ($("show-hands").checked) {
    ctx.globalAlpha = 1;
    ctx.lineWidth = 1.5 * px;
    for (const hand of (k < 0.5 || !b ? a : b).hands) {
      if (hand.person == null) continue;
      ctx.strokeStyle = "#fff";
      path(hand.polygons, W, H);
      ctx.stroke();
    }
  }

  const boxes = $("show-boxes").checked;
  for (const item of items) {
    if (item.state === "rest" || (item.state === "candidate" && !item.holder)) continue;
    const c = item.holder ? colour(item.holder) : "#ffffff";
    ctx.globalAlpha = item.alpha;
    path(item.polygons, W, H);
    ctx.fillStyle = c + (item.state === "held" ? "88" : "33");
    ctx.fill();
    ctx.lineWidth = 2.5 * px;
    ctx.strokeStyle = c;
    ctx.stroke();
    if (boxes && item.state === "held") {
      // the box the likeness check and the second model are shown
      const [x0, y0, x1, y1] = item.box;
      const pad = 4 * px;
      ctx.lineWidth = 1.5 * px;
      ctx.setLineDash([6 * px, 4 * px]);
      ctx.strokeRect(x0 * W - pad, y0 * H - pad, (x1 - x0) * W + 2 * pad, (y1 - y0) * H + 2 * pad);
      ctx.setLineDash([]);
    }
  }

  // What is in each shopper's hands, by name. A label sits under the item it names and follows it. A title that is
  // on their bag but whose outline is hidden (the lower books of a stack) is listed under the same hands, so the
  // picture always says what the bag says.
  ctx.globalAlpha = 1;
  const placed = [];
  const named = new Set();
  const hands = (k < 0.5 || !b ? a : b).hands;
  for (const cart of S.carts) {
    const pid = cart.person;
    if (!people.some((p) => p.id === pid && p.alpha >= 0.5)) continue;
    let last = null;
    const hidden = [];
    for (const line of cart.lines) {
      if (line.state !== "in_cart") continue;
      const item = items.find((o) => o.id === line.item && o.state === "held" && o.alpha >= 0.5);
      if (!item) {
        if (!(line.product && !line.billable && !line.unread)) hidden.push(line);
        continue;
      }
      named.add(item.id);
      last = tag(...labelOf(line), item.box[0] * W, item.box[3] * H + 6 * px, colour(pid), px, placed);
    }
    if (!hidden.length) continue;
    if (!last) {
      const mine = hands.filter((h) => h.person === pid);
      if (!mine.length) continue; // their hands are out of view: the label over their head still counts the items
      last = { x: Math.min(...mine.map((h) => h.box[0])) * W, y: Math.max(...mine.map((h) => h.box[3])) * H + 4 * px, w: 0, h: 0 };
    }
    for (const line of hidden) last = tag(...labelOf(line), last.x, last.y + last.h + 2 * px, colour(pid), px, placed);
  }
  // something in a hand that is on nobody's bag yet
  for (const item of items) {
    if (item.state === "held" && item.alpha >= 0.5 && !named.has(item.id)) tag(item.name, false, item.box[0] * W, item.box[3] * H + 6 * px, item.holder ? colour(item.holder) : "#fff", px, placed);
  }

  const carts = new Map(S.carts.map((cart) => [cart.person, cart]));
  for (const p of people) {
    if (p.alpha < 0.5) continue;
    const cart = carts.get(p.id);
    const n = cart?.count ?? 0;
    const out = S.meta.people?.[p.id]?.status === "not_registered" ? ", not identified" : "";
    const label = `${nameOf(p.id)}${out}${n ? `  ${n} item${n > 1 ? "s" : ""}  ${money(cart.total)}` : ""}`;
    chip(label, p.box[0] * W, p.box[1] * H - 28 * px, colour(p.id), px);
  }
  liveStamp(frame.t, px);

  const sent = S.sentTimes.filter((t) => performance.now() - t < 5000);
  S.sentTimes = sent;
  if (!S.ended)
    $("meter").textContent =
      `Watching at ${(sent.length / 5).toFixed(1)} frames a second, ${S.delay.toFixed(1)} s behind the footage` +
      (S.meta.shelf ? `. ${S.meta.shelf.read} of ${S.meta.shelf.items} items on the shelf identified` : "") +
      (S.meta.dry ? ". Rehearsal: nobody is charged" : "");
}

// A state or update whose moment the picture has reached.
function apply(msg) {
  S.carts = msg.carts ?? S.carts;
  S.meta = msg.meta ?? S.meta;
  if (msg.zone) S.zone = msg.zone;
  for (const e of msg.events ?? []) if (!S.events.some((x) => x.id === e.id)) S.events.push(e);
  // someone recognised again by their face: what was listed under their new outline's number is theirs
  for (const same of S.events.filter((e) => e.type === "same")) {
    for (const e of S.events) {
      if (e.person === same.was) e.person = same.person;
      if (e.from_person === same.was) e.from_person = same.person;
    }
    if (S.only === String(same.was)) S.only = String(same.person);
  }
  S.events = unfolded(S.events);
  if (msg.type === "finished") {
    S.finished = true;
    note(`The footage ended. Everyone still inside was treated as walking out.${S.meta.dry ? " Rehearsal: nobody was charged." : ""}`);
  }
  renderCarts();
  renderEvents();
}

// ---------------------------------------------------------------- carts

// What happened at the door, in the order it happens: the last check, the charge, the invoice, the email.
function door(pay, who, cart) {
  const step = (cls, html) => `<li class="${cls}">${cls === "done" ? TICK : ""}<span>${html}</span></li>`;
  if (!pay) return who.checking ? `<ol class="door">${step("wait", "Final check in progress")}</ol>` : "";
  const steps = [];
  const amount = money(pay.amount, pay.currency);
  const method = esc(pay.method || "the saved method");
  const check = pay.check ?? {};
  const seen = (check.kept ?? 0) + (check.held ?? 0);
  const shots = check.pictures
    ? `<span class="shots">${Array.from({ length: check.pictures }, (_, i) => {
        const src = `/api/evidence/${esc(check.session)}/${check.person}/${i}`;
        return `<a href="${src}" target="_blank" rel="noopener"><img src="${src}" alt="Final check, picture ${String.fromCharCode(65 + i)}" loading="lazy" /></a>`;
      }).join("")}</span>`
    : "";
  if (check.status === "checked")
    steps.push(step(check.held ? "stop" : "done", `Final check: ${check.kept} of ${seen} item${seen === 1 ? "" : "s"} confirmed${check.held ? `, ${check.held} held for review` : ""}${shots}`));
  else if (check.status === "failed") steps.push(step("stop", "Final check unavailable, nothing charged"));
  switch (pay.status) {
    case "succeeded":
      steps.push(step("done", `Paid ${amount} with ${method}`));
      break;
    case "dry":
      steps.push(step("wait", `Rehearsal: ${amount} not charged`));
      break;
    case "nothing_due":
      // empty hands, or things in them that could not be billed: say which, the lines above say why
      if (cart?.review) steps.push(step("stop", `${cart.review} item${cart.review === 1 ? "" : "s"} held for review, nothing charged`));
      else steps.push(step("wait", "Left with nothing to pay"));
      break;
    case "not_registered":
      steps.push(step("stop", `${amount} not charged: shopper not registered`));
      break;
    case "declined":
      steps.push(step("bad", `${method} declined for ${amount}`));
      break;
    case "no_card":
      steps.push(step("bad", `${amount} not charged: no payment method on file`));
      break;
    default:
      steps.push(step("bad", `Not charged (${amount}): ${esc(pay.message)}`));
  }
  if (pay.status === "succeeded") {
    const sv = S.config.services;
    const inv = pay.invoice;
    if (inv?.status === "created")
      steps.push(step("done", `Invoice ${esc(inv.number ?? "")} created. <a href="${esc(inv.url)}" target="_blank" rel="noopener">View on PayPal</a>`));
    else if (inv) steps.push(step("bad", `PayPal invoice not created: ${esc(inv.message)}`));
    else if (sv.provider === "paypal" && !S.finished) steps.push(step("wait", "Creating PayPal invoice"));
    if (pay.email) steps.push(pay.email.status === "sent" ? step("done", `Invoice emailed to ${esc(pay.email.to)}`) : step("bad", `Invoice email failed: ${esc(pay.email.message)}`));
    else if (sv.email && who.profile?.email && !S.finished) steps.push(step("wait", `Emailing invoice to ${esc(who.profile.email)}`));
  }
  return `<ol class="door">${steps.join("")}</ol>`;
}

// One bag as its parts. Each part is put on the page only when it has changed, so a line being confirmed does not
// redraw the shopper's picture, and a new bag is the only thing that moves when someone walks in.
function bagParts(cart) {
  const pid = cart.person;
  const who = S.meta.people?.[pid] ?? { status: "no_face" };
  const p = who.profile;
  const left = cart.state === "left";
  const picture = p?.has_face ? p.id : who.face;
  const face = picture
    ? `<img class="face" style="--c:${colour(pid)}" src="/api/face/${esc(picture)}" alt="" width="44" height="44" />`
    : `<span class="face none" style="--c:${colour(pid)}" aria-hidden="true">${pid}</span>`;
  let sub = "";
  let notice = "";
  if (p) {
    const nth = p.visits + (left ? 0 : 1);
    const sure = who.score ? `. Face match ${Math.round(who.score * 100)}%` : "";
    sub = `<span class="sub">${esc(p.method || "No payment method on file")}${sure}</span><span class="sub">${nth > 1 ? `${ordinal(nth)} visit` : "First visit"}${p.recent?.length && !left ? `. Last time: ${esc(p.recent.join(", "))}` : ""}</span>`;
  } else if (who.status === "not_registered") {
    sub = `<span class="sub">${who.name ? `Resembles ${esc(who.name)}, match too weak` : "Not registered"}</span>`;
    if (!left)
      notice = `<p class="notice">${who.name ? `Not sure enough this is ${esc(who.name)}. A clearer view of their face checks them in.` : "This shopper has not registered at the gate."} Nothing will be charged until then. <a href="/gate" target="_blank" rel="noopener">Open the gate</a></p>`;
  } else sub = `<span class="sub">${left ? "Never recognised" : "Looking for their face"}</span>`;
  const lines = cart.lines
    .map((line) => {
      const [text, kind, story] = lineNote(line);
      const off = line.state === "in_cart" && line.billable && !line.charge;
      const cls = line.state !== "in_cart" ? "gone" : line.product && !line.billable && !line.unread ? "own" : off ? "off" : "";
      const shelf = [line.category, line.section].filter(Boolean).join(", ");
      return `<li class="${cls}"><span class="name">${esc(line.name)}${shelf ? `<small>${esc(shelf)}</small>` : ""}</span><span class="price">${line.price == null || !line.billable ? "" : line.price.toFixed(2)}</span><span class="note ${kind}"${story ? ` title="${esc(story)}"` : ""}>${kind === "took" ? TICK : ""}<span>${esc(text)}</span></span></li>`;
    })
    .join("");
  const totalLabel = who.status === "not_registered" ? "Not charged" : left ? "Total" : "To pay on the way out";
  return {
    head: `${face}<div class="who"><strong>${esc(nameOf(pid))}</strong>${sub}</div><span class="state">${left ? "Left" : "In the shop"}</span>`,
    notice,
    lines: lines ? `<ul class="lines">${lines}</ul>` : `<p class="nothing">Nothing picked up.</p>`,
    total: cart.lines.length ? `<div class="total"><span>${totalLabel}</span><span class="price">${money(cart.total)}</span></div>` : "",
    door: door(S.meta.payments?.[pid], who, cart),
  };
}

function renderCarts() {
  const key = JSON.stringify([S.carts, S.meta, S.finished]);
  if (key === S.cartsKey) return;
  S.cartsKey = key;
  const box = $("carts");
  if (!S.carts.length) {
    box.innerHTML = `<p class="empty">No shoppers in store. Shoppers appear here after entering through the gate.</p>`;
    return;
  }
  box.querySelector(":scope > .empty")?.remove();
  const here = new Map([...box.querySelectorAll(":scope > .bag")].map((el) => [el.dataset.person, el]));
  for (const cart of S.carts) {
    const id = String(cart.person);
    let el = here.get(id);
    if (!el) {
      el = document.createElement("article");
      el.className = "bag new";
      el.dataset.person = id;
      el.innerHTML = `<header class="bag-head" data-part="head"></header><div data-part="notice"></div><div data-part="lines"></div><div data-part="total"></div><div data-part="door"></div>`;
      el.addEventListener("animationend", () => el.classList.remove("new"), { once: true });
      box.append(el); // bags stay in the order people walked in
    }
    here.delete(id);
    for (const [part, html] of Object.entries(bagParts(cart))) {
      const slot = el.querySelector(`:scope > [data-part="${part}"]`);
      if (slot.shown !== html) slot.innerHTML = slot.shown = html;
    }
  }
  for (const el of here.values()) el.remove(); // an outline that turned out to be someone already here
}

// ---------------------------------------------------------------- events

const describe = (e) => describeEvent(e, { who: nameOf, line: (id) => S.carts.flatMap((c) => c.lines).find((l) => l.id === id), money });

const RECENT = 8; // the list shows this many until "Show all" is pressed

function renderEvents() {
  const people = S.carts.map((c) => [String(c.person), nameOf(c.person)]);
  const key = JSON.stringify([S.events, S.carts.map((c) => c.lines.map((l) => l.name)), people, S.only]);
  if (key === S.eventsKey) return;
  const had = new Set([...$("events").querySelectorAll("li[data-id]")].map((li) => li.dataset.id));
  S.eventsKey = key;
  // whose activity to list: everyone's, or one shopper's
  const pick = $("who");
  const choices = JSON.stringify(people);
  if (pick.dataset.people !== choices) {
    pick.dataset.people = choices;
    pick.innerHTML = `<option value="">Everyone</option>` + people.map(([id, name]) => `<option value="${id}">${esc(name)}</option>`).join("");
  }
  if (!people.some(([id]) => id === S.only)) S.only = "";
  pick.value = S.only;
  pick.hidden = !people.length;
  const listed = S.only ? S.events.filter((e) => String(e.person) === S.only || String(e.from_person) === S.only) : S.events;
  const more = $("more-events");
  more.hidden = listed.length <= RECENT;
  more.textContent = $("events").classList.contains("all") ? "Show the latest" : `Show all ${listed.length}`;
  if (!listed.length) {
    $("events").innerHTML = `<li class="empty">${S.only ? "Nothing from this shopper yet." : "Nothing yet. What shoppers take, put back and pay for is listed here."}</li>`;
    return;
  }
  $("events").innerHTML = [...listed]
    .sort((a, b) => b.id - a.id)
    .map((e, i) => {
      const [text, minor] = describe(e);
      const cls = [had.has(String(e.id)) ? "" : "new", i >= RECENT ? "older" : ""].join(" ").trim();
      const at = S.kind === "camera" ? clockTime(S.wall0 + e.t * 1000) : stamp(e.t);
      return `<li data-id="${e.id}" class="${cls}"><time>${at}</time><span class="swatch" style="background:${colour(e.person)}"></span><span class="${minor ? "minor" : ""}">${text}</span></li>`;
    })
    .join("");
}

$("who").onchange = () => {
  S.only = $("who").value;
  renderEvents();
};

$("more-events").onclick = () => {
  $("events").classList.toggle("all");
  S.eventsKey = "";
  renderEvents();
};

// ---------------------------------------------------------------- controls

$("start").onclick = () => (S.running && !S.ended ? end() : start());
$("shop").onchange = async () => {
  rememberShop($("shop").value);
  if (S.running && !S.ended) end();
  await loadShop($("shop").value);
};
$("source").onchange = () => {
  const choice = $("source").value;
  if (choice === "file") $("file").click();
  $("camera-field").hidden = choice !== "camera";
  if (choice === "camera") listCameras(true);
};
$("camera").onchange = () => {
  localStorage.setItem("floor-camera", $("camera").value);
  if (S.running && !S.ended && S.kind === "camera") end(); // a different camera is a different visit
};
$("file").onchange = () => {
  const file = $("file").files[0];
  if (!file) return;
  if (S.fileUrl) URL.revokeObjectURL(S.fileUrl);
  S.fileUrl = URL.createObjectURL(file);
  S.fileName = file.name;
  $("source").selectedOptions[0].textContent = `Video file: ${file.name}`;
};

// Sending does not wait for the screen: a timer keeps frames going out when the window is not being painted.
setInterval(() => {
  if (!S.running) return;
  send();
  if (S.kind !== "camera" && video.ended && !S.ended) end();
}, 50);

Object.assign(window, { checkout: S, checkoutDraw: draw }); // for poking at the state from the console
boot().catch((e) => note(`Could not load: ${e.message}`));
requestAnimationFrame(draw);

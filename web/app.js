// The shop floor, browser side.
//
// The page plays footage (a clip, an uploaded video or the camera), sends about three frames a second to the local
// server, and gets back who is in the picture and what each of them holds. Answers take about a second, so the
// picture is shown that much behind the footage: every outline is then drawn on the frame it was computed from,
// and moved smoothly between two analysed frames.

const $ = (id) => document.getElementById(id);
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
};

const video = $("video");
const canvas = $("view");
const ctx = canvas.getContext("2d");
const grab = document.createElement("canvas");

const colour = (pid) => PALETTE[(pid - 1) % PALETTE.length];
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const formats = {};
// "$249.00" for the shop's currency, or for the one a payment was made in
function money(n, code) {
  code = String(code || S.config?.catalog.currency || "USD").toUpperCase();
  try {
    formats[code] ??= new Intl.NumberFormat("en", { style: "currency", currency: code });
    return formats[code].format(Number(n || 0));
  } catch {
    return `${code} ${Number(n || 0).toFixed(2)}`;
  }
}
const TICK = `<svg class="mark" viewBox="0 0 12 12" aria-hidden="true"><path d="M2 6.5l2.6 2.6L10 3.4" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg>`;
const stamp = (t) => `${Math.floor(t / 60)}:${String(Math.floor(t % 60)).padStart(2, "0")}`;
const ordinal = (n) => `${n}${["th", "st", "nd", "rd"][n % 100 > 10 && n % 100 < 14 ? 0 : Math.min(n % 10, 4) % 4]}`;

function nameOf(pid) {
  return S.meta.people?.[pid]?.profile?.name ?? `Shopper ${pid}`;
}

// ---------------------------------------------------------------- setup

async function loadShop(shop) {
  S.config = await (await fetch(`/api/config?shop=${encodeURIComponent(shop ?? "")}`)).json();
  const name = S.config.shops.find((x) => x.id === S.config.shop)?.name ?? "Shop";
  $("brand").textContent = name;
  document.title = `${name}: shop floor`;
  $("shop").innerHTML = S.config.shops.map((x) => `<option value="${esc(x.id)}" ${x.id === S.config.shop ? "selected" : ""}>${esc(x.name)}</option>`).join("");
  fillSources();
  renderCatalog(S.config.catalog);
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
  await loadShop();
  renderProfiles(S.config.profiles);
  listCameras(); // names are there already if this browser has used a camera here before
  const sv = S.config.services;
  $("pay-note").textContent = sv.payments ? "Charged on the way out" : "Payments are off";
  const row = (on, text) => `<li><span class="dot ${on ? "ok" : "bad"}"></span><span>${esc(text)}</span></li>`;
  $("services").innerHTML =
    row(sv.faces, sv.faces ? (sv.face_search === "pgvector" ? "Faces are searched in the Postgres vector index" : "Faces are matched on this computer") : "Face recognition is off: the model files are missing") +
    row(sv.verifier, sv.verifier ? "Every bag is checked by the second model before it is charged" : "The second model is off, so bags are charged unchecked") +
    row(sv.payments, sv.payments ? "Payments go through the PayPal sandbox" : `Payments are off: ${sv.payments_note}`) +
    row(sv.email, sv.email ? "Invoices are emailed" : "Invoice email is off");
  wake();
}

async function wake() {
  const dot = $("model-dot");
  const text = $("model-text");
  dot.className = "dot";
  text.textContent = "Waking the cameras, up to a minute";
  try {
    const r = await fetch("/api/health");
    if (!r.ok) throw new Error((await r.json()).error);
    dot.className = "dot ok";
    text.textContent = "Cameras ready";
    return true;
  } catch (e) {
    dot.className = "dot bad";
    text.textContent = `The camera model is not answering: ${e.message}`;
    return false;
  }
}

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
  if (now - lastCapture >= 66 && video.readyState >= 2 && video.videoWidth) {
    lastCapture = now;
    const t = clock();
    const w = Math.min(VIEW_WIDTH, video.videoWidth);
    const h = Math.round((w * video.videoHeight) / video.videoWidth);
    createImageBitmap(video, { resizeWidth: w, resizeHeight: h, resizeQuality: "low" })
      .then((bmp) => {
        if (!S.running) return bmp.close();
        S.frames.push({ t, bmp });
        if (S.frames.length > 1 && S.frames.at(-2).t > t) S.frames.sort((a, b) => a.t - b.t);
        while (S.frames.length > 140) S.frames.shift().bmp.close();
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
    if (!(await wake())) throw new Error("the camera model is not reachable");
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
    S.running = true;
    $("viewer").classList.add("live");
    $("viewer-note").textContent = "";
    button.textContent = "Stop";
    capture();
  } catch (e) {
    note(`Could not start: ${e.message}`);
    stopMedia();
  }
  button.disabled = false;
}

function openSocket() {
  return new Promise((resolve, reject) => {
    if (S.ws) S.ws.close();
    const ws = new WebSocket(`ws://${location.host}/ws`);
    S.ws = ws;
    ws.onopen = () => ws.send(JSON.stringify({ type: "start", shop: S.config.shop, catalog: readCatalog(), dry: $("dry").checked }));
    ws.onerror = () => reject(new Error("no connection to the local server"));
    ws.onclose = () => {
      if (S.running && S.ws === ws) note("The connection to the local server closed.");
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
    meta: { people: {}, payments: {} }, cartsKey: "", eventsKey: "",
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

// Objects of state `a`, eased towards where state `b` has them. Things only in one of the two fade in or out.
function between(listA, listB, k) {
  const out = [];
  const inB = new Map((listB ?? []).map((o) => [o.id, o]));
  for (const o of listA) {
    const next = inB.get(o.id);
    if (next) {
      const box = lerpBox(o.box, next.box, k);
      out.push({ ...o, box, polygons: movePolys(o.polygons, o.box, box), alpha: 1 });
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
  if (i < 0) return;
  const a = S.states[i];
  const b = S.states[i + 1];
  const k = b ? Math.min(1, Math.max(0, (T - a.t) / Math.max(1e-3, b.t - a.t))) : 0;
  if (!b && T - a.t > 1.5 && !S.ended) return; // nothing fresh to show: better no outline than a stale one

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
    if (item.state === "held") chip(item.name, item.box[0] * W, item.box[3] * H + 6 * px, "rgba(22,21,19,0.82)", px);
  }

  ctx.globalAlpha = 1;
  const carts = new Map(S.carts.map((cart) => [cart.person, cart]));
  for (const p of people) {
    if (p.alpha < 0.5) continue;
    const cart = carts.get(p.id);
    const n = cart?.count ?? 0;
    const label = `${nameOf(p.id)}${n ? `  ${n} item${n > 1 ? "s" : ""}  ${money(cart.total)}` : ""}`;
    chip(label, p.box[0] * W, p.box[1] * H - 28 * px, colour(p.id), px);
  }

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
  // a pick that turned out to be an item already on the cart under another track: its events go
  const folded = new Set(S.events.filter((e) => e.type === "void").map((e) => e.entry));
  if (folded.size) S.events = S.events.filter((e) => e.type !== "void" && !(folded.has(e.entry) && ["pick", "verify"].includes(e.type)));
  if (msg.type === "finished") {
    S.finished = true;
    note(`The footage ended. Everyone still inside was treated as walking out.${S.meta.dry ? " Rehearsal: nobody was charged." : ""}`);
    refreshProfiles();
  }
  if (msg.events?.some((e) => e.type === "checkin" || e.type === "payment")) refreshProfiles();
  renderCarts();
  renderEvents();
}

// ---------------------------------------------------------------- carts

// What the bag says under a line: [text, "" | "took" | "flag", the longer story for a tooltip].
function lineNote(line) {
  if (line.state === "returned") return ["Put back, not charged", ""];
  if (line.state === "given") return ["Handed to another shopper", ""];
  if (line.product && !line.billable && !line.unread) return ["Their own, not for sale", ""];
  // the last check before charging has the final word on a line
  if (line.audit === "put_back") return [`Not charged. The last check saw it put back: ${line.audit_why}`, "flag"];
  if (line.audit === "unclear") return [`Not charged. The last check could not confirm it: ${line.audit_why}`, "flag"];
  if (line.audit === "kept") return [`Took it: ${line.audit_why}`, "took"];
  if (line.paid) return ["Paid", "took"];
  if (line.verified === "disputed") return [`Held back: the second model saw no item in the hand (${line.why})`, "flag"];
  if (line.unread) return ["Not identified, so not charged", "flag"];
  if (line.repeat) return ["Not charged: the same title as a line above (one book seen twice)", "flag"];
  if (line.billable && !line.charge && line.verified == null) return ["Held back: named from the shelf, not yet read in their hands", "flag"];
  if (!line.product) return ["Held back: not in the catalogue", "flag"];
  if (!line.charge) return [`Held back: only ${Math.round(line.confidence * 100)}% sure`, "flag"];
  // on the bill and still in the shop: one short line, with how it got there kept for the tooltip
  const how = { carried_in: "Was holding it on arrival", inferred: "It left the shelf under their hand", handover: "Handed over by another shopper", seen: "Not seen leaving the shelf" }[line.origin];
  const story = [how, line.why].filter(Boolean).join(". ");
  if (line.verified === "confirmed") return ["Seen in their hands", "", story];
  return [`${line.origin === "inferred" ? "Left the shelf under their hand, " : ""}${Math.round(line.confidence * 100)}% sure`, "", story];
}

// What happened at the door, in the order it happens: the last check, the charge, the invoice, the email.
function door(pay, who) {
  const step = (cls, html) => `<li class="${cls}">${cls === "done" ? TICK : ""}<span>${html}</span></li>`;
  if (!pay) return who.checking ? `<ol class="door">${step("wait", "Checking the bag against the pictures before charging")}</ol>` : "";
  const steps = [];
  const amount = money(pay.amount, pay.currency);
  const method = esc(pay.method || "the saved method");
  const check = pay.check ?? {};
  const seen = (check.kept ?? 0) + (check.held ?? 0);
  const shots = check.pictures
    ? `<span class="shots">${Array.from({ length: check.pictures }, (_, i) => {
        const src = `/api/evidence/${esc(check.session)}/${check.person}/${i}`;
        return `<a href="${src}" target="_blank" rel="noopener"><img src="${src}" alt="Picture ${String.fromCharCode(65 + i)} the check was shown" loading="lazy" /></a>`;
      }).join("")}</span>`
    : "";
  if (check.status === "checked")
    steps.push(step(check.held ? "stop" : "done", `Checked before charging: ${check.kept} of ${seen} item${seen === 1 ? "" : "s"} confirmed${check.held ? `, ${check.held} not charged` : ""}${shots}`));
  else if (check.status === "failed") steps.push(step("stop", "The check before charging got no answer, so nothing was charged"));
  switch (pay.status) {
    case "succeeded":
      steps.push(step("done", `Paid ${amount} with ${method}`));
      break;
    case "dry":
      steps.push(step("wait", `Rehearsal: ${amount} would be charged to ${method}`));
      break;
    case "nothing_due":
      steps.push(step("wait", "Walked out with nothing to pay"));
      break;
    case "not_registered":
      steps.push(step("stop", `Not registered at the gate, so the ${amount} of items was not charged`));
      break;
    case "declined":
      steps.push(step("bad", `${method} was declined. ${amount} is unpaid`));
      break;
    case "no_card":
      steps.push(step("bad", `Walked out with ${amount} of items and no payment method on file`));
      break;
    default:
      steps.push(step("bad", `Not charged (${amount}): ${esc(pay.message)}`));
  }
  if (pay.status === "succeeded") {
    const sv = S.config.services;
    const inv = pay.invoice;
    if (inv?.status === "created")
      steps.push(step("done", `Invoice ${esc(inv.number ?? "")} written. <a href="${esc(inv.url)}" target="_blank" rel="noopener">View it on PayPal</a>`));
    else if (inv) steps.push(step("bad", `The PayPal invoice was not written: ${esc(inv.message)}`));
    else if (sv.provider === "paypal" && !S.finished) steps.push(step("wait", "Writing the PayPal invoice"));
    if (pay.email) steps.push(pay.email.status === "sent" ? step("done", `Invoice emailed to ${esc(pay.email.to)}`) : step("bad", `The invoice email was not sent: ${esc(pay.email.message)}`));
    else if (sv.email && who.profile?.email && !S.finished) steps.push(step("wait", `Emailing the invoice to ${esc(who.profile.email)}`));
  }
  return `<ol class="door">${steps.join("")}</ol>`;
}

function renderCarts() {
  const key = JSON.stringify([S.carts, S.meta, S.finished]);
  if (key === S.cartsKey) return;
  S.cartsKey = key;
  const box = $("carts");
  if (!S.carts.length) {
    box.innerHTML = `<p class="empty">Nobody yet. Shoppers appear here with their bag once the gate has let them in.</p>`;
    return;
  }
  box.innerHTML = S.carts
    .map((cart) => {
      const pid = cart.person;
      const who = S.meta.people?.[pid] ?? { status: "no_face" };
      const p = who.profile;
      const left = cart.state === "left";
      const face = p?.has_face
        ? `<img class="face" style="--c:${colour(pid)}" src="/api/face/${esc(p.id)}" alt="" width="44" height="44" />`
        : `<span class="face none" style="--c:${colour(pid)}" aria-hidden="true">${pid}</span>`;
      let sub = "";
      let notice = "";
      if (p) {
        const nth = p.visits + (left ? 0 : 1);
        sub = `<span class="sub">${esc(p.method || "No payment method on file")}</span><span class="sub">${nth > 1 ? `${ordinal(nth)} visit` : "First visit"}${p.recent?.length && !left ? `. Last time: ${esc(p.recent.join(", "))}` : ""}</span>`;
      } else if (who.status === "not_registered") {
        sub = `<span class="sub">Not registered</span>`;
        if (!left)
          notice = `<p class="notice">${who.name ? `Looks like ${esc(who.name)}, who has not come through the gate this time.` : "This shopper has not registered at the gate."} Nothing will be charged to them. <a href="/gate" target="_blank" rel="noopener">Open the gate</a></p>`;
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
      return `<article class="bag">
        <header class="bag-head">${face}<div class="who"><strong>${esc(nameOf(pid))}</strong>${sub}</div><span class="state">${left ? "Left" : "In the shop"}</span></header>
        ${notice}
        ${lines ? `<ul class="lines">${lines}</ul>` : `<p class="nothing">Nothing picked up.</p>`}
        ${cart.lines.length ? `<div class="total"><span>${totalLabel}</span><span class="price">${money(cart.total)}</span></div>` : ""}
        ${door(S.meta.payments?.[pid], who)}
      </article>`;
    })
    .join("");
}

// ---------------------------------------------------------------- events

function describe(e) {
  const who = esc(nameOf(e.person));
  const line = S.carts.flatMap((c) => c.lines).find((l) => l.id === e.entry);
  const item = line?.unread ? `a ${esc(line.name.toLowerCase())} (not yet told which)` : esc(line?.name ?? "an item");
  const own = line?.product && !line.billable && !line.unread;
  switch (e.type) {
    case "enter":
      return [`${who} walked in`, true];
    case "reenter":
      return [`${who} came back`, true];
    case "checkin":
      return [`Recognised ${esc(e.name)}${e.score ? ` (${Math.round(e.score * 100)}% face match)` : ""}, let in at the gate. ${e.method ? `${esc(e.method)} is on file` : "No payment method on file"}`, false];
    case "unregistered":
      return [e.name ? `${who} looks like ${esc(e.name)}, who has not come through the gate. Nothing will be charged` : `${who} is not registered at the gate. Nothing will be charged`, false];
    case "audit":
      if (e.verdict === "kept") return [`Last check: ${who} took ${item} (${esc(e.why)})`, true];
      if (e.verdict === "put_back") return [`Last check: ${item} was put back, so it is not charged (${esc(e.why)})`, false];
      return [`Last check could not confirm ${item}, so it is not charged (${esc(e.why)})`, false];
    case "invoice":
      return e.status === "created" ? [`PayPal invoice ${esc(e.number ?? "")} written for ${who}`, true] : [`The PayPal invoice for ${who} was not written: ${esc(e.message)}`, false];
    case "pick":
      if (own) return [`${who} is holding their own ${item.toLowerCase()}`, true];
      if (e.origin === "carried_in") return [`${who} arrived holding ${item}`, false];
      if (e.origin === "inferred") return [`${who} took ${item} (it left the shelf under their hand)`, false];
      if (e.origin === "seen") return [`${who} is carrying ${item}`, false];
      return [`${who} took ${item}`, false];
    case "return":
      return [`${who} put ${item} back`, false];
    case "handover":
      return [`${esc(nameOf(e.from_person))} handed ${item} to ${who}`, false];
    case "verify":
      if (e.verdict === "disputed") return [`Second model disagrees on ${item}: ${esc(e.why)}`, false];
      if (e.verdict === "confirmed") return [`Verified as ${item}: ${esc(e.why)}`, true];
      return [`Not settled which item this is yet: ${esc(e.why)}`, true];
    case "exit":
      return [`${who} walked out`, true];
    case "payment":
      if (e.status === "succeeded") return [`Charged ${money(e.amount, e.currency)} to ${who}'s ${esc(e.method || "saved method")}`, false];
      if (e.status === "nothing_due") return [`${who} had nothing to pay`, true];
      if (e.status === "dry") return [`Rehearsal: ${who} would be charged ${money(e.amount, e.currency)}`, false];
      if (e.status === "not_registered") return [`${who} left with ${money(e.amount, e.currency)} of items and is not registered, so nothing was charged`, false];
      if (e.status === "declined") return [`${who}'s ${esc(e.method || "payment method")} was declined for ${money(e.amount, e.currency)}`, false];
      if (e.status === "no_card") return [`${who} left with ${money(e.amount, e.currency)} of items and no payment method on file`, false];
      return [`${who} was not charged: ${esc(e.message)}`, false];
    case "email":
      return [e.status === "sent" ? `Invoice emailed to ${esc(e.to)}` : `The invoice email to ${esc(e.to)} was not sent: ${esc(e.message)}`, e.status === "sent"];
    default:
      return [esc(e.type), true];
  }
}

const RECENT = 8; // the list shows this many until "Show all" is pressed

function renderEvents() {
  const key = JSON.stringify([S.events, S.carts.map((c) => c.lines.map((l) => l.name)), Object.keys(S.meta.people ?? {}).map((p) => nameOf(p))]);
  if (key === S.eventsKey) return;
  const had = new Set([...$("events").querySelectorAll("li[data-id]")].map((li) => li.dataset.id));
  S.eventsKey = key;
  const more = $("more-events");
  more.hidden = S.events.length <= RECENT;
  more.textContent = $("events").classList.contains("all") ? "Show the latest" : `Show all ${S.events.length}`;
  if (!S.events.length) {
    $("events").innerHTML = `<li class="empty">Nothing yet. What shoppers take, put back and pay for is listed here.</li>`;
    return;
  }
  $("events").innerHTML = [...S.events]
    .sort((a, b) => b.id - a.id)
    .map((e, i) => {
      const [text, minor] = describe(e);
      const cls = [had.has(String(e.id)) ? "" : "new", i >= RECENT ? "older" : ""].join(" ").trim();
      return `<li data-id="${e.id}" class="${cls}"><time>${stamp(e.t)}</time><span class="swatch" style="background:${colour(e.person)}"></span><span class="${minor ? "minor" : ""}">${text}</span></li>`;
    })
    .join("");
}

$("more-events").onclick = () => {
  $("events").classList.toggle("all");
  S.eventsKey = "";
  renderEvents();
};

// ---------------------------------------------------------------- profiles

async function refreshProfiles() {
  renderProfiles((await (await fetch("/api/profiles")).json()).profiles);
}

function renderProfiles(list) {
  $("profile-count").textContent = list.length ? `Registered shoppers (${list.length})` : "Registered shoppers";
  $("profile-list").innerHTML = list.length
    ? list
        .map(
          (p) => `<li>${p.has_face ? `<img class="face" src="/api/face/${esc(p.id)}" alt="" width="44" height="44" />` : `<span class="face none" aria-hidden="true">?</span>`}
            <div class="who"><strong>${esc(p.name)}</strong><span class="sub">${esc(p.method || "No payment method PayPal can charge")}${p.inside ? ". In the shop now" : ""}</span>
            <span class="sub">${esc(p.email ?? "")}</span>
            <span class="sub">${p.visits} visit${p.visits === 1 ? "" : "s"}, ${money(p.spent)} spent${p.unpaid ? `, ${money(p.unpaid)} unpaid` : ""}</span></div></li>`,
        )
        .join("")
    : `<li class="empty">Nobody has registered yet. Shoppers register at the gate.</li>`;
  $("forget").hidden = !list.length;
}

$("forget").addEventListener("click", async () => {
  if (!confirm("Delete every shopper: faces, payment links and visit history? This cannot be undone.")) return;
  await fetch("/api/profiles", { method: "DELETE" });
  refreshProfiles();
});

$("open-settings").onclick = () => {
  refreshProfiles();
  $("settings").showModal();
};
$("close-settings").onclick = () => $("settings").close();
$("settings").addEventListener("click", (e) => {
  if (e.target === $("settings")) $("settings").close(); // a click on the backdrop
});

// ---------------------------------------------------------------- catalogue

function productRow(p = { name: "", prompt: "", price: "", category: "", section: "", author: "" }) {
  const tr = document.createElement("tr");
  if (p.id) tr.dataset.id = p.id;
  const cell = (label, value) => `<td><input type="text" aria-label="${label}" value="${esc(value ?? "")}" /></td>`;
  // the rest of what is known about it, read-only: format, publisher, what the spine looks like
  const more = [...Object.values(p.meta ?? {}), p.cues].filter(Boolean).join(", ");
  tr.innerHTML =
    `<td><input type="text" aria-label="Name" value="${esc(p.name ?? "")}" />${more ? `<small>${esc(more)}</small>` : ""}</td>` +
    cell("Author or maker", p.author) + cell("Category", p.category) + cell("Section", p.section) + cell("What the camera looks for", p.prompt) +
    `<td><input type="number" aria-label="Price" class="num" min="0" step="0.5" value="${esc(p.price)}" /></td><td><button type="button" aria-label="Remove ${esc(p.name || "product")}">×</button></td>`;
  tr.querySelector("button").onclick = () => {
    tr.remove();
    countPrompts();
  };
  return tr;
}

// Rows grouped under "Category › Section" headings, in the order the sections first appear.
function renderCatalog(catalog) {
  const rows = [];
  let last = null;
  const where = (p) => [p.category, p.section].filter(Boolean).join(", ");
  const order = [...new Set(catalog.products.map(where))];
  for (const p of [...catalog.products].sort((a, b) => order.indexOf(where(a)) - order.indexOf(where(b)))) {
    if (where(p) !== last) {
      last = where(p);
      const n = catalog.products.filter((x) => where(x) === last).length;
      const head = document.createElement("tr");
      head.className = "group";
      head.innerHTML = `<th colspan="7" scope="colgroup">${esc(last || "No category")} <span>${n} product${n === 1 ? "" : "s"}</span></th>`;
      rows.push(head);
    }
    rows.push(productRow(p));
  }
  $("products").replaceChildren(...rows);
  $("catalog-title").textContent = `Catalogue: ${S.config?.shops.find((x) => x.id === S.config.shop)?.name ?? catalog.name}`;
  $("belongings").value = catalog.belongings.join(", ");
  countPrompts();
}

function readCatalog() {
  const products = [...$("products").querySelectorAll("tr:not(.group)")]
    .map((tr, i) => {
      const [name, author, category, section, prompt, price] = [...tr.querySelectorAll("input")].map((el) => el.value.trim());
      const old = S.config.catalog.products.find((p) => p.id === tr.dataset.id) ?? {};
      // with no phrase given, the camera looks for the product by its name
      return { ...old, id: tr.dataset.id || `p${Date.now().toString(36)}${i}`, name: name || prompt, author, category, section, prompt: prompt || name.toLowerCase(), price: Number(price) || 0 };
    })
    .filter((p) => p.prompt);
  const belongings = $("belongings").value.split(",").map((s) => s.trim()).filter(Boolean);
  return { ...S.config.catalog, products, belongings };
}

function countPrompts() {
  const c = readCatalog();
  const n = 3 + new Set(c.products.map((p) => p.prompt.toLowerCase())).size + c.belongings.length;
  $("prompt-count").textContent =
    n <= S.config.max_prompts
      ? `${n} phrases a frame (person, hand and "item held in hand" are always asked). Products that share a phrase, like books, are told apart by reading them.`
      : `${n} phrases a frame. More than ${S.config.max_prompts} need a second request for every frame, which halves the frame rate.`;
}

// A new product starts at the top of the list, where it can be seen and typed into straight away.
$("add-product").onclick = () => {
  const row = productRow();
  $("products").prepend(row);
  row.querySelector("input").focus();
  $("catalog-note").textContent = "Fill in the new product, then save the catalogue.";
};
$("catalog").addEventListener("input", countPrompts);
$("save-catalog").onclick = async () => {
  const r = await fetch(`/api/catalog/${encodeURIComponent(S.config.shop)}`, { method: "PUT", headers: { "content-type": "application/json" }, body: JSON.stringify(readCatalog()) });
  const body = await r.json();
  if (r.ok) {
    S.config.catalog = body.catalog;
    renderCatalog(body.catalog); // regroup under the headings
  }
  $("catalog-note").textContent = r.ok ? "Saved. It applies the next time you press Start." : body.error;
};

// ---------------------------------------------------------------- controls

$("start").onclick = () => (S.running && !S.ended ? end() : start());
$("shop").onchange = async () => {
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

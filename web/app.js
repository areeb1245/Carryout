// Walk-out checkout, browser side.
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
  prompted: new Set(),
  dialogFor: null,
  cartsKey: "",
  eventsKey: "",
};

const video = $("video");
const canvas = $("view");
const ctx = canvas.getContext("2d");
const grab = document.createElement("canvas");

const colour = (pid) => PALETTE[(pid - 1) % PALETTE.length];
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const money = (n) => `${S.config?.catalog.currency ?? ""} ${Number(n || 0).toFixed(2)}`;
const cardName = (p) => (p?.last4 ? `${p.brand[0].toUpperCase()}${p.brand.slice(1)} ending ${p.last4}` : "no card");
const stamp = (t) => `${Math.floor(t / 60)}:${String(Math.floor(t % 60)).padStart(2, "0")}`;
const ordinal = (n) => `${n}${["th", "st", "nd", "rd"][n % 100 > 10 && n % 100 < 14 ? 0 : Math.min(n % 10, 4) % 4]}`;

function nameOf(pid) {
  return S.meta.people?.[pid]?.profile?.name ?? `Shopper ${pid}`;
}

// ---------------------------------------------------------------- setup

async function loadShop(shop) {
  S.config = await (await fetch(`/api/config?shop=${encodeURIComponent(shop ?? "")}`)).json();
  $("shop").innerHTML = S.config.shops.map((x) => `<option value="${esc(x.id)}" ${x.id === S.config.shop ? "selected" : ""}>${esc(x.name)}</option>`).join("");
  $("source").innerHTML =
    S.config.samples.filter((x) => x.shop === S.config.shop).map((x) => `<option value="sample:${esc(x.file)}">${esc(x.title)}</option>`).join("") +
    `<option value="camera">This computer's camera</option><option value="file">A video file from this computer…</option>`;
  renderCatalog(S.config.catalog);
}

async function boot() {
  await loadShop();
  $("checkin-card").innerHTML = Object.entries(S.config.test_cards)
    .map(([id, label]) => `<option value="${esc(id)}">${esc(label)}</option>`)
    .join("");
  renderProfiles(S.config.profiles);
  const sv = S.config.services;
  $("pay-note").textContent = sv.payments
    ? `Cards are charged on Stripe's sandbox when a shopper walks out.${sv.email ? " The invoice goes to the email on their profile." : ""}`
    : `Payments are off: ${sv.payments_note}.`;
  wake();
}

async function wake() {
  const dot = $("model-dot");
  const text = $("model-text");
  text.textContent = "Waking the GPU, up to a minute";
  try {
    const r = await fetch("/api/health");
    if (!r.ok) throw new Error((await r.json()).error);
    dot.className = "dot ok";
    text.textContent = "SAM 3.1 is ready";
    return true;
  } catch (e) {
    dot.className = "dot bad";
    text.textContent = `SAM 3.1 is not answering: ${e.message}`;
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
    if (!(await wake())) throw new Error("the model endpoint is not reachable");
    if (choice === "camera") {
      S.kind = "camera";
      S.stream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false });
      video.srcObject = S.stream;
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
    ws.onopen = () => ws.send(JSON.stringify({ type: "start", shop: S.config.shop, catalog: readCatalog(), auto_checkin: $("auto-checkin").checked }));
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
      if (msg.type === "refused") return note(msg.message);
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
    meta: { people: {}, payments: {} }, prompted: new Set(), cartsKey: "", eventsKey: "",
  });
  if ($("checkin").open) $("checkin").close("cancel");
  renderCarts();
  renderEvents();
  $("meter").innerHTML = "&nbsp;";
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
  ctx.font = `500 ${10 * px}px ui-sans-serif, system-ui, sans-serif`;
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
  ctx.font = `500 ${13 * px}px ui-sans-serif, system-ui, sans-serif`;
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
    const label = `${nameOf(p.id)}${n ? ` · ${n} item${n > 1 ? "s" : ""} · ${money(cart.total)}` : ""}`;
    chip(label, p.box[0] * W, p.box[1] * H - 28 * px, colour(p.id), px);
  }

  const sent = S.sentTimes.filter((t) => performance.now() - t < 5000);
  S.sentTimes = sent;
  if (!S.ended)
    $("meter").textContent =
      `Analysing ${(sent.length / 5).toFixed(1)} frames a second · picture runs ${S.delay.toFixed(1)} s behind the footage${S.modelMs ? ` · SAM ${S.modelMs} ms a frame` : ""}` +
      (S.meta.shelf ? ` · ${S.meta.shelf.read} of ${S.meta.shelf.items} items on the shelf identified` : "");
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
    note("The footage ended. Everyone still inside was treated as walking out.");
    refreshProfiles();
  }
  if (msg.events?.some((e) => e.type === "checkin" || e.type === "payment")) refreshProfiles();
  renderCarts();
  renderEvents();
  askForCards();
}

// ---------------------------------------------------------------- carts

function lineNote(line) {
  if (line.state === "returned") return ["Put back, not charged", ""];
  if (line.state === "given") return ["Handed to another shopper", ""];
  if (line.product && !line.billable && !line.unread) return ["Their own, not for sale", ""];
  if (line.paid) return ["Paid", ""];
  if (line.verified === "disputed") return [`Held back: the second model saw no item in the hand (${line.why})`, "flag"];
  if (line.unread) return ["Not identified, so not charged", "flag"];
  if (line.repeat) return ["Not charged: the same title as a line above (one book seen twice)", "flag"];
  if (line.billable && !line.charge && line.verified == null) return ["Held back: named from the shelf, not yet read in their hands", "flag"];
  if (!line.product) return ["Held back: not in the catalogue", "flag"];
  if (!line.charge) return [`Held back: only ${Math.round(line.confidence * 100)}% sure`, "flag"];
  const parts = [];
  if (line.origin === "carried_in") parts.push("Was holding it on arrival");
  if (line.origin === "inferred") parts.push("Not seen in the hand: it left the shelf under their hand");
  if (line.origin === "handover") parts.push("Handed over by another shopper");
  if (line.origin === "seen") parts.push("Not seen leaving the shelf");
  parts.push(line.verified === "confirmed" ? `Verified: ${line.why || "the second model agrees"}` : `${Math.round(line.confidence * 100)}% sure`);
  return [parts.join(" · "), ""];
}

function outcome(pay) {
  if (!pay) return "";
  const amount = `${pay.currency.toUpperCase()} ${pay.amount.toFixed(2)}`;
  const mail = pay.email ? (pay.email.status === "sent" ? ` Receipt emailed to ${esc(pay.email.to)}.` : ` The receipt email was not sent: ${esc(pay.email.message)}`) : "";
  switch (pay.status) {
    case "succeeded":
      return `<p class="outcome ok">Charged ${amount} to ${esc(cardName(pay))}.${pay.receipt_url ? ` <a href="${esc(pay.receipt_url)}" target="_blank" rel="noopener">Stripe receipt</a>.` : ""}${mail}</p>`;
    case "declined":
      return `<p class="outcome bad">${esc(cardName(pay))} was declined. ${amount} is unpaid.</p>`;
    case "no_card":
      return `<p class="outcome bad">Walked out with ${amount} of items and no card on file.</p>`;
    case "nothing_due":
      return `<p class="outcome">Walked out with nothing to pay.</p>`;
    default:
      return `<p class="outcome bad">Not charged (${amount}): ${esc(pay.message)}</p>`;
  }
}

function renderCarts() {
  const key = JSON.stringify([S.carts, S.meta]);
  if (key === S.cartsKey) return;
  S.cartsKey = key;
  const box = $("carts");
  if (!S.carts.length) {
    box.innerHTML = `<p class="empty">No shoppers in view.</p>`;
    return;
  }
  box.innerHTML = S.carts
    .map((cart) => {
      const pid = cart.person;
      const who = S.meta.people?.[pid] ?? { status: "no_face" };
      const p = who.profile;
      const face = p?.has_face
        ? `<img class="face" style="--c:${colour(pid)}" src="/api/face/${esc(p.id)}" alt="" width="40" height="40" />`
        : `<span class="face none" style="--c:${colour(pid)}" aria-hidden="true">${pid}</span>`;
      let sub;
      if (p) {
        const history = p.visits ? `${ordinal(p.visits + (cart.state === "left" ? 0 : 1))} visit · ${money(p.spent)} spent` : "First visit";
        sub = `<span class="sub">${esc(cardName(p))} · ${history}</span>${p.recent?.length && cart.state !== "left" ? `<span class="sub">Last time: ${esc(p.recent.join(", "))}</span>` : ""}${who.note ? `<span class="sub flag">${esc(who.note)}</span>` : ""}`;
      } else if (who.status === "checking_in") sub = `<span class="sub">Saving the card…</span>`;
      else if (cart.state === "left") sub = `<span class="sub">Never checked in</span>`;
      else
        sub = `<span class="sub">${who.status === "new_face" ? "New face, no profile yet." : "Face not seen yet."} <button type="button" class="quiet" data-checkin="${pid}">Add a card</button></span>`;
      const lines = cart.lines
        .map((line) => {
          const [text, flag] = lineNote(line);
          const cls = line.state !== "in_cart" ? "gone" : line.product && !line.billable && !line.unread ? "own" : "";
          const shelf = [line.category, line.section].filter(Boolean).join(" · ");
          return `<li class="${cls}"><span class="name">${esc(line.name)}${shelf ? `<small>${esc(shelf)}</small>` : ""}</span><span class="price">${line.price == null || !line.billable ? "" : line.price.toFixed(2)}</span><span class="note ${flag}">${esc(text)}</span></li>`;
        })
        .join("");
      return `<article class="cart">
        <header class="cart-head">${face}<div class="who"><strong>${esc(nameOf(pid))}</strong>${sub}</div><span class="state">${cart.state === "left" ? "Left" : "In the shop"}</span></header>
        ${lines ? `<ul class="lines">${lines}</ul>` : `<p class="nothing">Nothing picked up.</p>`}
        ${cart.lines.length ? `<div class="total"><span>${cart.state === "left" ? "Total" : "Will be charged on the way out"}</span><span class="price">${money(cart.total)}</span></div>` : ""}
        ${outcome(S.meta.payments?.[pid])}
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
      return e.how === "recognised"
        ? [`Recognised ${esc(e.name)} by face (${Math.round(e.score * 100)}% match). ${esc(cardName(e))} is on file`, false]
        : [`${esc(e.name)} checked in${e.face ? ", face saved" : " without a visible face"}. ${e.last4 ? `${esc(cardName(e))} is on file` : "No card saved"}`, false];
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
      if (e.status === "succeeded") return [`Charged ${e.currency.toUpperCase()} ${e.amount.toFixed(2)} to ${who}'s ${esc(cardName(e))}`, false];
      if (e.status === "nothing_due") return [`${who} had nothing to pay`, true];
      if (e.status === "declined") return [`${who}'s card was declined for ${e.currency.toUpperCase()} ${e.amount.toFixed(2)}`, false];
      if (e.status === "no_card") return [`${who} left with ${e.currency.toUpperCase()} ${e.amount.toFixed(2)} of items and no card on file`, false];
      return [`${who} was not charged: ${esc(e.message)}`, false];
    case "email":
      return [e.status === "sent" ? `Receipt emailed to ${esc(e.to)}` : `Receipt email to ${esc(e.to)} was not sent: ${esc(e.message)}`, e.status === "sent"];
    default:
      return [esc(e.type), true];
  }
}

function renderEvents() {
  const key = JSON.stringify([S.events, S.carts.map((c) => c.lines.map((l) => l.name)), Object.keys(S.meta.people ?? {}).map((p) => nameOf(p))]);
  if (key === S.eventsKey) return;
  const had = new Set([...$("events").querySelectorAll("li[data-id]")].map((li) => li.dataset.id));
  S.eventsKey = key;
  if (!S.events.length) {
    $("events").innerHTML = `<li class="empty">Nothing yet. Picks, returns and payments are listed here as they happen.</li>`;
    return;
  }
  $("events").innerHTML = [...S.events]
    .sort((a, b) => b.id - a.id)
    .map((e) => {
      const [text, minor] = describe(e);
      const sure = e.confidence != null && (e.type === "pick" || e.type === "return") ? `${Math.round(e.confidence * 100)}%` : "";
      return `<li data-id="${e.id}" class="${had.has(String(e.id)) ? "" : "new"}"><time>${stamp(e.t)}</time><span class="swatch" style="background:${colour(e.person)}"></span><span class="${minor ? "minor" : ""}">${text}</span><span class="sure">${sure}</span></li>`;
    })
    .join("");
}

// ---------------------------------------------------------------- check-in

function askForCards() {
  if ($("auto-checkin").checked || $("checkin").open || S.ended) return;
  for (const [pid, who] of Object.entries(S.meta.people ?? {})) {
    if (who.status === "new_face" && !S.prompted.has(pid) && S.carts.find((c) => String(c.person) === pid)?.state === "shopping") {
      S.prompted.add(pid);
      return openCheckin(Number(pid));
    }
  }
}

function openCheckin(pid) {
  S.dialogFor = pid;
  S.prompted.add(String(pid));
  $("checkin-title").textContent = S.meta.people?.[pid]?.status === "new_face" ? `New face: Shopper ${pid}` : `Check in Shopper ${pid}`;
  $("checkin-name").value = "";
  $("checkin-email").value = S.config.services.demo_email || "";
  $("checkin").showModal();
}

$("checkin").addEventListener("close", () => {
  if ($("checkin").returnValue !== "ok" || S.dialogFor == null || S.ws?.readyState !== 1) return;
  S.ws.send(JSON.stringify({ type: "checkin", person: S.dialogFor, name: $("checkin-name").value, email: $("checkin-email").value, card: $("checkin-card").value }));
});

$("carts").addEventListener("click", (e) => {
  const pid = e.target.closest("[data-checkin]")?.dataset.checkin;
  if (pid) openCheckin(Number(pid));
});

// ---------------------------------------------------------------- profiles

async function refreshProfiles() {
  renderProfiles((await (await fetch("/api/profiles")).json()).profiles);
}

function renderProfiles(list) {
  $("profile-count").textContent = list.length ? `Profiles (${list.length})` : "Profiles";
  $("profile-list").innerHTML = list.length
    ? list
        .map(
          (p) => `<li>${p.has_face ? `<img class="face" src="/api/face/${esc(p.id)}" alt="" width="40" height="40" />` : `<span class="face none" aria-hidden="true">?</span>`}
            <div class="who"><strong>${esc(p.name)}</strong><span class="sub">${esc(cardName(p))}${p.email ? ` · ${esc(p.email)}` : ""}</span>
            <span class="sub">${p.visits} visit${p.visits === 1 ? "" : "s"} · ${money(p.spent)} spent${p.unpaid ? ` · ${money(p.unpaid)} unpaid` : ""}</span></div></li>`,
        )
        .join("")
    : `<li class="empty">Nobody has checked in yet.</li>`;
  $("forget").hidden = !list.length;
}

$("forget").addEventListener("click", async () => {
  if (!confirm("Delete every profile on this computer: faces, card links and visit history? This cannot be undone.")) return;
  await fetch("/api/profiles", { method: "DELETE" });
  refreshProfiles();
});

// ---------------------------------------------------------------- catalogue

function productRow(p = { name: "", prompt: "", price: "", category: "", section: "", author: "" }) {
  const tr = document.createElement("tr");
  if (p.id) tr.dataset.id = p.id;
  const cell = (label, value) => `<td><input type="text" aria-label="${label}" value="${esc(value ?? "")}" /></td>`;
  // the rest of what is known about it, read-only: format, publisher, what the spine looks like
  const more = [...Object.values(p.meta ?? {}), p.cues].filter(Boolean).join(" · ");
  tr.innerHTML =
    `<td><input type="text" aria-label="Name" value="${esc(p.name ?? "")}" />${more ? `<small>${esc(more)}</small>` : ""}</td>` +
    cell("Author or maker", p.author) + cell("Category", p.category) + cell("Section", p.section) + cell("What SAM looks for", p.prompt) +
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
  const where = (p) => [p.category, p.section].filter(Boolean).join(" › ");
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
  $("belongings").value = catalog.belongings.join(", ");
  countPrompts();
}

function readCatalog() {
  const products = [...$("products").querySelectorAll("tr:not(.group)")]
    .map((tr, i) => {
      const [name, author, category, section, prompt, price] = [...tr.querySelectorAll("input")].map((el) => el.value.trim());
      const old = S.config.catalog.products.find((p) => p.id === tr.dataset.id) ?? {};
      return { ...old, id: tr.dataset.id || `p${Date.now().toString(36)}${i}`, name: name || prompt, author, category, section, prompt, price: Number(price) || 0 };
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

$("add-product").onclick = () => {
  $("products").append(productRow());
  $("products").lastElementChild.querySelector("input").focus();
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
  if ($("source").value === "file") $("file").click();
};
$("file").onchange = () => {
  const file = $("file").files[0];
  if (!file) return;
  if (S.fileUrl) URL.revokeObjectURL(S.fileUrl);
  S.fileUrl = URL.createObjectURL(file);
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

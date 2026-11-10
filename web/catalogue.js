// The catalogue screen: the products the camera looks for, with their pictures, and footage to pick new ones out of.
//
// Products     one row per product, grouped by category and section. Pictures are cut out of footage (or chosen
//              from a file) and are shown to the second model and compared with what shoppers pick up.
// Hold to camera  the product is held up to a camera and turned; a frame goes to SAM about every 0.7 s, and the frames
//              it clearly sees an item in hand on are kept. The server keeps the four views that differ most as
//              pictures, asks the vision model what it is and tries its phrases: the form is filled, bar the price.
// From footage a clip becomes frames (server side). SAM outlines what it finds on each one; the person picks an
//              outline, or drags a box, and asks the vision model what it is. The phrases it suggests are then
//              tried on that frame and its neighbours: the phrase whose outline covers the object is the one to use.

import { brand, currentShop, mountShell, rememberShop } from "/static/shell.js";
import { esc } from "/static/words.js";

const $ = (id) => document.getElementById(id);
mountShell("catalogue");

const GENERIC = ["item held in hand", "product", "package", "bottle", "box"];
const PICTURE_SIDE = 384;
const C = {
  config: null,
  dirty: false,
  clip: null, // {clip, title, frames, times, width, height}
  n: 0,
  images: new Map(), // frame number -> loaded <img>
  answers: new Map(), // frame number -> detections
  looking: new Set(),
  like: null, // {frame, detections} from "find more like it"
  sel: null, // {frame, box, polygons, label, score}
  drag: null,
  described: null,
  phrases: [],
  ranked: null,
  chosen: "",
  views: [], // the pictures a camera capture keeps: [{frame, box}]
  live: null, // a camera capture in progress: {clip, stream, kept, timer, latest, t0}
};
const CAPTURE_EVERY_MS = 700;
const CAPTURE_VIEWS = 10; // clear views kept before the capture stops by itself
const CAPTURE_FOR_MS = 30000;
const SURE_HELD = 0.45;

const shopName = () => C.config.shops.find((x) => x.id === C.config.shop)?.name ?? "Shop";
const plural = (n, word) => `${n} ${word}${n === 1 ? "" : "s"}`;

// ---------------------------------------------------------------- tabs

function route() {
  const tab = location.hash === "#footage" ? "footage" : "products";
  for (const link of document.querySelectorAll(".tabs a")) link.toggleAttribute("aria-current", link.dataset.tab === tab);
  $("products-view").hidden = tab !== "products";
  $("footage-view").hidden = tab !== "footage";
  if (tab !== "footage" && C.live) stopCapture();
  if (tab === "footage" && !C.clip) loadClips();
}
window.addEventListener("hashchange", route);

// ---------------------------------------------------------------- products

async function loadShop(shop) {
  C.config = await (await fetch(`/api/config?shop=${encodeURIComponent(shop ?? "")}`)).json();
  rememberShop(C.config.shop);
  brand(shopName(), "catalogue");
  $("shop").innerHTML = C.config.shops.map((x) => `<option value="${esc(x.id)}" ${x.id === C.config.shop ? "selected" : ""}>${esc(x.name)}</option>`).join("");
  render(C.config.catalog);
  setDirty(false);
}

const pictureUrl = (file) => `/api/examples/${encodeURIComponent(C.config.shop)}/${encodeURIComponent(file)}`;

function thumbs(tr) {
  const cell = tr.querySelector(".pics");
  cell.innerHTML =
    tr.examples.map((f, i) => `<button type="button" class="thumb" data-i="${i}" title="Remove this picture"><img src="${pictureUrl(f)}" alt="" width="40" height="40" /></button>`).join("") +
    (tr.examples.length < 4 ? `<button type="button" class="thumb add" title="Add a picture from a file" aria-label="Add a picture">+</button>` : "");
  cell.querySelectorAll(".thumb[data-i]").forEach((b) => {
    b.onclick = () => {
      tr.examples.splice(Number(b.dataset.i), 1);
      thumbs(tr);
      setDirty(true);
    };
  });
  const add = cell.querySelector(".thumb.add");
  if (add)
    add.onclick = () => {
      C.pictureFor = tr;
      $("picture-file").click();
    };
}

function productRow(p = { name: "", prompt: "", price: "", category: "", section: "", author: "" }) {
  const tr = document.createElement("tr");
  if (p.id) tr.dataset.id = p.id;
  tr.examples = [...(p.examples ?? [])];
  const cell = (label, value, cls = "") => `<td><input type="text" aria-label="${label}" class="${cls}" value="${esc(value ?? "")}" /></td>`;
  const more = [...Object.values(p.meta ?? {}), p.cues].filter(Boolean).join(", ");
  tr.innerHTML =
    `<td class="pics"></td>` +
    `<td><input type="text" aria-label="Name" value="${esc(p.name ?? "")}" />${more ? `<small>${esc(more)}</small>` : ""}</td>` +
    cell("Maker", p.author) + cell("Category", p.category) + cell("Section", p.section) + cell("Camera phrase", p.prompt, "phrase") +
    `<td><input type="number" aria-label="Price" class="num" min="0" step="0.5" value="${esc(p.price)}" /></td>` +
    `<td><button type="button" class="icon" aria-label="Remove ${esc(p.name || "product")}">×</button></td>`;
  tr.querySelector("td:last-child button").onclick = () => {
    tr.remove();
    setDirty(true);
  };
  thumbs(tr);
  return tr;
}

// Rows grouped under "Category, Section" headings, in the order the sections first appear.
function render(catalog) {
  const rows = [];
  const where = (p) => [p.category, p.section].filter(Boolean).join(", ");
  const order = [...new Set(catalog.products.map(where))];
  let last = null;
  for (const p of [...catalog.products].sort((a, b) => order.indexOf(where(a)) - order.indexOf(where(b)))) {
    if (where(p) !== last) {
      last = where(p);
      const head = document.createElement("tr");
      head.className = "group";
      head.innerHTML = `<th colspan="8" scope="colgroup">${esc(last || "No category")} <span>${catalog.products.filter((x) => where(x) === last).length}</span></th>`;
      rows.push(head);
    }
    rows.push(productRow(p));
  }
  $("products").replaceChildren(...rows);
  $("belongings").value = catalog.belongings.join(", ");
  $("leave-after").value = catalog.leave_after_s ?? 3;
  filter();
  countPrompts();
}

function readCatalog() {
  const products = [...$("products").querySelectorAll("tr:not(.group)")]
    .map((tr, i) => {
      const [name, author, category, section, prompt, price] = [...tr.querySelectorAll("input")].map((el) => el.value.trim());
      const old = C.config.catalog.products.find((p) => p.id === tr.dataset.id) ?? {};
      // with no phrase given, the camera looks for the product by its name
      return { ...old, id: tr.dataset.id || `p${Date.now().toString(36)}${i}`, name: name || prompt, author, category, section, prompt: (prompt || name).toLowerCase(), price: Number(price) || 0, examples: tr.examples };
    })
    .filter((p) => p.prompt);
  const belongings = $("belongings").value.split(",").map((s) => s.trim()).filter(Boolean);
  return { ...C.config.catalog, products, belongings, leave_after_s: Number($("leave-after").value) || 3 };
}

function countPrompts() {
  const c = readCatalog();
  const n = 3 + new Set(c.products.map((p) => p.prompt)).size + c.belongings.length;
  const pictures = c.products.reduce((sum, p) => sum + (p.examples?.length ?? 0), 0);
  $("count").textContent = `${plural(c.products.length, "product")}, ${plural(pictures, "picture")}`;
  $("prompt-count").textContent =
    `${n} phrases a frame, with person, hand and "item held in hand". Products that share a phrase (every book is "book") are told apart by their printed text.` +
    (n > C.config.max_prompts ? ` Over ${C.config.max_prompts} needs a second request per frame, which halves the frame rate.` : "");
}

function filter() {
  const q = $("filter").value.trim().toLowerCase();
  let head = null;
  let shown = 0;
  for (const tr of $("products").children) {
    if (tr.classList.contains("group")) {
      if (head) head.hidden = !shown;
      head = tr;
      shown = 0;
      continue;
    }
    const text = [...tr.querySelectorAll("input")].map((el) => el.value).join(" ").toLowerCase();
    tr.hidden = q && !text.includes(q);
    if (!tr.hidden) shown++;
  }
  if (head) head.hidden = !shown;
}

function setDirty(on) {
  C.dirty = on;
  $("save").disabled = !on;
  $("note").textContent = on ? "Unsaved changes. They apply on the next Start after saving." : "";
  if (C.config) countPrompts();
}

async function save(catalog = readCatalog()) {
  const r = await fetch(`/api/catalog/${encodeURIComponent(C.config.shop)}`, { method: "PUT", headers: { "content-type": "application/json" }, body: JSON.stringify(catalog) });
  const body = await r.json();
  if (!r.ok) {
    $("note").textContent = body.error;
    return false;
  }
  C.config.catalog = body.catalog;
  render(body.catalog);
  setDirty(false);
  $("note").textContent = "Saved. It applies the next time the floor presses Start.";
  return true;
}

// A picture as the JPEG the server keeps: at most PICTURE_SIDE on its longer side.
async function toJpeg(source, box) {
  const [sx, sy, sw, sh] = box ?? [0, 0, source.width, source.height];
  const k = Math.min(1, PICTURE_SIDE / Math.max(sw, sh));
  const c = document.createElement("canvas");
  c.width = Math.round(sw * k);
  c.height = Math.round(sh * k);
  c.getContext("2d").drawImage(source, sx, sy, sw, sh, 0, 0, c.width, c.height);
  return new Promise((resolve) => c.toBlob(resolve, "image/jpeg", 0.9));
}

async function upload(blob) {
  const r = await fetch(`/api/examples/${encodeURIComponent(C.config.shop)}`, { method: "POST", headers: { "content-type": "image/jpeg" }, body: blob });
  const body = await r.json();
  if (!r.ok) throw new Error(body.error);
  return body.file;
}

$("picture-file").onchange = async () => {
  const file = $("picture-file").files[0];
  const tr = C.pictureFor;
  $("picture-file").value = "";
  if (!file || !tr) return;
  try {
    tr.examples.push(await upload(await toJpeg(await createImageBitmap(file))));
    thumbs(tr);
    setDirty(true);
  } catch (e) {
    $("note").textContent = `The picture could not be added: ${e.message}`;
  }
};
$("add-product").onclick = () => {
  const row = productRow();
  $("products").prepend(row);
  row.querySelectorAll("input")[0].focus();
  setDirty(true);
};
$("save").onclick = () => save();
$("products-view").addEventListener("input", (e) => {
  if (e.target.id === "filter") return filter();
  setDirty(true);
});
$("shop").onchange = async () => {
  if (C.dirty && !confirm("Leave this shop without saving its changes?")) {
    $("shop").value = C.config.shop;
    return;
  }
  await loadShop($("shop").value);
  C.clip = null;
  C.answers.clear();
  if (location.hash === "#footage") loadClips();
};
window.addEventListener("beforeunload", (e) => {
  if (C.dirty) e.preventDefault();
});

// ---------------------------------------------------------------- footage

async function loadClips(pick) {
  const { clips, samples } = await (await fetch("/api/clips")).json();
  C.clips = clips;
  const mine = samples.filter((s) => s.shop === C.config.shop);
  $("clip").innerHTML =
    `<option value="">Choose footage</option>` +
    (clips.length ? `<optgroup label="Imported">${clips.map((c) => `<option value="${c.clip}">${esc(c.title)} (${c.frames} frames)</option>`).join("")}</optgroup>` : "") +
    (mine.length ? `<optgroup label="Samples">${mine.map((s) => `<option value="sample:${esc(s.file)}">${esc(s.title)}</option>`).join("")}</optgroup>` : "") +
    `<option value="import">Import a video…</option>`;
  if (pick) {
    $("clip").value = pick;
    openClip(pick);
  }
}

async function openClip(id) {
  const meta = await (await fetch(`/api/clips/${id}`)).json();
  C.clip = meta;
  C.images.clear();
  C.answers.clear();
  C.like = null;
  select(null);
  $("scrubber").max = meta.frames - 1;
  $("strip").innerHTML = Array.from({ length: meta.frames }, (_, i) => `<span data-n="${i}"></span>`).join("");
  $("frame").width = meta.width;
  $("frame").height = meta.height;
  $("frame-box").style.aspectRatio = `${meta.width} / ${meta.height}`;
  showFrame(0);
}

async function newClip(body, headers, title) {
  $("frame-note").hidden = false;
  $("frame-note").textContent = `Cutting ${title} into frames…`;
  const r = await fetch("/api/clips", { method: "POST", headers, body });
  const meta = await r.json();
  if (!r.ok) {
    $("frame-note").textContent = meta.error;
    return;
  }
  await loadClips(meta.clip);
}

$("clip").onchange = () => {
  const v = $("clip").value;
  if (v === "import") return $("clip-file").click();
  if (v.startsWith("sample:")) {
    const file = v.slice(7);
    const title = $("clip").selectedOptions[0].textContent;
    const done = C.clips.find((c) => c.title === title);
    return done ? loadClips(done.clip) : newClip(JSON.stringify({ sample: file }), { "content-type": "application/json" }, title);
  }
  if (v) openClip(v);
};
$("clip-file").onchange = () => {
  const file = $("clip-file").files[0];
  $("clip-file").value = "";
  if (file) newClip(file, { "content-type": file.type || "video/mp4", "x-title": file.name.slice(0, 80) }, file.name);
};

function frameImage(n) {
  if (!C.images.has(n)) {
    const img = new Image();
    img.src = `/api/clips/${C.clip.clip}/frames/${n}`;
    img.onload = () => n === C.n && draw();
    C.images.set(n, img);
  }
  return C.images.get(n);
}

function showFrame(n) {
  if (!C.clip) return;
  C.n = Math.max(0, Math.min(C.clip.frames - 1, n));
  $("scrubber").value = C.n;
  $("frame-at").textContent = `${C.n + 1} of ${C.clip.frames}, ${C.clip.times[C.n].toFixed(1)} s`;
  $("frame-note").hidden = true;
  frameImage(C.n);
  frameImage(C.n + 1); // the next one is usually wanted next
  strip();
  draw();
  if (!C.answers.has(C.n)) look(C.n);
}

const phrasesAsked = () => [
  ...new Set([...GENERIC, ...C.config.catalog.products.map((p) => p.prompt), ...$("look-for").value.split(",").map((s) => s.trim().toLowerCase()).filter(Boolean)]),
].slice(0, 24);

async function look(n, extra = {}) {
  if (C.looking.has(n) && !extra.example) return;
  C.looking.add(n);
  strip();
  try {
    const r = await fetch(`/api/clips/${C.clip.clip}/frames/${n}/look`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ prompts: extra.example ? [] : phrasesAsked(), ...extra }) });
    const body = await r.json();
    if (!r.ok) throw new Error(body.error);
    if (extra.example) return body.detections;
    C.answers.set(n, body.detections);
  } catch (e) {
    if (extra.example) throw e;
    $("frame-note").hidden = false;
    $("frame-note").textContent = `The camera model could not look at this frame: ${e.message}`;
  } finally {
    C.looking.delete(n);
    strip();
    if (n === C.n) draw();
  }
}

function strip() {
  if (!C.clip) return;
  for (const cell of $("strip").children) {
    const n = Number(cell.dataset.n);
    const found = C.answers.get(n);
    cell.className = [n === C.n ? "at" : "", C.looking.has(n) ? "busy" : "", found ? (found.length ? "seen" : "none") : ""].join(" ");
  }
}

const COLOURS = ["#3b82f6", "#f59e0b", "#10b981", "#a855f7", "#06b6d4", "#ec4899", "#84cc16", "#f97316"];
const hue = (label) => COLOURS[[...label].reduce((h, ch) => (h * 31 + ch.charCodeAt(0)) >>> 0, 7) % COLOURS.length];

function outline(ctx, polys, W, H) {
  ctx.beginPath();
  for (const poly of polys) poly.forEach(([x, y], i) => (i ? ctx.lineTo(x * W, y * H) : ctx.moveTo(x * W, y * H)));
  ctx.closePath();
}

function shown() {
  const like = C.like?.frame === C.n ? C.like.detections : [];
  return [...(C.answers.get(C.n) ?? []), ...like];
}

function draw() {
  const canvas = $("frame");
  const ctx = canvas.getContext("2d");
  const W = canvas.width;
  const H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  if (C.live) return drawLive(ctx, W, H);
  const img = C.clip && frameImage(C.n);
  if (!img?.complete) return;
  ctx.drawImage(img, 0, 0, W, H);
  const px = W / 960;
  const sel = C.sel?.frame === C.n ? C.sel : null;
  // largest first, so a small outline inside a big one stays visible and clickable
  const dets = shown().sort((a, b) => area(b.box) - area(a.box));
  for (const d of dets) {
    const c = d.exemplar ? "#ffffff" : hue(d.label);
    outline(ctx, d.polygons, W, H);
    ctx.fillStyle = c + (sel && same(sel.box, d.box) ? "66" : "22");
    ctx.fill();
    ctx.lineWidth = (sel && same(sel.box, d.box) ? 3 : 1.5) * px;
    ctx.strokeStyle = c;
    ctx.stroke();
  }
  ctx.font = `500 ${12 * px}px "Instrument Sans", system-ui, sans-serif`;
  ctx.textBaseline = "middle";
  for (const d of dets) {
    const text = `${d.label} ${Math.round(d.score * 100)}`;
    const w = ctx.measureText(text).width + 10 * px;
    const x = Math.min(d.box[0] * W, W - w - 2);
    const y = Math.max(0, d.box[1] * H - 18 * px);
    ctx.fillStyle = "rgba(22,21,19,0.8)";
    ctx.fillRect(x, y, w, 18 * px);
    ctx.fillStyle = "#fff";
    ctx.fillText(text, x + 5 * px, y + 9 * px);
  }
  const box = C.drag?.box ?? (sel && !sel.polygons?.length ? sel.box : null);
  if (box) {
    ctx.setLineDash([6 * px, 4 * px]);
    ctx.lineWidth = 2 * px;
    ctx.strokeStyle = "#facc15";
    ctx.strokeRect(box[0] * W, box[1] * H, (box[2] - box[0]) * W, (box[3] - box[1]) * H);
    ctx.setLineDash([]);
  }
}

const area = (b) => (b[2] - b[0]) * (b[3] - b[1]);
const same = (a, b) => a.every((v, i) => Math.abs(v - b[i]) < 1e-4);

function inside([x, y], poly) {
  let hit = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i];
    const [xj, yj] = poly[j];
    if (yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) hit = !hit;
  }
  return hit;
}

function pointAt(e) {
  const r = $("frame").getBoundingClientRect();
  return [Math.min(1, Math.max(0, (e.clientX - r.left) / r.width)), Math.min(1, Math.max(0, (e.clientY - r.top) / r.height))];
}

$("frame").addEventListener("pointerdown", (e) => {
  if (!C.clip) return;
  C.drag = { from: pointAt(e), box: null };
  try {
    $("frame").setPointerCapture(e.pointerId); // keep the drag when the pointer leaves the frame
  } catch {
    // not every pointer can be captured; the drag still works inside the frame
  }
});
$("frame").addEventListener("pointermove", (e) => {
  if (!C.drag) return;
  const [x0, y0] = C.drag.from;
  const [x1, y1] = pointAt(e);
  C.drag.box = [Math.min(x0, x1), Math.min(y0, y1), Math.max(x0, x1), Math.max(y0, y1)];
  draw();
});
$("frame").addEventListener("pointerup", (e) => {
  const drag = C.drag;
  C.drag = null;
  if (!drag) return;
  if (drag.box && area(drag.box) > 0.0004) return pickDrawn(drag.box);
  // a click: the smallest outline under the pointer
  const at = pointAt(e);
  const hit = shown()
    .filter((d) => d.polygons.some((p) => inside(at, p)))
    .sort((a, b) => area(a.box) - area(b.box))[0];
  select(hit ? { frame: C.n, box: hit.box, polygons: hit.polygons, label: hit.label, score: hit.score } : null);
});

// A box drawn by hand. SAM is asked for everything that looks like it on this frame; the outline that fits the box
// best becomes the pick, so its phrase tests and its picture follow the object, not the rough box.
async function pickDrawn(box) {
  select({ frame: C.n, box, polygons: [], label: "", score: null });
  try {
    const found = await look(C.n, { example: box });
    const fit = found.map((d) => [d, iou(d.box, box)]).sort((a, b) => b[1] - a[1])[0];
    if (fit && fit[1] > 0.4) select({ frame: C.n, box: fit[0].box, polygons: fit[0].polygons, label: "", score: fit[0].score });
  } catch {
    // the endpoint is the older one, without examples: the drawn box stands as it is
  }
}

function iou(a, b) {
  const ix = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0]));
  const iy = Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
  const inter = ix * iy;
  return inter / (area(a) + area(b) - inter || 1);
}

$("scrubber").oninput = () => showFrame(Number($("scrubber").value));
$("prev").onclick = () => showFrame(C.n - 1);
$("next").onclick = () => showFrame(C.n + 1);
$("strip").onclick = (e) => e.target.dataset.n && showFrame(Number(e.target.dataset.n));
document.addEventListener("keydown", (e) => {
  if ($("footage-view").hidden || ["INPUT", "SELECT", "TEXTAREA"].includes(document.activeElement?.tagName)) return;
  if (e.key === "ArrowLeft") showFrame(C.n - 1);
  if (e.key === "ArrowRight") showFrame(C.n + 1);
});
$("look").onclick = () => {
  if (!C.clip) return;
  C.answers.clear();
  showFrame(C.n);
};
$("scan").onclick = async () => {
  if (!C.clip) return;
  const button = $("scan");
  button.disabled = true;
  const todo = Array.from({ length: C.clip.frames }, (_, i) => i).filter((i) => !C.answers.has(i));
  let done = 0;
  const worker = async () => {
    while (todo.length) {
      await look(todo.shift());
      button.textContent = `Looked at ${++done} of ${done + todo.length}`;
    }
  };
  await Promise.all([worker(), worker(), worker()]);
  button.textContent = "Look at every frame";
  button.disabled = false;
};

// ---------------------------------------------------------------- the pick

function select(sel) {
  C.sel = sel;
  C.described = null;
  C.ranked = null;
  C.chosen = "";
  C.views = [];
  C.phrases = sel?.label && !GENERIC.includes(sel.label) ? [sel.label] : [];
  draw();
  renderPick();
}

function cropCanvas(sel, side = 240) {
  const img = frameImage(sel.frame);
  const W = img.naturalWidth;
  const H = img.naturalHeight;
  const [x0, y0, x1, y1] = sel.box;
  const pad = 0.06;
  const sx = Math.max(0, (x0 - pad * (x1 - x0)) * W);
  const sy = Math.max(0, (y0 - pad * (y1 - y0)) * H);
  const sw = Math.min(W, (x1 + pad * (x1 - x0)) * W) - sx;
  const sh = Math.min(H, (y1 + pad * (y1 - y0)) * H) - sy;
  return { img, rect: [sx, sy, sw, sh], url: (() => {
    const c = document.createElement("canvas");
    const k = side / Math.max(sw, sh);
    c.width = Math.round(sw * k);
    c.height = Math.round(sh * k);
    c.getContext("2d").drawImage(img, sx, sy, sw, sh, 0, 0, c.width, c.height);
    return c.toDataURL("image/jpeg", 0.85);
  })() };
}

function renderPick() {
  const sel = C.sel;
  const pane = $("pick");
  if (!sel) {
    pane.innerHTML = `<p class="empty">Nothing picked yet. Click an outline on the frame, or drag a box round an object.</p>`;
    return;
  }
  const d = C.described ?? {};
  const products = C.config.catalog.products;
  const seenAs = sel.camera ? `Held up to the camera, ${plural(sel.camera, "view")} kept` : sel.label ? `Outlined as "${esc(sel.label)}", ${Math.round(sel.score * 100)} %` : sel.polygons.length ? `Outlined from your box, ${Math.round((sel.score ?? 0) * 100)} %` : "Your box";
  const ranked = C.ranked
    ? `<table class="ranked-phrases"><thead><tr><th scope="col">Phrase</th><th scope="col">Outlines it</th><th scope="col" class="num">Score</th><th scope="col" class="num">Frames</th><th scope="col" class="num">Other outlines</th></tr></thead><tbody>${C.ranked
        .map(
          (r) => `<tr class="${r.phrase === C.chosen ? "chosen" : ""}"><td><label class="check tight"><input type="radio" name="phrase" value="${esc(r.phrase)}" ${r.phrase === C.chosen ? "checked" : ""} /> ${esc(r.phrase)}</label></td>
          <td>${r.fits ? `<span class="state ok">Yes</span>` : `<span class="state">${r.fit ? "Partly" : "No"}</span>`}</td><td class="num">${Math.round(r.score * 100)}</td><td class="num">${r.frames} of ${r.of}</td><td class="num">${r.others}</td></tr>`,
        )
        .join("")}</tbody></table>
       <p class="hint">${C.ranked[0]?.fits ? `"${esc(C.ranked[0].phrase)}" outlines this object best.` : "None of these outlines the whole object. Try a phrase of your own, or describe the packaging."}</p>`
    : "";
  pane.innerHTML = `
    <div class="pick-head">
      <img src="${cropCanvas(sel).url}" alt="The picked object" />
      <div>
        <p><strong>${esc(d.name || "Picked object")}</strong></p>
        <p class="hint">${seenAs}. Frame ${sel.frame + 1}.</p>
        <div class="row">
          <button type="button" id="describe">${C.described ? "Describe again" : "Describe it"}</button>
          <button type="button" id="more-like">Find more like it</button>
        </div>
      </div>
    </div>
    ${C.views.length ? `<div class="views"><h3>Pictures to keep</h3><div class="pics">${C.views.map((v, i) => `<button type="button" class="thumb big" data-view="${i}" title="Leave this picture out"><img src="${cropCanvas(v, 120).url}" alt="" /></button>`).join("")}</div></div>` : ""}
    <p class="hint" id="pick-note" role="status"></p>
    <form id="product-form" class="form-grid">
      <div class="field span"><label for="p-name">Name</label><input id="p-name" type="text" value="${esc(d.name ?? "")}" autocomplete="off" /></div>
      <div class="field"><label for="p-maker">Maker</label><input id="p-maker" type="text" value="${esc(d.maker ?? "")}" autocomplete="off" /></div>
      <div class="field"><label for="p-price">Price</label><input id="p-price" type="number" min="0" step="0.5" inputmode="decimal" /></div>
      <div class="field"><label for="p-category">Category</label><input id="p-category" type="text" value="${esc(d.category ?? "")}" autocomplete="off" /></div>
      <div class="field"><label for="p-section">Section</label><input id="p-section" type="text" value="${esc(d.section ?? "")}" autocomplete="off" /></div>
      <div class="field span"><label for="p-cues">Looks like</label><input id="p-cues" type="text" value="${esc(d.cues ?? "")}" autocomplete="off" /></div>
    </form>
    <div class="phrases">
      <h3>Camera phrase</h3>
      <div class="chips">${C.phrases.map((p) => `<span class="chip">${esc(p)}<button type="button" data-drop="${esc(p)}" aria-label="Drop ${esc(p)}">×</button></span>`).join("")}
        <input type="text" id="new-phrase" placeholder="Add a phrase" aria-label="Add a phrase" autocomplete="off" /></div>
      <button type="button" id="test-phrases" ${C.phrases.length ? "" : "disabled"}>Test ${C.phrases.length > 1 ? "these phrases" : "the phrase"} on this frame and its neighbours</button>
      ${ranked}
    </div>
    <div class="add">
      <button type="button" class="primary" id="add-new">Add as a new product</button>
      <span class="or">or add the picture to</span>
      <select id="add-to" aria-label="Product to add the picture to">${products.map((p) => `<option value="${esc(p.id)}">${esc(p.name)}</option>`).join("")}</select>
      <button type="button" id="add-picture">Add picture</button>
    </div>`;
  const say = (text) => ($("pick-note").textContent = text);
  $("describe").onclick = async () => {
    say("Asking the vision model…");
    const r = await fetch(`/api/clips/${C.clip.clip}/describe`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ frame: sel.frame, box: sel.box, shop: C.config.shop }) });
    const body = await r.json();
    if (!r.ok) return say(body.error);
    C.described = body;
    C.phrases = [...new Set([...C.phrases, ...body.phrases])];
    renderPick();
    say(body.text ? `Reads "${body.text}". Test the phrases to see which one the camera finds it by.` : "Test the phrases to see which one the camera finds it by.");
  };
  $("more-like").onclick = async () => {
    say("Looking for more like it on this frame…");
    try {
      const found = await look(sel.frame, { example: sel.box, hint: C.chosen || "" });
      C.like = { frame: sel.frame, detections: found };
      draw();
      say(found.length > 1 ? `${found.length} objects on this frame look like it, outlined in white.` : "Nothing else on this frame looks like it.");
    } catch (e) {
      say(e.message);
    }
  };
  pane.querySelectorAll("[data-drop]").forEach((b) => {
    b.onclick = () => {
      keepForm();
      C.phrases = C.phrases.filter((p) => p !== b.dataset.drop);
      C.ranked = C.ranked?.filter((r) => r.phrase !== b.dataset.drop) ?? null;
      renderPick();
    };
  });
  $("new-phrase").onkeydown = (e) => {
    if (e.key !== "Enter" || !e.target.value.trim()) return;
    e.preventDefault();
    keepForm();
    C.phrases = [...new Set([...C.phrases, e.target.value.trim().toLowerCase()])];
    renderPick();
    $("new-phrase").focus();
  };
  $("test-phrases").onclick = async () => {
    keepForm();
    say("Trying each phrase on five frames…");
    $("test-phrases").disabled = true;
    const r = await fetch(`/api/clips/${C.clip.clip}/phrases`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ frame: sel.frame, box: sel.box, phrases: C.phrases }) });
    const body = await r.json();
    if (!r.ok) return say(body.error);
    C.ranked = body.ranked;
    C.chosen = body.ranked[0]?.phrase ?? "";
    renderPick();
  };
  pane.querySelectorAll('input[name="phrase"]').forEach((el) => (el.onchange = () => {
    keepForm();
    C.chosen = el.value;
    renderPick();
  }));
  pane.querySelectorAll("[data-view]").forEach((b) => {
    b.onclick = () => {
      keepForm();
      C.views.splice(Number(b.dataset.view), 1);
      renderPick();
    };
  });
  if (C.form) restoreForm();
  $("add-new").onclick = () => addProduct(say);
  $("add-picture").onclick = () => addPicture($("add-to").value, say);
}

// The form is drawn again when phrases change; what was typed into it is kept.
function keepForm() {
  C.form = Object.fromEntries(["p-name", "p-maker", "p-price", "p-category", "p-section", "p-cues"].map((id) => [id, $(id)?.value ?? ""]));
}
function restoreForm() {
  for (const [id, value] of Object.entries(C.form)) if ($(id) && value) $(id).value = value;
  C.form = null;
}

async function pictureOf(sel) {
  const { img, rect } = cropCanvas(sel);
  return upload(await toJpeg(img, rect));
}

// The pictures that go on the product: every view a camera capture kept, or the one pick.
async function pictures(room = 4) {
  const views = C.views.length ? C.views : [C.sel];
  const files = [];
  for (const v of views.slice(0, room)) files.push(await pictureOf(v));
  return files;
}

async function addProduct(say) {
  const name = $("p-name").value.trim();
  const phrase = C.chosen || C.phrases[0] || "";
  if (!name || !phrase) return say(!name ? "Give the product a name first." : "Add a camera phrase and test it first.");
  if (C.dirty && !confirm("The product list has unsaved changes. Save them together with this product?")) return;
  say("Adding…");
  try {
    const files = await pictures();
    const catalog = readCatalog();
    catalog.products.unshift({
      id: `p${Date.now().toString(36)}`,
      name,
      author: $("p-maker").value.trim(),
      category: $("p-category").value.trim(),
      section: $("p-section").value.trim(),
      cues: $("p-cues").value.trim(),
      prompt: phrase,
      price: Number($("p-price").value) || 0,
      examples: files,
    });
    if (await save(catalog)) say(`Added ${name}: the camera looks for "${phrase}", with ${plural(files.length, "picture")}.`);
  } catch (e) {
    say(`Could not add it: ${e.message}`);
  }
}

async function addPicture(id, say) {
  const catalog = readCatalog();
  const product = catalog.products.find((p) => p.id === id);
  if (!product) return;
  if ((product.examples ?? []).length >= 4) return say(`${product.name} has four pictures already. Remove one on the product list first.`);
  say("Adding the picture…");
  try {
    const files = await pictures(4 - (product.examples ?? []).length);
    product.examples = [...(product.examples ?? []), ...files];
    if (await save(catalog)) say(`Added ${plural(files.length, "picture")} to ${product.name}.`);
  } catch (e) {
    say(`Could not add it: ${e.message}`);
  }
}

// ---------------------------------------------------------------- hold to camera

async function listCams() {
  const cams = (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === "videoinput" && d.label);
  const before = localStorage.getItem("catalogue-camera") || "";
  $("cam-pick").innerHTML = cams.map((d) => `<option value="${esc(d.deviceId)}">${esc(d.label)}</option>`).join("");
  if (cams.some((d) => d.deviceId === before)) $("cam-pick").value = before;
  $("cam-pick").hidden = cams.length < 2;
}

async function startCapture() {
  const id = $("cam-pick").value || localStorage.getItem("catalogue-camera") || "";
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: { ...(id ? { deviceId: { exact: id } } : {}), width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false });
  } catch (e) {
    return capture(`The camera could not be opened: ${e.message}`);
  }
  listCams();
  const meta = await (await fetch("/api/clips/live", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ title: `Camera, ${new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}` }) })).json();
  const cam = $("cam");
  cam.srcObject = stream;
  cam.hidden = false;
  await cam.play();
  C.clip = null;
  select(null);
  C.live = { clip: meta.clip, stream, kept: [], latest: null, t0: performance.now(), busy: false };
  $("frame").width = cam.videoWidth;
  $("frame").height = cam.videoHeight;
  $("frame-box").style.aspectRatio = `${cam.videoWidth} / ${cam.videoHeight}`;
  $("frame-note").hidden = true;
  $("scrub").hidden = $("strip").hidden = true;
  $("hold").textContent = "Done";
  capture("Hold the product up to the camera and turn it slowly, so every side is seen.");
  C.live.timer = setInterval(snap, CAPTURE_EVERY_MS);
}

function capture(text) {
  $("capture").hidden = !text;
  $("capture").textContent = text;
}

async function snap() {
  const live = C.live;
  if (!live || live.busy) return;
  const elapsed = performance.now() - live.t0;
  if (live.kept.length >= CAPTURE_VIEWS || elapsed > CAPTURE_FOR_MS) return stopCapture();
  live.busy = true;
  const cam = $("cam");
  const grab = document.createElement("canvas");
  grab.width = cam.videoWidth;
  grab.height = cam.videoHeight;
  const g = grab.getContext("2d", { willReadFrequently: true });
  g.drawImage(cam, 0, 0);
  if (dark(g, grab.width, grab.height)) {
    live.busy = false;
    live.dark = (live.dark ?? 0) + 1;
    // a camera that another app holds, or one with its shutter closed, sends black pictures
    if (live.dark >= 3) capture("The camera is sending a black picture. Close other apps using it, or choose another camera.");
    return;
  }
  live.dark = 0;
  const jpeg = await new Promise((resolve) => grab.toBlob(resolve, "image/jpeg", 0.88));
  try {
    const look = encodeURIComponent($("look-for").value.trim());
    const r = await fetch(`/api/clips/${live.clip}/frames?t=${(elapsed / 1000).toFixed(2)}&look=${look}`, { method: "POST", headers: { "content-type": "image/jpeg" }, body: jpeg });
    const body = await r.json();
    if (C.live !== live) return;
    const held = body.held;
    live.latest = held ? { ...held, frame: body.frame } : null;
    if (held && held.score >= SURE_HELD) live.kept.push({ frame: body.frame, box: held.box, score: held.score });
    capture(
      body.error
        ? `The camera model did not answer: ${body.error}`
        : `${held ? (held.score >= SURE_HELD ? "Got it." : "Hold it a little closer.") : "No item in hand yet."} ${live.kept.length} of ${CAPTURE_VIEWS} views. Keep turning it, or press Done.`,
    );
    draw();
  } finally {
    live.busy = false;
  }
}

function drawLive(ctx, W, H) {
  const d = C.live.latest;
  if (!d) return;
  const px = W / 960;
  const c = d.score >= SURE_HELD ? "#10b981" : "#f59e0b";
  outline(ctx, d.polygons, W, H);
  ctx.fillStyle = c + "44";
  ctx.fill();
  ctx.lineWidth = 3 * px;
  ctx.strokeStyle = c;
  ctx.stroke();
  const text = `${d.label} ${Math.round(d.score * 100)}`;
  ctx.font = `500 ${14 * px}px "Instrument Sans", system-ui, sans-serif`;
  ctx.textBaseline = "middle";
  const w = ctx.measureText(text).width + 12 * px;
  const x = Math.min(d.box[0] * W, W - w - 2);
  const y = Math.max(0, d.box[1] * H - 22 * px);
  ctx.fillStyle = "rgba(22,21,19,0.8)";
  ctx.fillRect(x, y, w, 22 * px);
  ctx.fillStyle = "#fff";
  ctx.fillText(text, x + 6 * px, y + 11 * px);
}

// True when a grabbed camera picture is all but black. Read from a small copy, so it costs next to nothing.
function dark(ctx, W, H) {
  const small = document.createElement("canvas");
  small.width = 32;
  small.height = 18;
  const s = small.getContext("2d", { willReadFrequently: true });
  s.drawImage(ctx.canvas, 0, 0, W, H, 0, 0, 32, 18);
  const px = s.getImageData(0, 0, 32, 18).data;
  let sum = 0;
  for (let i = 0; i < px.length; i += 4) sum += px[i] + px[i + 1] + px[i + 2];
  return sum / (px.length / 4) / 3 < 6;
}

async function stopCapture() {
  const live = C.live;
  if (!live) return;
  clearInterval(live.timer);
  C.live = null;
  live.stream.getTracks().forEach((t) => t.stop());
  $("cam").hidden = true;
  $("cam").srcObject = null;
  $("hold").textContent = "Hold to camera";
  $("scrub").hidden = $("strip").hidden = false;
  if (!live.kept.length) {
    // still show what was filmed: a box dragged round the item on any frame works as well
    await loadClips();
    const filmed = C.clips.find((c) => c.clip === live.clip)?.frames;
    if (filmed) {
      $("clip").value = live.clip;
      await openClip(live.clip);
    } else draw();
    capture(
      filmed
        ? `No item in hand was outlined in ${plural(filmed, "frame")}. Drag a box round it on a frame, or type what it is in "Also look for" and try again.`
        : live.dark
          ? "The camera sent only black pictures. Close other apps using it, or choose another camera."
          : "Nothing came through from the camera. Try again, or choose another camera.",
    );
    return;
  }
  await finishCapture(live.clip, live.kept);
}

// From the views a capture kept to a filled-in form: the server picks the pictures, names it and tests phrases.
async function finishCapture(clip, kept) {
  capture(`Choosing the best of ${plural(kept.length, "view")} and asking what it is…`);
  await loadClips();
  $("clip").value = clip;
  await openClip(clip);
  const r = await fetch(`/api/clips/${clip}/auto`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ views: kept, shop: C.config.shop }) });
  const body = await r.json();
  if (!r.ok) return capture(body.error);
  const best = kept.find((v) => v.frame === body.best.frame);
  // the pictures are cut out of these frames, so they have to be loaded first
  await Promise.all([body.best, ...body.pictures].map((v) => frameImage(v.frame).decode().catch(() => {})));
  showFrame(body.best.frame);
  select({ frame: body.best.frame, box: body.best.box, polygons: [], label: "", score: best?.score ?? null, camera: kept.length });
  C.views = body.pictures;
  if (body.described) {
    C.described = body.described;
    C.phrases = body.described.phrases;
  }
  if (body.ranked?.length) {
    C.ranked = body.ranked;
    C.chosen = body.ranked[0].phrase;
  }
  renderPick();
  capture(body.note || `Filled in from ${plural(kept.length, "view")}. Check it, add the price, then add the product.`);
  $("p-price")?.focus();
}

$("hold").onclick = () => (C.live ? stopCapture() : startCapture());
$("cam-pick").onchange = () => localStorage.setItem("catalogue-camera", $("cam-pick").value);
navigator.mediaDevices?.enumerateDevices && listCams().catch(() => {});

// ---------------------------------------------------------------- start

Object.assign(window, { catalogue: C, finishCapture }); // for poking at the state from the console

loadShop(currentShop()).then(route).catch((e) => ($("note").textContent = `Could not load: ${e.message}`));

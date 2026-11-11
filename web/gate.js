// The gate, browser side.
//
// The camera (or a photo, where there is no camera) is shown to the server about once a second. The server answers
// who it sees: nobody, someone it knows, or a new face. Someone known with a payment method on file is let in;
// anyone else registers here first. The same answer has to come a few times running before the screen acts on it,
// so one poor frame neither lets anyone in nor offers a regular the sign-up form.

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const ordinal = (n) => `${n}${["th", "st", "nd", "rd"][n % 100 > 10 && n % 100 < 14 ? 0 : Math.min(n % 10, 4) % 4]}`;
const LOOK_MS = 700;
const TICK = `<svg viewBox="0 0 32 32" aria-hidden="true"><circle cx="16" cy="16" r="15" fill="none" stroke="currentColor" stroke-width="2"/><path d="M9.5 16.5l4.5 4.5 8.5-9.5" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"/></svg>`;

const G = {
  config: null,
  source: "none", // none | camera | photo
  stream: null,
  photo: null, // the chosen picture, as a Blob
  mode: "watch", // watch | form | approving | in
  shown: "", // which panel is on screen
  streak: { key: "", n: 0 },
  looking: false,
  ticket: null,
  pass: null,
  profile: null,
  faceUrl: "",
  draft: {}, // what was typed, kept across a trip to PayPal and back
  reg: null,
  popup: null,
  approveUrl: "",
  away: 0, // looks in a row without the person who was let in
  others: 0, // looks in a row with somebody else in front of the gate
  ok: false, // can the gate work at all: face recognition and payments are on
};

const camera = $("camera");
const grab = document.createElement("canvas");

// ---------------------------------------------------------------- the sign at the top: is the gate open?

// The gate is closed until it has let in the person standing in front of it, and closes again the moment they
// walk on or somebody else steps up. It says "open" for nobody else.
function setGate(open) {
  if (!G.ok) return; // the sign is saying what is switched off
  $("gate-dot").className = `dot ${open ? "ok" : "off"}`;
  $("gate-status").textContent = open ? `Gate open for ${G.profile?.name ?? "you"}` : "Gate closed";
}

// ---------------------------------------------------------------- the panel beside the camera

function show(name, html) {
  if (G.shown === name) return;
  G.shown = name;
  $("panel").innerHTML = `<section>${html}</section>`;
}

function showIdle(kind = "idle") {
  if (G.source === "none") return show("off", `<h1>Entry gate</h1><p class="lede">Turn on the camera to walk in</p>`);
  if (kind === "closer") return show("closer", `<h1>Come a little closer</h1><p class="lede">Your face is not clear enough yet</p>`);
  show("idle", `<h1>Look at the camera</h1><p class="lede">Returning shoppers walk straight in. New here? Registering takes a minute, once.</p>`);
}

function showIn(profile, isNew) {
  const method = esc(profile.method);
  const head = isNew
    ? `<h1>You're in, ${esc(profile.name.split(" ")[0])}</h1><p class="lede"><strong>${method}</strong> is charged as you leave. Invoice to <strong>${esc(profile.email)}</strong></p>`
    : `<h1>Welcome back, ${esc(profile.name.split(" ")[0])}</h1><p class="lede"><strong>${method}</strong> on file</p>`;
  const nth = profile.visits + 1;
  const history = !isNew && profile.visits ? `<p class="hint history">Your ${ordinal(nth)} visit.${profile.recent?.length ? ` Last time: ${esc(profile.recent.join(", "))}.` : ""}</p>` : "";
  G.shown = "";
  show("in", `${head}<p class="in">${TICK}<span>${isNew ? "Take what you need and walk out" : "Take what you need and walk out"}</span></p>${history}<p class="row"><button type="button" class="quiet" id="next">Next shopper</button></p>`);
  $("next").onclick = reset;
}

function showForm(problem = "") {
  const known = G.profile;
  const cards = Object.entries(G.config.test_cards).map(([id, label]) => `<option value="${esc(id)}">${esc(label)}</option>`).join("");
  const d = G.draft;
  const head = known
    ? `<h1>Welcome back, ${esc(known.name.split(" ")[0])}</h1><p class="lede">Add a way to pay to walk in</p>`
    : `<h1>First time here?</h1><p class="lede">Register once, then just walk in</p>`;
  const you = known
    ? ""
    : `<div class="you"><img class="face" src="${esc(G.faceUrl)}" alt="Your face as the gate sees it" width="64" height="64" /><span class="hint">How the gate will recognise you</span></div>
       <div class="field"><label for="name">Name</label><input id="name" type="text" autocomplete="name" maxlength="40" value="${esc(d.name ?? "")}" /></div>`;
  const email = known?.email ? "" : `<div class="field"><label for="email">Email for your invoices</label><input id="email" type="email" autocomplete="email" maxlength="120" value="${esc(d.email ?? G.config.services.demo_email ?? "")}" /></div>`;
  G.shown = "";
  show(
    "form",
    `${head}
    <form class="register" id="register" novalidate>
      ${you}${email}
      <fieldset class="methods">
        <legend>How you pay when you walk out</legend>
        <label class="method"><input type="radio" name="method" value="paypal" ${d.method === "card" ? "" : "checked"} /><span>PayPal account</span><small>Approve once in PayPal</small></label>
        <label class="method"><input type="radio" name="method" value="card" ${d.method === "card" ? "checked" : ""} /><span>Card</span><small>Sandbox test card, saved with PayPal</small></label>
      </fieldset>
      <div id="card-box" hidden>
        <div class="field"><label for="test-card">Card</label><select id="test-card">${cards}<option value="">Type the card details</option></select></div>
        <div class="card-fields" id="typed" hidden>
          <div class="field wide"><label for="card-number">Card number</label><input id="card-number" type="text" inputmode="numeric" autocomplete="off" maxlength="23" placeholder="A PayPal sandbox test number" /></div>
          <div class="field"><label for="card-expiry">Expires</label><input id="card-expiry" type="text" inputmode="numeric" autocomplete="off" maxlength="5" placeholder="MM/YY" /></div>
          <div class="field"><label for="card-code">Security code</label><input id="card-code" type="text" inputmode="numeric" autocomplete="off" maxlength="4" /></div>
        </div>
      </div>
      <p class="problem" id="problem" role="alert">${esc(problem)}</p>
      <div class="row"><button class="primary" type="submit" id="submit">Continue to PayPal</button><button type="button" class="quiet" id="restart">${known ? "Not you?" : "Not you? Start again"}</button></div>
    </form>`,
  );
  const form = $("register");
  const sync = () => {
    const card = form.method.value === "card";
    $("card-box").hidden = !card;
    $("typed").hidden = $("test-card").value !== "";
    $("submit").textContent = card ? "Register and come in" : "Continue to PayPal";
    if (known) $("submit").textContent = card ? "Save the card and come in" : "Continue to PayPal";
  };
  form.addEventListener("change", sync);
  form.addEventListener("submit", register);
  $("restart").onclick = reset;
  sync();
  (known ? form.querySelector("input[name=method]") : $("name")).focus({ preventScroll: true });
}

function showApproving() {
  G.shown = "";
  show(
    "approving",
    `<h1>Approve in the PayPal window</h1><p class="lede">Use a sandbox buyer account. This screen continues on its own</p>
     <p class="row"><a class="button" href="${esc(G.approveUrl)}" target="paypal" rel="opener">Open PayPal again</a><button type="button" class="quiet" id="cancel">Cancel</button></p>`,
  );
  $("cancel").onclick = () => {
    G.reg = null;
    G.popup?.close();
    G.mode = "form";
    showForm();
  };
}

// ---------------------------------------------------------------- registering

async function post(path, body) {
  const r = await fetch(path, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) });
  return r.json();
}

async function register(event) {
  event.preventDefault();
  const form = $("register");
  const method = form.method.value;
  G.draft = { name: $("name")?.value ?? "", email: $("email")?.value ?? "", method };
  const body = { ticket: G.profile ? null : G.ticket, pass: G.profile ? G.pass : null, name: G.draft.name, email: G.draft.email, method };
  if (method === "card") {
    body.test_card = $("test-card").value;
    if (!body.test_card) {
      const [mm, yy] = $("card-expiry").value.split("/").map((x) => x.trim());
      body.card = { number: $("card-number").value, expiry: `20${yy ?? ""}-${(mm ?? "").padStart(2, "0")}`, security_code: $("card-code").value };
    }
  }
  // the window has to open on the click itself, or the browser blocks it; it is pointed at PayPal once the address is known
  if (method === "paypal") G.popup = window.open("", "paypal", "width=480,height=760");
  $("submit").disabled = true;
  $("problem").textContent = "";
  let answer;
  try {
    answer = await post("/api/gate/register", body);
  } catch {
    answer = { status: "error", message: "The gate could not reach the shop. Please try again." };
  }
  if (answer.status === "admitted") return letIn(answer.profile, answer.new);
  if (answer.status === "approve") {
    G.reg = answer.reg;
    G.approveUrl = answer.approve_url;
    if (G.popup && !G.popup.closed) G.popup.location = answer.approve_url;
    G.mode = "approving";
    showApproving();
    return waitForPayPal(answer.reg);
  }
  G.popup?.close();
  $("submit").disabled = false;
  $("problem").textContent = answer.message ?? "That did not work. Please try again.";
  if (answer.field && $(answer.field)) $(answer.field).focus();
}

async function waitForPayPal(reg) {
  while (G.reg === reg) {
    await new Promise((r) => setTimeout(r, 1500));
    if (G.reg !== reg) return;
    let answer;
    try {
      answer = await (await fetch(`/api/gate/progress/${reg}`)).json();
    } catch {
      continue;
    }
    if (answer.status === "waiting") continue;
    G.reg = null;
    if (answer.status === "admitted") return letIn(answer.profile, answer.new);
    G.mode = "form";
    return showForm(answer.message ?? "PayPal was not set up. Please try again.");
  }
}

function letIn(profile, isNew) {
  G.mode = "in";
  G.profile = profile;
  G.away = 0;
  G.others = 0;
  G.draft = {};
  setGate(true);
  showIn(profile, isNew);
}

function reset() {
  Object.assign(G, { mode: "watch", streak: { key: "", n: 0 }, ticket: null, pass: null, profile: null, faceUrl: "", draft: {}, reg: null, away: 0, others: 0 });
  setGate(false);
  showIdle();
}

// ---------------------------------------------------------------- looking

function frameBlob() {
  if (G.source === "photo") return Promise.resolve(G.photo);
  if (G.source !== "camera" || camera.readyState < 2 || !camera.videoWidth) return Promise.resolve(null);
  const k = Math.min(1, 720 / camera.videoWidth);
  grab.width = Math.round(camera.videoWidth * k);
  grab.height = Math.round(camera.videoHeight * k);
  grab.getContext("2d").drawImage(camera, 0, 0, grab.width, grab.height);
  return new Promise((resolve) => grab.toBlob(resolve, "image/jpeg", 0.85));
}

// Put the frame round the face. The picture fills its box and is cropped to it, and the camera is shown mirrored.
function frameFace(answer) {
  const seen = $("seen");
  const box = answer.box;
  if (!box) return seen.classList.remove("on");
  const view = G.source === "photo" ? $("photo") : camera;
  const w = G.source === "photo" ? view.naturalWidth : view.videoWidth;
  const h = G.source === "photo" ? view.naturalHeight : view.videoHeight;
  const W = $("lens").clientWidth;
  const H = $("lens").clientHeight;
  if (!w || !h) return seen.classList.remove("on");
  const scale = Math.max(W / w, H / h);
  const dx = (W - w * scale) / 2;
  const dy = (H - h * scale) / 2;
  let [x0, y0, x1, y1] = box;
  if (G.source === "camera") [x0, x1] = [1 - x1, 1 - x0];
  Object.assign(seen.style, { left: `${dx + x0 * w * scale}px`, top: `${dy + y0 * h * scale}px`, width: `${(x1 - x0) * w * scale}px`, height: `${(y1 - y0) * h * scale}px` });
  seen.classList.add("on");
  seen.classList.toggle("known", answer.status === "known");
  // who the gate takes this face for, and how well it matches the face they registered with
  const sure = answer.score ? `, ${Math.round(answer.score * 100)}% match` : "";
  $("seen-name").textContent = answer.status === "known" ? `${answer.profile.name}${sure}` : answer.status === "new" ? "Not registered" : "";
}

async function look() {
  if (G.looking || G.source === "none" || G.mode === "approving") return;
  if (G.source === "photo" && G.mode !== "watch") return; // a photo does not change: nothing new to see
  G.looking = true;
  try {
    const blob = await frameBlob();
    if (!blob) return;
    const answer = await (await fetch("/api/gate/look", { method: "POST", body: blob })).json();
    frameFace(answer);
    await act(answer);
  } catch {
    // the next look will try again
  } finally {
    G.looking = false;
  }
}

async function act(answer) {
  const key = `${answer.status}:${answer.profile?.id ?? answer.ticket ?? ""}`;
  G.streak = { key, n: key === G.streak.key ? G.streak.n + 1 : 1 };
  if (G.mode === "in") {
    // The welcome is for one person. It stays while they stand there, and goes when they walk on or when anyone
    // else steps up: a different face is never greeted with the last shopper's name.
    const same = answer.status === "known" && answer.profile.id === G.profile?.id;
    const someoneElse = answer.status === "new" || answer.status === "closer" || (answer.status === "known" && !same);
    G.away = same ? 0 : G.away + 1;
    G.others = someoneElse ? G.others + 1 : 0;
    if (same) return setGate(true);
    if (someoneElse) setGate(false); // closed at once; the welcome follows on the next look
    if (G.others >= 2) {
      const streak = G.streak;
      reset();
      G.streak = streak; // the newcomer has already been seen twice: carry on from there
    } else {
      if (G.away >= 2) setGate(false);
      if (G.away >= 4) reset();
      return;
    }
  }
  if (G.mode !== "watch") return;
  if (answer.status === "no_face") return showIdle();
  if (answer.status === "closer") return showIdle("closer");
  if (answer.status === "known" && G.streak.n >= 2) {
    G.pass = answer.pass;
    if (answer.ready) {
      const let_in = await post("/api/gate/admit", { pass: answer.pass });
      if (let_in.status === "admitted") return letIn(let_in.profile, false);
    }
    G.mode = "form";
    G.profile = answer.profile;
    return showForm();
  }
  if (answer.status === "new" && G.streak.n >= 3) {
    G.mode = "form";
    G.ticket = answer.ticket;
    G.faceUrl = answer.face;
    return showForm();
  }
}

// ---------------------------------------------------------------- camera and photo

// The gate has its own camera, apart from the shop floor's. With more than one plugged in, a list under the picture
// chooses which, and the choice is remembered on this screen.
async function listCameras() {
  const pick = $("camera-pick");
  let cameras = [];
  try {
    cameras = (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === "videoinput" && d.label);
  } catch {
    // no device list here: the gate uses whatever camera the browser gives it
  }
  pick.hidden = cameras.length < 2;
  pick.innerHTML = cameras.map((d) => `<option value="${esc(d.deviceId)}">${esc(d.label)}</option>`).join("");
  const used = G.stream?.getVideoTracks()[0]?.getSettings().deviceId;
  if (used) pick.value = used;
}

async function useCamera(deviceId = localStorage.getItem("gate-camera") || "") {
  G.stream?.getTracks().forEach((t) => t.stop());
  const size = { width: { ideal: 1280 }, height: { ideal: 720 } };
  try {
    G.stream = await navigator.mediaDevices.getUserMedia({ video: deviceId ? { deviceId: { exact: deviceId }, ...size } : { facingMode: "user", ...size }, audio: false });
  } catch {
    if (deviceId) {
      localStorage.removeItem("gate-camera"); // that camera is gone: fall back to any
      return useCamera("");
    }
    $("lens-note").querySelector("p").textContent = "Camera blocked. Allow it in the browser, or use a photo";
    return;
  }
  listCameras();
  camera.srcObject = G.stream;
  camera.hidden = false;
  $("photo").hidden = true;
  $("lens-note").hidden = true;
  $("back-to-camera").hidden = true;
  $("use-photo").hidden = false;
  G.source = "camera";
  reset();
}

function usePhoto(file) {
  if (!file) return;
  G.stream?.getTracks().forEach((t) => t.stop());
  G.stream = null;
  G.photo = file;
  const img = $("photo");
  img.src = URL.createObjectURL(file);
  img.hidden = false;
  camera.hidden = true;
  $("camera-pick").hidden = true;
  $("lens-note").hidden = true;
  $("back-to-camera").hidden = false;
  G.source = "photo";
  $("seen").classList.remove("on");
  reset();
}

$("use-camera").onclick = () => useCamera();
$("back-to-camera").onclick = () => useCamera();
$("camera-pick").onchange = () => {
  localStorage.setItem("gate-camera", $("camera-pick").value);
  useCamera($("camera-pick").value);
};
navigator.mediaDevices?.addEventListener?.("devicechange", listCameras);
$("use-photo").onclick = () => $("photo-file").click();
$("photo-file").onchange = () => usePhoto($("photo-file").files[0]);

async function boot() {
  G.config = await (await fetch("/api/gate/config")).json();
  const shop = G.config.shops.find((x) => x.id === G.config.shop)?.name ?? "Shop";
  $("brand").textContent = shop;
  document.title = `${shop}: the gate`;
  const sv = G.config.services;
  G.ok = Boolean(sv.faces && sv.payments);
  if (!G.ok) {
    $("gate-dot").className = "dot bad";
    $("gate-status").textContent = !sv.faces ? "Face recognition is off" : "Payments are off, so nobody can register";
  }
  setGate(false);
  showIdle();
  // a camera that was allowed before starts by itself; otherwise the shopper turns it on
  try {
    if ((await navigator.permissions.query({ name: "camera" })).state === "granted") await useCamera();
  } catch {
    // no Permissions API for the camera here: the button is there
  }
  setInterval(look, LOOK_MS);
}

Object.assign(window, { gate: G, gateUsePhoto: usePhoto, gateAct: act }); // for poking at the state from the console
boot();

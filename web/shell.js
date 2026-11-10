// What the three staff screens share: the header (marked up in each page, so nothing moves as it loads) and the
// settings dialog behind its gear, which is built here. The gate is the shoppers' screen and has neither.
//
// Settings hold only what is not part of a page: what the floor's picture shows, which services are on, and the
// shoppers' data. The catalogue has its own page; the shoppers are listed on the dashboard.

import { esc } from "/static/words.js";

const $ = (id) => document.getElementById(id);

// A staff session that has run out (or was signed out elsewhere): any call answering 401 sends the page to sign in,
// and back here afterwards.
const send = window.fetch.bind(window);
window.fetch = async (...args) => {
  const r = await send(...args);
  if (r.status === 401 && !location.pathname.startsWith("/signin")) toSignIn();
  return r;
};
export function toSignIn() {
  location.assign(`/signin?next=${encodeURIComponent(location.pathname + location.search + location.hash)}`);
}

// The shop the staff are working on, kept across the three screens.
export const currentShop = () => localStorage.getItem("shop") || "";
export const rememberShop = (shop) => localStorage.setItem("shop", shop);

const OVERLAYS = [
  ["show-hands", "Hands", true],
  ["show-boxes", "Boxes round items in hand", true],
  ["show-stock", "Stock on the shelves", true],
  ["show-zone", "Shelf area", false],
];

export function mountShell(page) {
  for (const link of document.querySelectorAll(".pages a")) link.toggleAttribute("aria-current", link.dataset.page === page);
  const dialog = document.createElement("dialog");
  dialog.className = "settings";
  dialog.id = "settings";
  dialog.setAttribute("aria-labelledby", "settings-title");
  dialog.innerHTML = `
    <div class="settings-head"><h2 id="settings-title">Settings</h2><button type="button" id="close-settings">Done</button></div>
    <div class="settings-body">
      <section>
        <h3>Floor picture</h3>
        <fieldset class="toggles"><legend class="sr">Show on the floor picture</legend>
          ${OVERLAYS.map(([id, label]) => `<label class="check"><input type="checkbox" id="${id}" /> ${label}</label>`).join("")}
        </fieldset>
      </section>
      <section>
        <h3>Services</h3>
        <ul class="services" id="services"><li class="hint">Checking</li></ul>
      </section>
      <section>
        <h3>Staff</h3>
        <div class="row between"><p class="hint" id="staff-note">Signed in on this device.</p><button type="button" id="sign-out" hidden>Sign out</button></div>
      </section>
      <section>
        <h3>Shoppers</h3>
        <div class="row between"><p class="hint" id="shopper-count">Registered at the gate.</p><button type="button" class="danger" id="forget" hidden>Delete all shoppers</button></div>
      </section>
    </div>`;
  document.body.append(dialog);
  for (const [id, , on] of OVERLAYS) {
    const box = $(id);
    box.checked = (localStorage.getItem(`overlay-${id}`) ?? (on ? "1" : "0")) === "1";
    box.onchange = () => localStorage.setItem(`overlay-${id}`, box.checked ? "1" : "0");
  }
  $("open-settings").onclick = () => {
    refresh();
    dialog.showModal();
  };
  $("close-settings").onclick = () => dialog.close();
  dialog.addEventListener("click", (e) => {
    if (e.target === dialog) dialog.close(); // a click on the backdrop
  });
  $("sign-out").onclick = async () => {
    await fetch("/api/staff/signout", { method: "POST" });
    location.assign("/signin");
  };
  $("forget").onclick = async () => {
    if (!confirm("Delete every shopper: faces, payment links and visit history? This cannot be undone.")) return;
    const r = await fetch("/api/profiles", { method: "DELETE" });
    if (!r.ok) alert((await r.json()).error);
    refresh();
  };
}

async function refresh() {
  const me = await (await fetch("/api/staff/me")).json().catch(() => ({}));
  $("sign-out").hidden = !me.signed_in;
  $("staff-note").textContent = me.signed_in ? "Signed in on this device." : me.sign_in ? "Not signed in." : "Open on this computer: no staff code is set up.";
  const config = await (await fetch(`/api/config?shop=${encodeURIComponent(currentShop())}`)).json();
  const sv = config.services;
  const row = (on, name, text) => `<li><span class="dot ${on ? "ok" : "bad"}"></span><span><strong>${name}</strong> ${esc(text)}</span></li>`;
  $("services").innerHTML =
    row(sv.faces, "Faces", sv.faces ? (sv.face_search === "pgvector" ? "Postgres vector index" : "Matched on this computer") : "Off: model files missing") +
    row(sv.verifier, "Final check", sv.verifier ? "Before every charge" : "Off") +
    row(sv.payments, "Payments", sv.payments ? "PayPal sandbox" : `Off: ${sv.payments_note}`) +
    row(sv.email, "Invoice email", sv.email ? "On" : "Off");
  const n = config.profiles.length;
  $("shopper-count").innerHTML = n
    ? `${n} registered shopper${n === 1 ? "" : "s"}, ${config.profiles.filter((p) => p.inside).length} in the shop. <a href="/dashboard#customers">See them on the dashboard</a>.`
    : "Nobody has registered yet. Shoppers register at the gate.";
  $("forget").hidden = !n;
}

// The shop's name in the header and the tab title.
export function brand(name, screen) {
  $("brand").textContent = name;
  document.title = `${name}: ${screen}`;
}

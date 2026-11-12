// The dashboard, browser side: the shop's ledger for the people who run it.
//
// Three views off one address: #overview (the totals and two charts), #transactions (every walk-out; one of them
// opens at #transactions/19 with the steps, the pictures and the numbers behind its bill, and the refund buttons),
// and #customers. Everything comes from /api/dashboard; nothing is kept here but what is on screen.

import { brand, mountShell } from "/static/shell.js";
import { TICK, clockTime, dayOf, describe, esc, moneyIn, stamp, unfolded } from "/static/words.js";

const $ = (id) => document.getElementById(id);
mountShell("dashboard");
const D = { data: null, rehearsals: localStorage.getItem("dash-rehearsals") === "1", tx: null };

const money = (n, code) => moneyIn(n, code || D.data?.summary.currency);
const when = (seconds) => `${dayOf(seconds * 1000)}, ${clockTime(seconds * 1000).slice(0, 5)}`;
const plural = (n, word) => `${n} ${word}${n === 1 ? "" : "s"}`;
const face = (who) =>
  who.has_face ? `<img class="face" src="/api/face/${esc(who.id)}" alt="" width="44" height="44" />` : `<span class="face none" aria-hidden="true">${esc((who.name || "?").slice(0, 1).toUpperCase())}</span>`;

function route() {
  const [tab, a, b] = location.hash.replace(/^#/, "").split("/");
  return { tab: ["transactions", "customers"].includes(tab) ? tab : "overview", id: a && !["by", "review"].includes(a) ? Number(a) : null, by: a === "by" ? b : null, review: a === "review" };
}

async function load() {
  D.data = await (await fetch(`/api/dashboard?rehearsals=${D.rehearsals}`)).json();
  brand(D.data.shop, "dashboard");
}

async function show() {
  const r = route();
  for (const link of document.querySelectorAll(".tabs a")) link.toggleAttribute("aria-current", link.dataset.tab === r.tab);
  try {
    await load();
    if (r.tab === "transactions" && r.id) $("dash").innerHTML = await transaction(r.id);
    else $("dash").innerHTML = { overview, transactions, customers }[r.tab](r);
  } catch (e) {
    $("dash").innerHTML = `<p class="empty">The dashboard could not be loaded: ${esc(e.message)}</p>`;
  }
  window.scrollTo(0, 0);
}

// ---------------------------------------------------------------- overview

function overview() {
  const s = D.data.summary;
  const latest = D.data.transactions.filter((t) => !t.rehearsal).slice(0, 5);
  if (!s.transactions && !latest.length)
    return `<section class="blank"><h1>No sales yet</h1><p>Sales show once a shopper leaves with items. Start the <a href="/">shop floor</a> camera</p></section>`;
  const figure = (label, value, note = "") => `<div><dt>${label}</dt><dd>${value}</dd>${note ? `<small>${note}</small>` : ""}</div>`;
  const others = s.other.map((o) => `${plural(o.transactions, "earlier sale")} in ${esc(o.currency.toUpperCase())} (${moneyIn(o.sales, o.currency)})`).join(", ");
  return `
    <dl class="figures">
      ${figure("Sales", money(s.sales), s.refunded ? `${money(s.refunded)} refunded` : s.unpaid ? `${money(s.unpaid)} left unpaid` : "")}
      ${figure("Transactions", s.transactions)}
      ${figure("Items sold", s.items)}
      ${figure("Average basket", money(s.basket))}
      ${figure("Shoppers", s.shoppers, s.inside ? `${s.inside} in store now` : "")}
      ${s.review ? figure("Needs review", `<a href="#transactions/review">${s.review}</a>`) : ""}
    </dl>
    <div class="charts">
      <section>
        <div class="section-head"><h2>Sales by day</h2><span class="sub">${rangeOf(s.by_day)}</span></div>
        ${byDay(s.by_day)}
      </section>
      <section>
        <div class="section-head"><h2>Top products</h2><span class="sub">By units sold</span></div>
        ${topProducts(s.top)}
      </section>
    </div>
    <section>
      <div class="section-head"><h2>Latest transactions</h2><a class="sub" href="#transactions">View all</a></div>
      ${latest.length ? table(latest) : `<p class="empty">No transactions yet.</p>`}
    </section>
    ${others ? `<p class="hint foot-note">Counted in ${esc(s.currency.toUpperCase())}. ${others} ${s.other.length + s.other[0].transactions > 2 ? "are" : "is"} listed under Transactions but not counted here.</p>` : ""}`;
}

const dayLabel = (iso) => {
  const d = new Date(`${iso}T12:00:00`);
  return dayOf(d).replace(/ \d{4}$/, "");
};
const rangeOf = (days) => (days.length ? `${dayLabel(days[0].day)} to ${dayOf(new Date(`${days.at(-1).day}T12:00:00`))}` : "");

// The top of an axis: the next round number above the largest value.
function niceMax(v) {
  if (v <= 0) return 1;
  const p = 10 ** Math.floor(Math.log10(v));
  return [1, 2, 2.5, 5, 10].map((m) => m * p).find((x) => x >= v);
}

// Columns, one per day. Each column is a hover and focus target; the largest carries its value, the rest are read
// off the axis, the tooltip, or the table kept for screen readers.
function byDay(days) {
  const top = niceMax(Math.max(...days.map((d) => d.sales)));
  const best = days.reduce((a, d) => (d.sales > a.sales ? d : a), days[0]);
  const whole = (n) => moneyIn(n, D.data.summary.currency).replace(/\.00$/, "");
  if (!best.sales) return `<p class="empty">No sales in these two weeks.</p>`;
  const cols = days
    .map((d) => {
      const tip = `data-tip-value="${esc(money(d.sales))}" data-tip-label="${esc(`${dayLabel(d.day)}, ${plural(d.transactions, "transaction")}`)}"`;
      const bar = d.sales ? `<span class="fill" style="height:${((d.sales / top) * 100).toFixed(2)}%">${d === best ? `<b>${esc(whole(d.sales))}</b>` : ""}</span>` : "";
      return `<div class="col" tabindex="0" ${tip}>${bar}</div>`;
    })
    .join("");
  const ticks = [1, 0.5].map((k) => `<span style="bottom:${k * 100}%"><i>${esc(whole(top * k))}</i></span>`).join("");
  const axis = days.map((d, i) => `<span${i === days.length - 1 ? ' class="now"' : ""}>${Number(d.day.slice(8))}</span>`).join("");
  const rows = days.map((d) => `<tr><td>${dayLabel(d.day)}</td><td>${money(d.sales)}</td><td>${d.transactions}</td></tr>`).join("");
  return `<div class="columns" role="group" aria-label="Sales by day">
      <div class="plot"><div class="grid">${ticks}</div><div class="cols">${cols}</div></div>
      <div class="axis" aria-hidden="true">${axis}</div>
    </div>
    <table class="sr"><caption>Sales by day</caption><thead><tr><th>Day</th><th>Sales</th><th>Transactions</th></tr></thead><tbody>${rows}</tbody></table>`;
}

// Ranked bars, most copies first, each with its count at the tip.
function topProducts(top) {
  if (!top.length) return `<p class="empty">Nothing sold yet.</p>`;
  const most = top[0].sold;
  return `<ol class="ranked">${top
    .map(
      (p) => `<li tabindex="0" data-tip-value="${esc(money(p.sales))}" data-tip-label="${esc(`${p.name}, ${p.sold} sold`)}">
        <span class="name">${esc(p.name)}</span>
        <span class="track"><span class="fill" style="width:${((p.sold / most) * 100).toFixed(2)}%"></span><b>${p.sold}</b></span>
      </li>`,
    )
    .join("")}</ol>`;
}

// ---------------------------------------------------------------- transactions

function statusOf(t) {
  if (t.rehearsal) return ["Rehearsal", "off"];
  if (t.review) return ["Needs review", "warn"];
  if (t.status === "succeeded") {
    if (t.refunded >= t.amount) return ["Refunded", "off"];
    return [t.refunded ? `Paid, ${money(t.refunded, t.currency)} refunded` : "Paid", "ok"];
  }
  return { nothing_due: ["Not charged", "off"], declined: ["Declined", "bad"], no_card: ["No payment method", "bad"] }[t.status] ?? ["Payment failed", "bad"];
}

function table(rows) {
  return `<div class="scroll"><table class="ledger">
    <thead><tr><th scope="col">When</th><th scope="col">Shopper</th><th scope="col">Items</th><th scope="col">Lines</th><th scope="col" class="num">Total</th><th scope="col">Status</th></tr></thead>
    <tbody>${rows
      .map((t) => {
        const [text, dot] = statusOf(t);
        return `<tr>
          <td><a href="#transactions/${t.id}">${when(t.ended)}</a></td>
          <td>${esc(t.shopper.name)}</td>
          <td class="clip">${t.names.length ? esc(t.names.join(", ")) : `<span class="dim">None charged</span>`}</td>
          <td>${[t.items ? `${t.items} ${t.rehearsal ? "rehearsed" : t.status === "succeeded" ? "charged" : "unpaid"}` : "", t.held ? `${t.held} held` : ""].filter(Boolean).join(", ")}</td>
          <td class="num${t.rehearsal || t.status !== "succeeded" ? " dim" : ""}">${money(t.amount, t.currency)}</td>
          <td><span class="state"><span class="dot ${dot}"></span>${esc(text)}</span></td>
        </tr>`;
      })
      .join("")}</tbody></table></div>`;
}

function transactions(r) {
  const who = r.by ? D.data.customers.find((c) => c.id === r.by) : null;
  const rows = D.data.transactions.filter((t) => (!who || t.shopper.id === who.id) && (!r.review || t.review));
  const title = r.review ? "Needs review" : who ? esc(who.name) : "Transactions";
  return `<section>
    <div class="section-head"><h1>${title}</h1>${who || r.review ? `<a class="sub" href="#transactions">All transactions</a>` : ""}</div>
    <div class="filters"><label class="check"><input type="checkbox" id="rehearsals" ${D.rehearsals ? "checked" : ""} /> Show rehearsals</label></div>
    ${rows.length ? table(rows) : `<p class="empty">${r.review ? "Nothing to review." : who ? "No transactions for this shopper." : "No transactions yet."}</p>`}
  </section>`;
}

// ---------------------------------------------------------------- one transaction

async function transaction(id) {
  const r = await fetch(`/api/dashboard/transactions/${id}`);
  if (!r.ok) return `<p class="back"><a href="#transactions">All transactions</a></p><p class="empty">Transaction ${id} was not found.</p>`;
  D.tx = await r.json();
  return transactionView(D.tx);
}

function transactionView(tx) {
  const d = tx.detail;
  const [status, dot] = statusOf(tx);
  const method = d?.payment?.method || "";
  return `
    <p class="back"><a href="#transactions">All transactions</a></p>
    <header class="tx-head">
      ${face(tx.shopper)}
      <div>
        <h1>${money(tx.amount, tx.currency)} <span>${esc(tx.shopper.name)}</span></h1>
        <p class="sub">Transaction ${tx.id} · ${when(tx.ended)}${method ? ` · ${esc(method)}` : ""}${tx.shop ? ` · ${esc(tx.shop)}` : ""}</p>
      </div>
      <p class="state"><span class="dot ${dot}"></span>${esc(status)}</p>
    </header>
    <div class="tx">
      <div class="story">
        <section>
          <div class="section-head"><h2>Timeline</h2>${d && !d.live ? `<span class="sub">Recorded footage, playback times</span>` : ""}</div>
          ${d ? chain(tx) : `<p class="empty">No timeline was recorded for this transaction.</p>`}
        </section>
        ${
          tx.pictures.length
            ? `<section>
          <div class="section-head"><h2>Evidence</h2><span class="sub">${plural(tx.pictures.length, "picture")}</span></div>
          <div class="proofs">${tx.pictures
            .map((p) => {
              const src = `/api/dashboard/proof/${tx.id}/${p.n}`;
              const at = d?.t0 != null && p.t != null ? clockTime((d.t0 + p.t) * 1000) : "";
              return `<figure><a href="${src}" target="_blank" rel="noopener"><img src="${src}" alt="" loading="lazy" /></a><figcaption>${esc(p.caption)}${at ? ` <time>${at}</time>` : ""}</figcaption></figure>`;
            })
            .join("")}</div>
        </section>`
            : ""
        }
      </div>
      <aside>
        <section>
          <div class="section-head"><h2>Items</h2></div>
          ${bill(tx)}
        </section>
        ${d ? `<section><div class="section-head"><h2>Details</h2></div>${numbers(tx)}</section>` : ""}
      </aside>
    </div>`;
}

// The steps, in three places: the gate, the floor, the door. Times are the time of day each one happened.
function chain(tx) {
  const d = tx.detail;
  const lines = new Map(d.lines.map((l) => [l.id, l]));
  const ctx = { who: (pid) => (pid === d.person ? tx.shopper.name : "Another shopper"), line: (id) => lines.get(id), money: moneyIn };
  const at = (t) => (d.t0 != null ? `<time title="${stamp(t)} into the footage">${clockTime((d.t0 + t) * 1000)}</time>` : `<time>${stamp(t)}</time>`);
  const step = (time, html, minor, done) => `<li class="${minor ? "minor" : ""}">${time}<span>${done ? TICK : ""}<span>${html}</span></span></li>`;
  const floor = [];
  const door = [];
  for (const e of unfolded(d.events)) {
    const [html, minor] = describe(e, ctx);
    const good = (e.type === "audit" && e.verdict === "kept") || (e.type === "payment" && e.status === "succeeded") || (["invoice", "email"].includes(e.type) && ["created", "sent"].includes(e.status));
    (["exit", "check", "audit", "payment", "invoice", "email"].includes(e.type) ? door : floor).push(step(at(e.t), html, minor && !good, good));
  }
  for (const r of tx.refunds) {
    const line = tx.lines.find((l) => l.id === r.purchase);
    const what = line ? ` for ${esc(line.name)}` : "";
    const time = `<time title="${dayOf(r.created * 1000)}">${clockTime(r.created * 1000)}</time>`;
    if (r.status === "failed") door.push(step(time, `Refund of ${moneyIn(r.amount, r.currency)}${what} failed: ${esc(r.message)}`, false, false));
    else door.push(step(time, `Refunded ${moneyIn(r.amount, r.currency)}${what}${r.status === "pending" ? " (pending at PayPal)" : ""}${r.refund ? `, PayPal refund ${esc(r.refund)}` : ""}`, false, true));
  }
  if (d.by_hand) return `<ol class="steps">${floor.join("")}${door.join("")}</ol>`;
  const gate = d.admitted
    ? step(`<time>${clockTime(d.admitted * 1000)}</time>`, `${esc(tx.shopper.name)} recognised at the entry gate`, false, false)
    : d.entered === "floor"
      ? step("<time></time>", "Did not use the entry gate, checked in by the store camera", false, false)
      : "";
  const part = (title, steps) => (steps ? `<h3>${title}</h3><ol class="steps">${steps}</ol>` : "");
  return part("Entry", gate) + part("In store", floor.join("")) + part("Checkout", door.join(""));
}

function bill(tx) {
  const d = tx.detail;
  const can = tx.refundable > 0;
  const rows = tx.lines.map((line) => {
    let note;
    if (line.later) note = `<span class="note took">${TICK}<span>Charged after review in <a href="#transactions/${line.later}">transaction ${line.later}</a></span></span>`;
    else if (line.cleared) note = `<span class="note"><span>Not charged, marked not taken. ${esc(line.note)}</span></span>`;
    else if (line.review) note = `<span class="note flag"><span><b>Needs review.</b> ${esc(line.note)}</span></span>${chargeControl(tx, line)}`;
    else if (!line.charged) note = `<span class="note"><span>Not charged. ${esc(line.note)}</span></span>`;
    else if (line.refunded) note = `<span class="note"><span>Refunded</span></span>`;
    else if (tx.rehearsal || tx.status !== "succeeded") note = `<span class="note"><span>${esc(line.note || "Not charged")}</span></span>`;
    else note = `<span class="note took">${TICK}<span>${esc(line.note || "Charged")}</span></span>`;
    const refund = can && line.charged && !line.refunded ? `<button type="button" class="quiet" data-refund="${line.id}" data-amount="${Math.min(line.price, tx.refundable)}" data-name="${esc(line.name)}">Refund ${money(Math.min(line.price, tx.refundable), tx.currency)}</button>` : "";
    return `<li class="${!line.charged ? "off" : line.refunded ? "gone" : ""}"><span class="name">${esc(line.name)}</span><span class="price">${line.price ? Number(line.price).toFixed(2) : ""}</span>${note}${refund}</li>`;
  });
  // what they picked up and put back before leaving was never on the bill
  for (const line of d?.lines ?? []) if (line.state === "returned") rows.push(`<li class="gone"><span class="name">${esc(line.name)}</span><span class="price"></span><span class="note"><span>Returned to shelf</span></span></li>`);
  const label = tx.rehearsal ? "Rehearsal total" : ["succeeded", "nothing_due"].includes(tx.status) ? "Charged" : "Unpaid";
  const back = tx.refunded ? `<div class="total small"><span>Refunded</span><span class="price">${money(-tx.refunded, tx.currency)}</span></div>` : "";
  const whole = can ? `<button type="button" data-refund="" data-amount="${tx.refundable}">${tx.refunded ? `Refund remaining ${money(tx.refundable, tx.currency)}` : "Refund all"}</button>` : "";
  const inv = d?.payment?.invoice;
  const invoice = inv?.url || tx.receipt_url ? `<a class="button" href="${esc(inv?.url || tx.receipt_url)}" target="_blank" rel="noopener">${inv?.number ? `Invoice ${esc(inv.number)} on PayPal` : "Receipt"}</a>` : "";
  const why = !can && tx.status === "succeeded" && !tx.rehearsal ? `<p class="hint">${esc(tx.refund_note)}</p>` : "";
  return `<ul class="lines">${rows.join("")}</ul>
    <div class="total"><span>${label}</span><span class="price">${money(tx.amount, tx.currency)}</span></div>${back}
    ${whole || invoice ? `<div class="row">${whole}${invoice}</div>` : ""}${why}
    <p class="problem" id="refund-note" role="status"></p>`;
}

// A line held for review is settled by whoever has looked at the evidence: charge it as the item it was, or mark
// it not taken.
function chargeControl(tx, line) {
  const c = tx.charging;
  const clear = `<button type="button" class="quiet" data-clear="${line.id}">Not taken</button>`;
  if (!c?.ok || !c.products.length) return `<span class="charge">${clear}</span>`;
  const options = c.products.map((p) => `<option value="${esc(p.id)}" ${p.id === line.product ? "selected" : ""}>${esc(p.name)}, ${moneyIn(p.price, c.currency)}</option>`).join("");
  const known = c.products.some((p) => p.id === line.product);
  return `<span class="charge"><select data-charge-pick="${line.id}" aria-label="Item"><option value="">Select item</option>${options}</select><button type="button" data-charge="${line.id}" ${known ? "" : "disabled"}>Charge</button>${clear}</span>`;
}

async function clearLine(button) {
  const tx = D.tx;
  const line = tx.lines.find((l) => l.id === Number(button.dataset.clear));
  if (!line || !confirm(`Mark ${line.name} as not taken? It stays uncharged and leaves the review queue.`)) return;
  let problem = "";
  try {
    const r = await fetch(`/api/dashboard/transactions/${tx.id}/clear`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ line: line.id }) });
    const body = await r.json();
    if (!r.ok) problem = body.error || "The line could not be updated.";
    if (body.transaction) D.tx = body.transaction;
  } catch (e) {
    problem = `The line could not be updated: ${e.message}`;
  }
  $("dash").innerHTML = transactionView(D.tx);
  $("refund-note").textContent = problem;
}

async function chargeLater(button) {
  const tx = D.tx;
  const line = Number(button.dataset.charge);
  const picked = document.querySelector(`select[data-charge-pick="${line}"]`).value;
  const product = tx.charging.products.find((p) => p.id === picked);
  if (!product) return;
  if (!confirm(`Charge ${moneyIn(product.price, tx.charging.currency)} for ${product.name} to ${tx.shopper.name}'s saved payment method? An invoice is emailed to them.`)) return;
  for (const b of document.querySelectorAll("[data-charge], [data-refund]")) b.disabled = true;
  let problem = "";
  try {
    const r = await fetch(`/api/dashboard/transactions/${tx.id}/charge`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ line, product: product.id }) });
    const body = await r.json();
    if (!r.ok) problem = body.error || "The charge did not go through.";
    if (body.transaction) D.tx = body.transaction;
  } catch (e) {
    problem = `The charge could not be sent: ${e.message}`;
  }
  $("dash").innerHTML = transactionView(D.tx);
  $("refund-note").textContent = problem;
}

function numbers(tx) {
  const d = tx.detail;
  const n = d.numbers ?? {};
  const check = d.payment?.check ?? {};
  const row = (label, value) => (value == null || value === "" ? "" : `<div><dt>${label}</dt><dd>${value}</dd></div>`);
  const left = n.left_how === "ended" ? "End of footage" : n.leave_after_s ? `Out of view for ${n.leave_after_s} s` : "";
  const checked = { checked: `${check.kept} of ${(check.kept ?? 0) + (check.held ?? 0)} lines confirmed, ${plural(check.pictures ?? 0, "picture")}`, failed: "No answer, nothing charged", skipped: "Not run" }[check.status];
  return `<dl class="numbers">
    ${row("Frames analysed", n.frames ? `${n.frames}${n.model_ms ? `, ${n.model_ms} ms each` : ""}` : "")}
    ${row("Time in view", n.in_view_s != null ? `${n.in_view_s} s` : "")}
    ${row("Face match, store camera", n.face_match ? `${Math.round(n.face_match * 100)}%` : "")}
    ${row("Hand readings", n.hand_looks || "")}
    ${row("Item readings", n.item_looks || "")}
    ${row("Exit trigger", left)}
    ${row("Final check", checked ?? "")}
    ${row("Exit to payment", n.door_s != null ? `${n.door_s} s` : "")}
    ${row("PayPal capture", tx.payment && !tx.rehearsal ? `<code>${esc(tx.payment)}</code>` : "")}
    ${row("PayPal order", d.payment?.order ? `<code>${esc(d.payment.order)}</code>` : "")}
    ${row("Invoice email", d.payment?.email ? (d.payment.email.status === "sent" ? `Sent to ${esc(d.payment.email.to)}` : `Not sent: ${esc(d.payment.email.message)}`) : "")}
    ${row("Session", `<code>${esc(tx.session)}</code>`)}
  </dl>`;
}

async function refund(button) {
  const tx = D.tx;
  const line = button.dataset.refund ? Number(button.dataset.refund) : null;
  const amount = money(Number(button.dataset.amount), tx.currency);
  const to = tx.detail?.payment?.method || "the original payment method";
  if (!confirm(`Refund ${amount}${line ? ` for ${button.dataset.name}` : ""} to ${to}? This cannot be undone.`)) return;
  for (const b of document.querySelectorAll("[data-refund]")) b.disabled = true;
  let problem = "";
  try {
    const r = await fetch(`/api/dashboard/transactions/${tx.id}/refund`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(line ? { line } : {}) });
    const body = await r.json();
    if (!r.ok) problem = body.error || "The refund did not go through.";
    if (body.transaction) D.tx = body.transaction;
  } catch (e) {
    problem = `The refund could not be sent: ${e.message}`;
  }
  $("dash").innerHTML = transactionView(D.tx);
  $("refund-note").textContent = problem;
}

// ---------------------------------------------------------------- customers

function customers() {
  const list = D.data.customers;
  if (!list.length) return `<section class="blank"><h1>No customers yet</h1><p>Shoppers register at the <a href="/gate" target="_blank" rel="noopener">entry gate</a> with their face, an email and PayPal</p></section>`;
  return `<section>
    <div class="section-head"><h1>Customers</h1><span class="sub">${plural(list.length, "registered shopper")}</span></div>
    <div class="scroll"><table class="ledger people">
      <thead><tr><th scope="col">Shopper</th><th scope="col">Payment method</th><th scope="col">Top items</th><th scope="col" class="num">Transactions</th><th scope="col" class="num">Spent</th><th scope="col">Last bought</th></tr></thead>
      <tbody>${list
        .map(
          (c) => `<tr>
            <td><span class="person">${face(c)}<span><a href="#transactions/by/${esc(c.id)}">${esc(c.name)}</a><small>${c.inside ? `<span class="dot ok"></span>In store now` : esc(c.email ?? "")}</small></span></span></td>
            <td>${esc(c.method || "None")}</td>
            <td>${c.buys.length ? c.buys.map((b) => `${esc(b.name)}${b.times > 1 ? ` <span class="times">${b.times} times</span>` : ""}`).join("<br />") : `<span class="dim">None</span>`}</td>
            <td class="num">${c.transactions}</td>
            <td class="num">${money(c.spent)}</td>
            <td>${c.last ? when(c.last) : `<span class="dim">Never</span>`}</td>
          </tr>`,
        )
        .join("")}</tbody></table></div>
  </section>`;
}

// ---------------------------------------------------------------- wiring

// One tooltip for every chart mark: the value first, then what it is. The same on hover and on keyboard focus.
const tip = $("tip");
function showTip(el) {
  const value = document.createElement("strong");
  value.textContent = el.dataset.tipValue;
  const label = document.createElement("span");
  label.textContent = el.dataset.tipLabel;
  tip.replaceChildren(value, label);
  tip.hidden = false;
  const box = el.getBoundingClientRect();
  const x = Math.min(window.innerWidth - tip.offsetWidth - 8, Math.max(8, box.left + box.width / 2 - tip.offsetWidth / 2));
  tip.style.left = `${x + window.scrollX}px`;
  tip.style.top = `${box.top + window.scrollY - tip.offsetHeight - 8}px`;
}
const marked = (e) => e.target.closest?.("[data-tip-value]");
document.addEventListener("pointerover", (e) => (marked(e) ? showTip(marked(e)) : (tip.hidden = true)));
document.addEventListener("focusin", (e) => (marked(e) ? showTip(marked(e)) : (tip.hidden = true)));
document.addEventListener("focusout", () => (tip.hidden = true));

document.addEventListener("click", (e) => {
  const button = e.target.closest?.("button[data-refund]");
  if (button) return refund(button);
  const charge = e.target.closest?.("button[data-charge]");
  if (charge) return chargeLater(charge);
  const clear = e.target.closest?.("button[data-clear]");
  if (clear) return clearLine(clear);
  // a click anywhere on a row opens what its link opens
  const row = e.target.closest?.(".ledger tbody tr");
  const link = row?.querySelector("a[href^='#']");
  if (link && !e.target.closest("a")) location.hash = link.getAttribute("href");
});
document.addEventListener("change", (e) => {
  if (e.target.dataset.chargePick) return void (document.querySelector(`button[data-charge="${e.target.dataset.chargePick}"]`).disabled = !e.target.value);
  if (e.target.id !== "rehearsals") return;
  D.rehearsals = e.target.checked;
  localStorage.setItem("dash-rehearsals", D.rehearsals ? "1" : "0");
  show();
});
window.addEventListener("hashchange", show);
// back from the shop floor in another tab: the numbers may have moved
document.addEventListener("visibilitychange", () => {
  if (!document.hidden && !(route().tab === "transactions" && route().id)) show();
});
show();

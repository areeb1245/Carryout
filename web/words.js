// How the shop's events and bag lines are put into words. The shop floor (app.js) and the dashboard (dashboard.js)
// both say them, so they are written once, here.

export const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

export const TICK = `<svg class="mark" viewBox="0 0 12 12" aria-hidden="true"><path d="M2 6.5l2.6 2.6L10 3.4" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg>`;

// seconds of footage as 0:07
export const stamp = (t) => `${Math.floor(t / 60)}:${String(Math.floor(t % 60)).padStart(2, "0")}`;

const two = (n) => String(n).padStart(2, "0");
const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
// a moment (milliseconds, or a Date) as 14:12:31, and as 10 Nov 2026
export const clockTime = (ms) => ((d) => `${two(d.getHours())}:${two(d.getMinutes())}:${two(d.getSeconds())}`)(new Date(ms));
export const dayOf = (ms) => ((d) => `${d.getDate()} ${MONTHS[d.getMonth()]} ${d.getFullYear()}`)(new Date(ms));

const formats = {};
// "$249.00" in the currency a payment was made in
export function moneyIn(n, code = "USD") {
  code = String(code || "USD").toUpperCase();
  try {
    formats[code] ??= new Intl.NumberFormat("en", { style: "currency", currency: code });
    return formats[code].format(Number(n || 0));
  } catch {
    return `${code} ${Number(n || 0).toFixed(2)}`;
  }
}

// how sure the camera is of a line, to the nearest 5%: the figure wobbles from frame to frame, the page should not
const sureness = (line) => Math.round(line.confidence * 20) * 5;

// What the bag says under a line: [text, "" | "took" | "flag", the longer story for a tooltip].
export function lineNote(line) {
  if (line.state === "returned") return ["Returned to shelf", ""];
  if (line.state === "given") return ["Handed to another shopper", ""];
  if (line.product && !line.billable && !line.unread) return ["Shopper's own item", ""];
  // the final check has the last word on a line
  if (line.audit === "kept") return [line.audit_why || "Confirmed", "took"];
  if (line.audit === "put_back") return [`Held for review: ${line.audit_why}`, "flag"];
  if (line.audit === "unclear") return [`Held for review: ${line.audit_why}`, "flag"];
  if (line.paid) return ["Paid", "took"];
  if (line.verified === "disputed") return ["Held for review: no item seen in hand", "flag", line.why];
  if (line.unread) return ["Not yet identified", "flag"];
  if (line.repeat) return ["Duplicate of another line, not charged", "flag"];
  if (line.contested) return [`Held for review: seen in hand as ${line.contested}`, "flag"];
  if (line.billable && !line.charge && line.verified == null) return ["Awaiting confirmation in hand", "flag"];
  if (!line.product) return ["Not in catalogue", "flag"];
  if (!line.charge) return [`Low confidence (${sureness(line)}%)`, "flag"];
  // on the bill and still in the shop: one short line, with how it got there kept for the tooltip
  const how = { carried_in: "In hand on arrival", inferred: "Taken from the shelf", handover: "Handed over by another shopper", seen: "First seen in hand" }[line.origin];
  const story = [how, line.why].filter(Boolean).join(". ");
  if (line.verified === "confirmed") return ["Confirmed in hand", "", story];
  return [`${sureness(line)}% confidence`, "", story];
}

// What was read, without the "how it was read" lead-in the reason starts with.
const evidence = (why) => String(why ?? "").replace(/^(Read in hand twice|Read in hand at exit|Identified on the shelf, confirmed in hand): /, "");

// The question one step of the final check asked, in words.
const checkQuestion = (e, item) => ({ leaving: "In the hands as they leave?", held: `${item} in the hand?`, shelf: "Item put on the shelf?", which: "Unreadable book in the hand looks like?", placed: "Book going onto the shelf looks like?" })[e.ask] ?? "";

// One event as [html, is it a minor one]. `ctx` knows the scene: who(person id) is a name, line(entry id) is the
// bag line an event is about, money(amount, currency) writes a sum.
export function describe(e, ctx) {
  const who = esc(ctx.who(e.person));
  const line = ctx.line(e.entry);
  const item = line?.unread ? `an unidentified ${esc(line.name.toLowerCase())}` : esc(line?.name ?? "an item");
  const own = line?.product && !line.billable && !line.unread;
  const money = ctx.money;
  const method = esc(e.method || "saved payment method");
  switch (e.type) {
    case "enter":
      return [`${who} entered`, true];
    case "reenter":
      return [`${who} back in view`, true];
    case "by_hand":
      return [`${esc(e.name)} charged after review of <a href="#transactions/${Number(e.of)}">transaction ${Number(e.of)}</a>`, false];
    case "same":
      return [`${who} recognised again, cart continued`, false];
    case "checkin":
      return [`${esc(e.name)} recognised${e.how === "floor" ? " in store, checked in without the gate" : ""}${e.score ? ` (${Math.round(e.score * 100)}% face match)` : ""}. ${e.method ? `Pays with ${esc(e.method)}` : "No payment method on file"}`, false];
    case "unregistered":
      return [e.name ? `${who} resembles ${esc(e.name)}, match too weak to charge` : `${who} is not registered. No charge`, false];
    case "audit":
      if (e.verdict === "kept") return [`Final check: ${item} confirmed (${esc(e.why)})`, true];
      if (e.verdict === "put_back") return [`Final check: ${item} held for review (${esc(e.why)})`, false];
      return [`Final check: ${item} not confirmed, held for review (${esc(e.why)})`, false];
    case "check":
      return [`Final check, step ${Number(e.step)}: ${checkQuestion(e, e.name ? esc(e.name) : item)} ${esc(e.answer)}${e.why ? ` (${esc(e.why)})` : ""}`, true];
    case "invoice":
      return e.status === "created" ? [`PayPal invoice ${esc(e.number ?? "")} created`, true] : [`PayPal invoice not created: ${esc(e.message)}`, false];
    case "pick":
      if (own) return [`${who} is holding their own ${item.toLowerCase()}`, true];
      if (e.origin === "carried_in") return [`${who} entered holding ${item}`, false];
      if (e.origin === "inferred") return [`${who} took ${item} (inferred from the shelf)`, false];
      if (e.origin === "seen") return [`${who} is carrying ${item}`, false];
      return [`${who} took ${item}`, false];
    case "return":
      return [`${who} returned ${item} to the shelf`, false];
    case "handover":
      return [`${esc(ctx.who(e.from_person))} handed ${item} to ${who}`, false];
    case "verify":
      if (e.verdict === "disputed") return [`${item} disputed: ${esc(e.why)}`, false];
      if (e.verdict === "confirmed") return [`Identified ${item} in hand (${esc(evidence(e.why))})`, true];
      return [`${item} not yet identified: ${esc(e.why)}`, true];
    case "exit":
      if (e.how === "ended") return [`Footage ended with ${who} inside, treated as an exit`, true];
      return [`${who} left${e.after_s ? ` (out of view for ${e.after_s} s)` : ""}`, true];
    case "payment":
      if (e.status === "succeeded") return [`Charged ${money(e.amount, e.currency)} to ${method}`, false];
      if (e.status === "nothing_due") return ["Nothing to charge", true];
      if (e.status === "dry") return [`Rehearsal: ${money(e.amount, e.currency)} not charged`, false];
      if (e.status === "not_registered") return [`${money(e.amount, e.currency)} not charged: shopper not identified`, false];
      if (e.status === "declined") return [`${method} declined for ${money(e.amount, e.currency)}`, false];
      if (e.status === "no_card") return [`${money(e.amount, e.currency)} not charged: no payment method on file`, false];
      return [`Not charged: ${esc(e.message)}`, false];
    case "email":
      return [e.status === "sent" ? `Invoice emailed to ${esc(e.to)}` : `Invoice email to ${esc(e.to)} failed: ${esc(e.message)}`, e.status === "sent"];
    default:
      return [esc(e.type), true];
  }
}

// Events as the page lists them: a pick that turned out to be an item already on the bag under another track
// is folded away, with what was said about it.
export function unfolded(events) {
  const folded = new Set(events.filter((e) => e.type === "void").map((e) => e.entry));
  return events.filter((e) => e.type !== "void" && !(folded.has(e.entry) && ["pick", "verify"].includes(e.type)));
}

"""
The ledger: every walk-out, with what it takes to explain the bill afterwards, and to give money back if it was wrong.

The rows live in the same SQLite file as the profiles (identity.py). A walk-out is written in two goes: `record_visit`
the moment the payment has been attempted, so a charge is never left without a row, and `finish` once the invoice and
the email are through. What is kept for each one:

    the lines     what was on the bill, and what the shopper had that was kept off it, with the reason
    the steps     every event of that shopper's visit, in order: recognised, took, read in the hands, walked out,
                  the check before charging line by line, the charge, the invoice, the email
    the pictures  the moment each item left the shelf, and the pictures the check before charging was shown
    the numbers   frames analysed, looks by the second model, the face match, seconds from walking out to the charge

The dashboard (/dashboard) reads it back through `summary`, `transactions`, `transaction` and `customers`. Rehearsals
are kept, flagged, and never counted as sales.

Two things can be put right from there. Money that should not have been taken goes back (`refundable`, `add_refund`).
And an item a shopper left with that was held back, because nothing could confirm what it was, can be charged by a
person who has looked at the pictures (`chargeable`, `charge_later`): a transaction of its own, with its own invoice.
"""
from __future__ import annotations

import json
import time
from datetime import date, datetime, timedelta

from .identity import Profiles
from .payments import Card

#: refunds in these states count against what is left to refund
BACK = ("refunded", "pending")
_BACK = "('refunded', 'pending')"
#: a line is refunded when it was refunded itself, or the whole bill was
_KEPT = f"NOT EXISTS (SELECT 1 FROM refunds r WHERE r.visit = v.id AND r.status IN {_BACK} AND (r.purchase = p.id OR r.purchase IS NULL))"
_SOLD = "v.dry = 0 AND v.status = 'succeeded' AND p.charged = 1"


class Ledger:
    def __init__(self, book: Profiles):
        self.book, self.db = book, book.db

    # ------------------------------------------------------------------ writing

    def add_proofs(self, visit: int, proofs: list[dict]) -> None:
        """`proofs` are {kind, caption, t, jpeg}: pictures a bill rests on, in the order they should be shown."""
        self.db.executemany("INSERT INTO proofs (visit, kind, caption, t, jpeg) VALUES (?, ?, ?, ?, ?)", [(visit, p["kind"], p["caption"], p.get("t"), p["jpeg"]) for p in proofs])
        self.db.commit()

    def finish(self, visit: int, payment: dict, detail: dict) -> None:
        """The walk-out is through: how the payment ended, the invoice address, and the steps that led to the bill."""
        self.db.execute(
            "UPDATE visits SET status = ?, payment = ?, receipt_url = ?, detail = ? WHERE id = ?",
            (payment["status"], payment.get("id"), payment.get("receipt_url"), json.dumps(detail, separators=(",", ":"), default=str), visit),
        )
        self.db.commit()

    def add_refund(self, visit: int, purchase: int | None, amount: float, currency: str, result: dict, reason: str = "") -> int:
        cur = self.db.execute(
            "INSERT INTO refunds (visit, purchase, amount, currency, status, refund, reason, message, created) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (visit, purchase, round(amount, 2), currency.lower(), result["status"], result.get("id"), reason[:200], result.get("message", "")[:200], time.time()),
        )
        self.db.commit()
        return cur.lastrowid

    # ------------------------------------------------------------------ refunds

    def refundable(self, visit: int, purchase: int | None = None) -> tuple[float, str]:
        """(how much can be given back, or 0 and why not). For one line, its price; for the bill, what is left of it."""
        row = self.db.execute("SELECT amount, status, payment, dry, provider FROM visits WHERE id = ?", (visit,)).fetchone()
        if row is None:
            return 0.0, "There is no such transaction."
        amount, status, payment, dry, provider = row
        if dry:
            return 0.0, "A rehearsal charged nothing, so there is nothing to refund."
        if status != "succeeded" or not payment:
            return 0.0, "Nothing was charged for this transaction."
        if (provider or self.book.provider) != self.book.provider:
            return 0.0, f"This was paid through {str(provider).capitalize()}, in an earlier version of the shop. Refund it there."
        back = self.db.execute(f"SELECT COALESCE(SUM(amount), 0) FROM refunds WHERE visit = ? AND status IN {_BACK}", (visit,)).fetchone()[0]
        left = round(amount - back, 2)
        if left <= 0:
            return 0.0, "This transaction has been refunded in full."
        if purchase is None:
            return left, ""
        line = self.db.execute("SELECT price, charged FROM purchases WHERE id = ? AND visit = ?", (purchase, visit)).fetchone()
        if line is None or not line[1]:
            return 0.0, "That line was not charged."
        if self.db.execute(f"SELECT 1 FROM refunds WHERE visit = ? AND purchase = ? AND status IN {_BACK}", (visit, purchase)).fetchone():
            return 0.0, "That line has been refunded already."
        return round(min(line[0] or 0.0, left), 2), ""

    # ------------------------------------------------------------------ charging a held-back line afterwards

    def chargeable(self, visit: int, purchase: int | None = None) -> tuple[bool, str]:
        """Can a held-back line of this walk-out still be charged by hand? (yes, or no and why not)."""
        row = self.db.execute("SELECT profile, dry FROM visits WHERE id = ?", (visit,)).fetchone()
        if row is None:
            return False, "There is no such transaction."
        if row[1]:
            return False, "This was a rehearsal: nobody is charged for it."
        who = self.book.profiles.get(row[0])
        if who is None or not self.book.can_pay(who):
            return False, "This shopper has no saved payment method that can be charged."
        if purchase is None:
            return True, ""
        line = self.db.execute("SELECT charged, later, cleared FROM purchases WHERE id = ? AND visit = ?", (purchase, visit)).fetchone()
        if line is None:
            return False, "There is no such line on this transaction."
        if line[0]:
            return False, "That line was charged when they walked out."
        if line[1]:
            return False, "That line has been charged already."
        if line[2]:
            return False, "That line was marked as not taken."
        return True, ""

    def clear(self, visit: int, purchase: int) -> tuple[bool, str]:
        """Someone at the shop has looked at the pictures of a held-back line and says it was not taken: it leaves
        the review queue and stays on record as not charged."""
        row = self.db.execute("SELECT p.charged, p.later, p.cleared, v.dry FROM purchases p JOIN visits v ON v.id = p.visit WHERE p.id = ? AND p.visit = ?", (purchase, visit)).fetchone()
        if row is None:
            return False, "There is no such line on this transaction."
        charged, later, cleared, dry = row
        if charged or later:
            return False, "That line has been charged."
        if cleared or dry:
            return False, "That line needs no review."
        self.db.execute("UPDATE purchases SET cleared = ? WHERE id = ?", (time.time(), purchase))
        self.db.commit()
        return True, ""

    async def charge_later(self, visit: int, purchase: int, product, currency: str, shop: str, pay, mailer=None) -> dict:
        """Someone at the shop has looked at the pictures and says which item a held-back line was: charge it to the
        shopper's saved method, as a transaction of its own that points back at the walk-out. `product` is the
        catalogue entry. Returns {status: succeeded | declined | failed | refused, visit, message}."""
        ok, why_not = self.chargeable(visit, purchase)
        if not ok:
            return {"status": "refused", "visit": None, "message": why_not}
        if product is None or not product.billable or product.head or product.price <= 0:
            return {"status": "refused", "visit": None, "message": "Choose which item it was."}
        if pay is None or not pay.enabled:
            return {"status": "refused", "visit": None, "message": pay.problem if pay else "Payments are not set up."}
        book = self.book
        who = book.profiles[self.db.execute("SELECT profile FROM visits WHERE id = ?", (visit,)).fetchone()[0]]
        lines = [(product.name, product.price, " · ".join(x for x in (product.category, product.section) if x))]
        extra = {"lines": lines} if getattr(pay, "name", "") == "paypal" else {}
        card = Card(who.customer, who.payment_method, who.brand or "", who.last4 or "")
        # the key names the line and the item, so pressing twice is one charge
        result = await pay.charge(card, product.price, currency, f"Charged after your visit: {product.name}"[:200], f"later-{visit}-{purchase}-{product.id}", {"app": "sam3-supermarket", "shopper": who.id}, **extra)
        if result["status"] != "succeeded":
            return {"status": result["status"], "visit": None, "message": result.get("message") or "The payment did not go through."}
        payment = {**result, "items": 1, "check": {"status": "by_hand"}, "name": who.name, "brand": who.brand, "last4": who.last4, "method": book.method_name(who.brand, who.last4, who.payer)}
        new = book.record_visit(who, f"later-{visit}", payment, [(product.id, product.name, product.price, f"Charged after review of transaction {visit}")], shop=shop)
        self.db.execute("UPDATE purchases SET later = ? WHERE id = ?", (new, purchase))
        self.db.commit()
        started = time.time()
        events = [
            {"id": 1, "t": 0.0, "type": "by_hand", "person": 0, "name": product.name, "of": visit},
            {"id": 2, "t": 0.0, "type": "payment", "person": 0, "payment": payment["id"], **{k: payment.get(k) for k in ("status", "amount", "currency", "items", "method", "message")}},
        ]

        def detail() -> dict:
            kept = {k: payment.get(k) for k in ("status", "amount", "currency", "id", "order", "method", "message", "check", "invoice", "email")}
            return {"person": 0, "live": True, "t0": started, "admitted": None, "by_hand": {"of": visit, "line": purchase}, "lines": [], "events": events, "payment": kept, "numbers": {}}

        self.finish(new, payment, detail())
        try:
            if callable(getattr(pay, "invoice", None)):
                made = await pay.invoice(who.name, who.email or "", lines, currency, payment["id"], shop, f"later-{visit}-{purchase}")
                payment["invoice"], payment["receipt_url"] = made, made.get("url") or payment.get("receipt_url")
                events.append({"id": 3, "t": round(time.time() - started, 2), "type": "invoice", "person": 0, "status": made["status"], "number": made.get("number"), "url": made.get("url"), "message": made.get("message", "")})
            if who.email and mailer is not None:
                sent = await mailer.receipt(who.email, who.name, lines, payment, f"receipt-{payment['id']}")
                payment["email"] = sent
                events.append({"id": 4, "t": round(time.time() - started, 2), "type": "email", "person": 0, "status": sent["status"], "to": sent["to"], "message": sent["message"]})
        finally:
            self.finish(new, payment, detail())
        return {"status": "succeeded", "visit": new, "message": ""}

    # ------------------------------------------------------------------ reading

    def transactions(self, rehearsals: bool = False, profile: str | None = None, limit: int = 300) -> list[dict]:
        """Walk-outs that had something on the bag, newest first."""
        where, args = ["EXISTS (SELECT 1 FROM purchases p WHERE p.visit = v.id)"], []
        if not rehearsals:
            where.append("v.dry = 0")
        if profile:
            where.append("v.profile = ?")
            args.append(profile)
        rows = self.db.execute(
            f"""SELECT v.id, v.profile, v.ended, v.amount, v.currency, v.status, v.dry, v.provider, v.detail IS NOT NULL,
                       (SELECT COUNT(*) FROM purchases p WHERE p.visit = v.id AND p.charged = 1),
                       (SELECT COUNT(*) FROM purchases p WHERE p.visit = v.id AND p.charged = 0),
                       (SELECT COUNT(*) FROM purchases p WHERE p.visit = v.id AND p.charged = 0 AND p.later IS NULL AND p.cleared IS NULL AND v.dry = 0),
                       (SELECT COALESCE(SUM(amount), 0) FROM refunds r WHERE r.visit = v.id AND r.status IN {_BACK}),
                       (SELECT GROUP_CONCAT(name, '\x1f') FROM (SELECT name FROM purchases p WHERE p.visit = v.id AND p.charged = 1 ORDER BY p.id)),
                       (SELECT COUNT(*) FROM proofs f WHERE f.visit = v.id)
                FROM visits v WHERE {' AND '.join(where)} ORDER BY v.id DESC LIMIT ?""",
            (*args, limit),
        ).fetchall()
        out = []
        for vid, pid, ended, amount, currency, status, dry, provider, recorded, items, held, review, back, names, pictures in rows:
            who = self.book.profiles.get(pid)
            out.append({
                "id": vid, "ended": ended, "amount": amount, "currency": currency, "status": status, "rehearsal": bool(dry),
                "provider": provider, "recorded": bool(recorded), "items": items, "held": held, "review": review, "refunded": round(back, 2),
                "names": names.split("\x1f") if names else [], "pictures": pictures,
                "shopper": {"id": pid, "name": who.name if who else "Deleted shopper", "has_face": bool(who and who.has_thumb)},
            })
        return out

    def transaction(self, visit: int) -> dict | None:
        """One walk-out with everything kept about it."""
        row = self.db.execute("SELECT id, profile, session, ended, amount, currency, status, payment, receipt_url, dry, provider, shop, detail FROM visits WHERE id = ?", (visit,)).fetchone()
        if row is None:
            return None
        vid, pid, session, ended, amount, currency, status, payment, receipt_url, dry, provider, shop, detail = row
        who = self.book.profiles.get(pid)
        refunds = [
            {"id": r[0], "purchase": r[1], "amount": r[2], "currency": r[3], "status": r[4], "refund": r[5], "reason": r[6], "message": r[7], "created": r[8]}
            for r in self.db.execute("SELECT id, purchase, amount, currency, status, refund, reason, message, created FROM refunds WHERE visit = ? ORDER BY id", (vid,))
        ]
        whole = any(r["purchase"] is None and r["status"] in BACK for r in refunds)
        gone = {r["purchase"] for r in refunds if r["status"] in BACK}
        lines = [
            {"id": p[0], "product": p[1], "name": p[2], "price": p[3], "charged": bool(p[4]), "note": p[5] or "", "later": p[6], "cleared": p[7],
             "review": not p[4] and not p[6] and not p[7] and not dry, "refunded": bool(p[4]) and (whole or p[0] in gone)}
            for p in self.db.execute("SELECT id, product, name, price, charged, note, later, cleared FROM purchases WHERE visit = ? ORDER BY charged DESC, id", (vid,))
        ]
        left, why_not = self.refundable(vid)
        return {
            "id": vid, "session": session, "ended": ended, "amount": amount, "currency": currency, "status": status, "payment": payment,
            "receipt_url": receipt_url, "rehearsal": bool(dry), "provider": provider, "shop": shop,
            "shopper": {"id": pid, "name": who.name if who else "Deleted shopper", "email": who.email if who else None, "has_face": bool(who and who.has_thumb)},
            "lines": lines, "review": sum(1 for line in lines if line["review"]), "refunds": refunds, "refunded": round(sum(r["amount"] for r in refunds if r["status"] in BACK), 2),
            "refundable": left, "refund_note": why_not,
            "pictures": [{"n": n, "kind": f[0], "caption": f[1], "t": f[2]} for n, f in enumerate(self.db.execute("SELECT kind, caption, t FROM proofs WHERE visit = ? ORDER BY id", (vid,)))],
            "detail": json.loads(detail) if detail else None,
        }

    def proof(self, visit: int, n: int) -> bytes | None:
        row = self.db.execute("SELECT jpeg FROM proofs WHERE visit = ? ORDER BY id LIMIT 1 OFFSET ?", (visit, max(0, n))).fetchone()
        return row[0] if row else None

    def summary(self, currency: str, days: int = 14) -> dict:
        """The shop at a glance, for sales in `currency`: totals, sales by day, the products that sell most."""
        cur = currency.lower()
        paid = self.db.execute("SELECT id, ended, amount FROM visits v WHERE v.dry = 0 AND v.status = 'succeeded' AND lower(v.currency) = ? AND v.amount > 0", (cur,)).fetchall()
        sales = round(sum(r[2] for r in paid), 2)
        back = self.db.execute(f"SELECT COALESCE(SUM(r.amount), 0) FROM refunds r JOIN visits v ON v.id = r.visit WHERE r.status IN {_BACK} AND v.dry = 0 AND lower(v.currency) = ?", (cur,)).fetchone()[0]
        top = self.db.execute(
            f"""SELECT p.name, COUNT(*), SUM(p.price) FROM purchases p JOIN visits v ON v.id = p.visit
                WHERE {_SOLD} AND lower(v.currency) = ? AND {_KEPT} GROUP BY p.name ORDER BY COUNT(*) DESC, SUM(p.price) DESC, p.name""",
            (cur,),
        ).fetchall()
        today = date.today()
        by_day = {today - timedelta(days=k): [0.0, 0] for k in range(days - 1, -1, -1)}
        for _, ended, amount in paid:
            day = datetime.fromtimestamp(ended).date()
            if day in by_day:
                by_day[day][0] += amount
                by_day[day][1] += 1
        unpaid = self.db.execute("SELECT COALESCE(SUM(amount), 0) FROM visits WHERE dry = 0 AND lower(currency) = ? AND status IN ('declined', 'failed', 'no_card')", (cur,)).fetchone()[0]
        other = self.db.execute("SELECT lower(currency), COUNT(*), SUM(amount) FROM visits WHERE dry = 0 AND status = 'succeeded' AND amount > 0 AND lower(currency) != ? GROUP BY lower(currency)", (cur,)).fetchall()
        people = list(self.book.profiles.values())
        review = self.db.execute("SELECT COUNT(DISTINCT v.id) FROM purchases p JOIN visits v ON v.id = p.visit WHERE v.dry = 0 AND p.charged = 0 AND p.later IS NULL AND p.cleared IS NULL").fetchone()[0]
        return {
            "review": review,
            "currency": cur,
            "sales": sales,
            "refunded": round(back, 2),
            "transactions": len(paid),
            "items": sum(r[1] for r in top),
            "basket": round(sales / len(paid), 2) if paid else 0.0,
            "unpaid": round(unpaid, 2),
            "shoppers": len(people),
            "inside": sum(1 for p in people if self.book.open_visit(p) is not None),
            "by_day": [{"day": day.isoformat(), "sales": round(v[0], 2), "transactions": v[1]} for day, v in by_day.items()],
            "top": [{"name": r[0], "sold": r[1], "sales": round(r[2] or 0, 2)} for r in top[:8]],
            "other": [{"currency": r[0], "transactions": r[1], "sales": round(r[2], 2)} for r in other],
        }

    def customers(self, currency: str) -> list[dict]:
        """Every registered shopper with what they have spent in `currency` and what they buy, biggest spender first."""
        cur = currency.lower()
        out = []
        for person in self.book.profiles.values():
            count, spent, last = self.db.execute(
                "SELECT COUNT(*), COALESCE(SUM(amount), 0), MAX(ended) FROM visits v WHERE profile = ? AND dry = 0 AND status = 'succeeded' AND amount > 0 AND lower(currency) = ?", (person.id, cur)
            ).fetchone()
            back = self.db.execute(f"SELECT COALESCE(SUM(r.amount), 0) FROM refunds r JOIN visits v ON v.id = r.visit WHERE v.profile = ? AND v.dry = 0 AND lower(v.currency) = ? AND r.status IN {_BACK}", (person.id, cur)).fetchone()[0]
            buys = self.db.execute(
                f"""SELECT p.name, COUNT(*) FROM purchases p JOIN visits v ON v.id = p.visit
                    WHERE v.profile = ? AND {_SOLD} AND lower(v.currency) = ? AND {_KEPT} GROUP BY p.name ORDER BY COUNT(*) DESC, MAX(p.id) DESC LIMIT 3""",
                (person.id, cur),
            ).fetchall()
            out.append({
                "id": person.id, "name": person.name, "email": person.email, "has_face": person.has_thumb,
                "method": self.book.method_name(person.brand, person.last4, person.payer) if self.book.can_pay(person) else "",
                "inside": self.book.open_visit(person) is not None, "since": person.created,
                "transactions": count, "spent": round(spent - back, 2), "last": last,
                "buys": [{"name": r[0], "times": r[1]} for r in buys],
            })
        return sorted(out, key=lambda c: (-c["spent"], -(c["last"] or 0), c["name"]))

"""
The ledger: every walk-out is written down with the steps, the pictures and the numbers behind its bill, a rehearsal
is kept apart from real sales, money is given back at most once, and two runs watching one shopper charge once.
"""
import asyncio

import numpy as np

from checkout.ledger import Ledger
from checkout.pipeline import Services, Session
from tests.test_crowd import CATALOG, DT, FakeMailer, FakePayments, Shop, play
from tests.test_last_check import FakeVerifier, shop_with

JPEG = b"\xff\xd8"


def test_a_walk_out_is_written_down_with_its_steps_pictures_and_numbers(tmp_path):
    async def scenario():
        shop, ids, profiles = shop_with(tmp_path, FakeVerifier({2: "put_back"}))
        shop.session.pick_shots[shop.cart(ids[0])["lines"][0]["id"]] = (6.0, JPEG + b"pick")
        shop.session.wall0 = 1_800_000_000.0
        await shop.session.finish()
        ledger = Ledger(shop.book)

        listed = ledger.transactions()
        assert len(listed) == 3 and {t["shopper"]["name"] for t in listed} == {"Shopper 0", "Shopper 2", "Shopper 4"}
        paid = ledger.transaction(next(t["id"] for t in listed if t["shopper"]["id"] == profiles[0].id))
        assert (paid["status"], paid["amount"], paid["rehearsal"], paid["provider"]) == ("succeeded", 3.0, False, "paypal")
        assert [(l["name"], l["price"], l["charged"], l["note"]) for l in paid["lines"]] == [("Can", 3.0, True, "picture A: Can")]
        steps = [e["type"] for e in paid["detail"]["events"]]
        assert steps[0] == "enter" and "pick" in steps and steps[-4:] == ["exit", "audit", "payment", "email"]  # the whole visit, in order
        assert paid["detail"]["events"][-4]["how"] == "ended"  # the footage stopped with them inside
        assert paid["detail"]["payment"]["email"]["status"] == "sent"
        assert paid["detail"]["t0"] == 1_800_000_000.0 and paid["detail"]["payment"]["check"]["kept"] == 1
        numbers = paid["detail"]["numbers"]
        assert numbers["leave_after_s"] == 3.0 and numbers["left_how"] == "ended" and numbers["in_view_s"] > 5
        # the pictures: the item leaving the shelf, then the three the check before charging was shown
        assert [p["kind"] for p in paid["pictures"]] == ["pick", "check", "check", "check"]
        assert paid["pictures"][0]["caption"] == "Can, taken from the shelf"
        assert paid["pictures"][1]["caption"] == "Final check, picture A: Can in hand"
        assert paid["pictures"][3]["caption"].endswith("last full view")
        assert ledger.proof(paid["id"], 0) == JPEG + b"pick" and ledger.proof(paid["id"], 1)[:2] == JPEG and ledger.proof(paid["id"], 9) is None

        # what the check refused is on record as not charged, with the reason; it is no sale
        held = ledger.transaction(next(t["id"] for t in listed if t["shopper"]["id"] == profiles[2].id))
        assert (held["status"], held["amount"]) == ("nothing_due", 0.0)
        assert [(l["name"], l["charged"]) for l in held["lines"]] == [("Can", False)] and "Returned to the shelf" in held["lines"][0]["note"]
        # a held-back line waits for someone at the shop to look at it
        assert held["lines"][0]["review"] and held["review"] == 1 and next(t for t in listed if t["id"] == held["id"])["review"] == 1 and ledger.summary("USD")["review"] == 1
        assert not paid["lines"][0]["review"] and next(t for t in listed if t["id"] == paid["id"])["review"] == 0
        assert ledger.clear(paid["id"], paid["lines"][0]["id"]) == (False, "That line has been charged.")
        assert ledger.clear(held["id"], held["lines"][0]["id"]) == (True, "")
        assert not ledger.transaction(held["id"])["lines"][0]["review"] and ledger.summary("USD")["review"] == 0
        assert ledger.chargeable(held["id"], held["lines"][0]["id"]) == (False, "That line was marked as not taken.")

        total = ledger.summary("USD")
        assert (total["sales"], total["transactions"], total["items"], total["basket"]) == (6.0, 2, 2, 3.0)
        assert total["top"] == [{"name": "Can", "sold": 2, "sales": 6.0}] and total["by_day"][-1] == {"day": total["by_day"][-1]["day"], "sales": 6.0, "transactions": 2}
        people = {c["name"]: c for c in ledger.customers("USD")}
        assert people["Shopper 0"]["spent"] == 3.0 and people["Shopper 0"]["buys"] == [{"name": "Can", "times": 1}]
        assert people["Shopper 2"]["spent"] == 0.0 and people["Shopper 2"]["buys"] == []

    asyncio.run(scenario())


def test_a_rehearsal_is_kept_but_is_never_a_sale(tmp_path):
    async def scenario():
        shop, ids, profiles = shop_with(tmp_path, FakeVerifier(), dry=True)
        await shop.session.finish()
        ledger = Ledger(shop.book)
        assert ledger.transactions() == [] and len(ledger.transactions(rehearsals=True)) == 3
        kept = ledger.transactions(rehearsals=True)[0]
        assert kept["rehearsal"] and kept["status"] == "dry" and kept["amount"] == 3.0 and kept["pictures"] == 3
        total = ledger.summary("USD")
        assert (total["sales"], total["transactions"], total["items"], total["top"]) == (0.0, 0, 0, [])
        assert ledger.refundable(kept["id"]) == (0.0, "A rehearsal charged nothing, so there is nothing to refund.")

    asyncio.run(scenario())


def test_money_goes_back_at_most_once(tmp_path):
    async def scenario():
        shop, ids, profiles = shop_with(tmp_path, FakeVerifier())
        await shop.session.finish()
        ledger = Ledger(shop.book)
        visit = next(t["id"] for t in ledger.transactions() if t["shopper"]["id"] == profiles[0].id)
        line = ledger.transaction(visit)["lines"][0]["id"]
        assert ledger.refundable(visit, line) == (3.0, "") and ledger.refundable(visit) == (3.0, "")
        assert ledger.refundable(visit, 99999)[0] == 0.0 and ledger.refundable(99999)[0] == 0.0

        ledger.add_refund(visit, line, 3.0, "usd", {"status": "failed", "id": None, "message": "PayPal was down"})
        assert ledger.refundable(visit, line) == (3.0, "")  # a refund that did not go through can be tried again
        ledger.add_refund(visit, line, 3.0, "usd", {"status": "refunded", "id": "REF1", "message": ""}, "wrong book")
        assert ledger.refundable(visit, line) == (0.0, "This transaction has been refunded in full.")
        seen = ledger.transaction(visit)
        assert seen["refunded"] == 3.0 and seen["refundable"] == 0.0 and seen["lines"][0]["refunded"]
        assert [r["status"] for r in seen["refunds"]] == ["failed", "refunded"]

        total = ledger.summary("USD")
        assert (total["sales"], total["refunded"], total["items"]) == (9.0, 3.0, 2)  # a refunded line is not an item sold
        assert {c["name"]: c["spent"] for c in ledger.customers("USD")}["Shopper 0"] == 0.0

    asyncio.run(scenario())


def test_a_sale_through_another_provider_is_not_refunded_here(tmp_path):
    shop = Shop(tmp_path)
    person = shop.book.add("Old", None, "old@example.com")
    visit = shop.book.record_visit(person, "s0", {"amount": 18.5, "currency": "aed", "status": "succeeded", "id": "pi_123"}, [("meal", "Frozen meal", 18.5)])
    shop.book.db.execute("UPDATE visits SET provider = 'stripe' WHERE id = ?", (visit,))
    left, why = Ledger(shop.book).refundable(visit)
    assert left == 0.0 and "Stripe" in why
    total = Ledger(shop.book).summary("USD")
    assert total["sales"] == 0.0 and total["other"] == [{"currency": "aed", "transactions": 1, "sales": 18.5}]


def test_two_runs_watching_one_shopper_charge_once(tmp_path):
    async def scenario():
        first = Shop(tmp_path)
        second = Shop(tmp_path, book=first.book)
        second.pay = first.pay
        second.session.services.payments = first.pay
        person = first.book.add("Twice Seen", None, "twice@example.com")
        first.book.set_card(person, "cust_t", "tok_t", "visa", "1881")
        first.book.admit(person)
        for shop in (first, second):  # the page and a replay of the same clip, both linked before either charges
            shop.session.who[play(shop)[0]] = person
        await asyncio.gather(first.session.finish(), second.session.finish())

        assert [c["customer"] for c in first.pay.charges] == ["cust_t"]
        ends = sorted(s.session.payments[s.ids()[0] if s.ids() else 1][-1]["status"] for s in (first, second))
        assert ends == ["settled", "succeeded"]
        assert len(Ledger(first.book).transactions()) == 1  # and one transaction, not two

    asyncio.run(scenario())


def test_a_shopper_recognised_again_keeps_one_bag_and_pays_once(tmp_path):
    async def scenario():
        shop = Shop(tmp_path)
        ids = play(shop)  # shopper 0 holds a can
        old = ids[0]
        person = shop.book.add("Stepped Out", None, "out@example.com")
        shop.book.set_card(person, "cust_s", "tok_s", "visa", "1881")
        shop.book.admit(person)
        shop.session.who[old] = person

        shop.present[0], shop.item_seen[0] = False, False  # out of the picture, can and all, for longer than the cutoff
        shop.run(4.0)
        assert shop.cart(old)["state"] == "left"
        await shop.session._pay(old)
        assert [c["amount"] for c in shop.pay.charges] == [3.0]

        shop.present[0], shop.item_seen[0] = True, True  # back, still holding it, under a new outline
        shop.run(2.5)
        new = next(p["id"] for p in shop.engine.snapshot()["people"] if p["id"] not in ids)
        in_hand = next(i for i in shop.engine.items.values() if i.visible and i.state == "candidate")
        shop.engine._pick(in_hand, new, "hand", False)  # first seen already in their hand, as a real camera would have it
        assert shop.cart(new)["count"] == 1  # the engine takes them for a stranger carrying a can
        state = shop.engine.snapshot()
        assert shop.session._same_shopper(new, old, state)

        assert new not in {c["person"] for c in shop.engine.carts()} and old in {p["id"] for p in state["people"]}
        bag = shop.cart(old)
        assert bag["state"] == "shopping" and (bag["count"], bag["due"]) == (1, 0.0)  # the can they already paid for, once
        assert [e["type"] for e in shop.engine.events if e.get("was") == new] == ["same"]
        assert not any(e.get("person") == new for e in shop.engine.events)

        await shop.session.finish()  # walking out again costs nothing more
        assert [c["amount"] for c in shop.pay.charges] == [3.0]
        assert shop.session.payments[old][-1]["status"] == "nothing_due"

    asyncio.run(scenario())


def test_a_shopper_in_view_is_never_folded_into_another(tmp_path):
    shop = Shop(tmp_path)
    ids = play(shop)
    assert not shop.engine.same_person(ids[1], ids[0])  # both are standing there: two people
    assert len(shop.engine.carts()) == 5


def test_a_held_back_line_can_be_charged_by_hand_once(tmp_path):
    """Someone left with an item nothing could confirm. A person at the shop looks at the pictures and charges it:
    a transaction of its own, pointing back at the walk-out, and never a second time."""

    async def scenario():
        shop, ids, profiles = shop_with(tmp_path, FakeVerifier({2: "put_back"}))
        await shop.session.finish()
        ledger = Ledger(shop.book)
        walk_out = next(t["id"] for t in ledger.transactions() if t["shopper"]["id"] == profiles[2].id)
        held = ledger.transaction(walk_out)["lines"][0]
        assert not held["charged"] and held["later"] is None and ledger.chargeable(walk_out, held["id"]) == (True, "")
        can = CATALOG.by_id["can"]

        nothing = await ledger.charge_later(walk_out, held["id"], None, "USD", "Shop", shop.pay, shop.mail)
        assert nothing == {"status": "refused", "visit": None, "message": "Choose which item it was."} and len(shop.pay.charges) == 2

        done = await ledger.charge_later(walk_out, held["id"], can, "USD", "Shop", shop.pay, shop.mail)
        assert done["status"] == "succeeded" and shop.pay.charges[-1]["customer"] == "cust_2" and shop.pay.charges[-1]["amount"] == 3.0
        assert shop.pay.charges[-1]["key"] == f"later-{walk_out}-{held['id']}-can"
        assert shop.mail.sent[-1] == {"to": "s2@example.com", "lines": [("Can", 3.0, "")], "amount": 3.0}  # they get an invoice
        assert ledger.transaction(walk_out)["lines"][0]["later"] == done["visit"]
        made = ledger.transaction(done["visit"])
        assert (made["status"], made["amount"], made["shopper"]["id"]) == ("succeeded", 3.0, profiles[2].id)
        assert made["detail"]["by_hand"] == {"of": walk_out, "line": held["id"]}
        assert [e["type"] for e in made["detail"]["events"]] == ["by_hand", "payment", "email"]
        assert "transaction %d" % walk_out in made["lines"][0]["note"] and ledger.refundable(done["visit"]) == (3.0, "")

        again = await ledger.charge_later(walk_out, held["id"], can, "USD", "Shop", shop.pay, shop.mail)
        assert again["status"] == "refused" and "already" in again["message"] and len(shop.pay.charges) == 3
        # a line that was charged at the door cannot be charged again this way
        paid = next(t["id"] for t in ledger.transactions() if t["shopper"]["id"] == profiles[0].id and t["status"] == "succeeded")
        assert ledger.chargeable(paid, ledger.transaction(paid)["lines"][0]["id"]) == (False, "That line was charged when they walked out.")
        assert ledger.summary("USD")["sales"] == 9.0 and ledger.summary("USD")["transactions"] == 3

    asyncio.run(scenario())


def test_nothing_from_a_rehearsal_is_charged_by_hand(tmp_path):
    async def scenario():
        shop, ids, profiles = shop_with(tmp_path, FakeVerifier({2: "unclear"}), dry=True)
        await shop.session.finish()
        ledger = Ledger(shop.book)
        rehearsal = next(t for t in ledger.transactions(rehearsals=True) if t["held"])
        line = ledger.transaction(rehearsal["id"])["lines"][0]["id"]
        result = await ledger.charge_later(rehearsal["id"], line, CATALOG.by_id["can"], "USD", "Shop", shop.pay, shop.mail)
        assert result["status"] == "refused" and "rehearsal" in result["message"] and not shop.pay.charges

    asyncio.run(scenario())

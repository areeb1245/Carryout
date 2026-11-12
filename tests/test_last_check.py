"""
The check before charging: one call to the second model per walk-out, line by line. Only what it answers "kept" for
is charged, and "unclear" for a title already read in their hands on the floor; "put back", and "unclear" for anything
else, stay on the bill as not charged, with the reason, for someone at the shop to review.
"""
import asyncio

import numpy as np

from tests.test_crowd import Shop, play

PICTURE = np.zeros((40, 40, 3), np.uint8)


class FakeVerifier:
    """Answers the last check with the verdict a test gives each shopper (0, 2 or 4); "kept" for the others."""

    enabled = True

    def __init__(self, verdicts=None, answers=True):
        self.verdicts, self.answers, self.asked = verdicts or {}, answers, []

    async def audit(self, lines, pictures):
        self.asked.append((lines, len(pictures)))
        if not self.answers:
            return None
        shopper = round(pictures[0][0] - 5.0)  # shop_with() stamps each shopper's first picture with 5 + their place
        return {line["line"]: {"verdict": self.verdicts.get(shopper, "kept"), "why": f"picture A: {line['name']}"} for line in lines}


def shop_with(tmp_path, verifier, dry=False, pictures=True):
    shop = Shop(tmp_path, verifier=verifier, dry=dry)
    ids = play(shop)  # shoppers 0, 2 and 4 each hold a can (3.00)
    profiles = {}
    for i in (0, 2, 4):
        profile = shop.book.add(f"Shopper {i}", None, f"s{i}@example.com")
        shop.book.set_card(profile, f"cust_{i}", f"tok_{i}", "visa", "1881")
        shop.book.admit(profile)
        shop.session.who[ids[i]] = profile
        profiles[i] = profile
        if pictures:
            shop.session.evidence[ids[i]] = {"read": {"can": (5.0 + i, PICTURE)}, "recent": [(18.0, PICTURE), (19.0, PICTURE)], "scene": (19.0, PICTURE)}
    return shop, ids, profiles


def line(shop, pid):
    return shop.cart(pid)["lines"][0]


def test_only_lines_the_check_confirms_are_charged(tmp_path):
    async def scenario():
        verifier = FakeVerifier({2: "put_back"})
        shop, ids, profiles = shop_with(tmp_path, verifier)
        await shop.session.finish()

        assert sorted(c["customer"] for c in shop.pay.charges) == ["cust_0", "cust_4"]  # shopper 2 was not charged
        assert line(shop, ids[0])["audit"] == "kept" and line(shop, ids[0])["paid"]
        held = line(shop, ids[2])
        assert (held["audit"], held["charge"], held["paid"], held["state"]) == ("put_back", False, None, "in_cart")
        assert held["audit_why"] == "picture A: Can"
        pay = shop.session.payments
        assert pay[ids[0]][-1]["check"] == {"status": "checked", "kept": 1, "held": 0, "pictures": 3, "session": shop.session.id, "person": ids[0]}
        assert len(shop.session.shown[ids[0]]) == 3 and shop.session.shown[ids[0]][0][:2] == b"\xff\xd8"  # kept as JPEGs for the page
        assert pay[ids[2]][-1]["status"] == "nothing_due" and pay[ids[2]][-1]["check"]["held"] == 1
        # one call a walk-out, and only for people with something to pay: shoppers 1 and 3 hold nothing
        # each is shown three pictures: the one their item was read in, their hands last, the whole scene last
        assert len(verifier.asked) == 3 and all(n == 3 for _, n in verifier.asked)
        assert verifier.asked[0][0] == [{"line": 1, "name": "Can", "looks_like": "drink can"}]
        verdicts = {e["person"]: e["verdict"] for e in shop.engine.events if e["type"] == "audit"}
        assert verdicts == {ids[0]: "kept", ids[2]: "put_back", ids[4]: "kept"}

    asyncio.run(scenario())


def test_unclear_is_never_charged(tmp_path):
    async def scenario():
        shop, ids, _ = shop_with(tmp_path, FakeVerifier({0: "unclear", 4: "unclear"}))
        await shop.session.finish()
        assert [c["customer"] for c in shop.pay.charges] == ["cust_2"]
        assert line(shop, ids[0])["audit"] == "unclear" and shop.cart(ids[0])["total"] == 0.0 and shop.cart(ids[0])["held_back"] == 3.0
        await shop.session.finish()  # asking again does not bring a held line back onto a charge
        assert len(shop.pay.charges) == 1

    asyncio.run(scenario())


def test_no_answer_from_the_check_means_no_charge(tmp_path):
    async def scenario():
        verifier = FakeVerifier(answers=False)
        shop, ids, _ = shop_with(tmp_path, verifier)
        await shop.session.finish()
        assert not shop.pay.charges and not shop.mail.sent
        assert len(verifier.asked) == 6  # each walk-out is tried twice before giving up
        assert shop.session.payments[ids[0]][-1]["check"]["status"] == "failed"
        assert "no answer" in line(shop, ids[0])["audit_why"]

    asyncio.run(scenario())


def test_without_a_picture_of_them_carrying_it_nothing_is_charged(tmp_path):
    async def scenario():
        verifier = FakeVerifier()
        shop, ids, _ = shop_with(tmp_path, verifier, pictures=False)
        await shop.session.finish()
        assert not shop.pay.charges and not verifier.asked
        assert line(shop, ids[2])["audit"] == "unclear" and "No picture" in line(shop, ids[2])["audit_why"]

    asyncio.run(scenario())


def test_someone_not_registered_is_not_checked_or_charged(tmp_path):
    async def scenario():
        verifier = FakeVerifier()
        shop, ids, _ = shop_with(tmp_path, verifier)
        del shop.session.who[ids[4]]
        await shop.session.finish()
        assert sorted(c["customer"] for c in shop.pay.charges) == ["cust_0", "cust_2"] and len(verifier.asked) == 2
        assert shop.session.payments[ids[4]][-1]["status"] == "not_registered" and line(shop, ids[4])["audit"] is None

    asyncio.run(scenario())


def test_a_rehearsal_runs_the_check_but_charges_and_emails_nobody(tmp_path):
    async def scenario():
        verifier = FakeVerifier()
        shop, ids, profiles = shop_with(tmp_path, verifier, dry=True)
        await shop.session.finish()
        assert not shop.pay.charges and not shop.mail.sent and len(verifier.asked) == 3
        last = shop.session.payments[ids[2]][-1]
        assert (last["status"], last["amount"], last["check"]["kept"]) == ("dry", 3.0, 1)
        assert line(shop, ids[2])["audit"] == "kept" and not line(shop, ids[2])["paid"]
        assert shop.book.open_visit(profiles[2]) is not None and shop.book.summary(profiles[2])["visits"] == 0  # it does not count as a visit
        # nor for the two who had nothing to pay: a rehearsal leaves every visit open for the real run
        assert shop.book.db.execute("SELECT COUNT(*) FROM admissions WHERE closed IS NOT NULL").fetchone()[0] == 0
        # the ledger keeps the three rehearsed bills, flagged, so their steps can be looked at; nothing real is written
        assert shop.book.db.execute("SELECT COUNT(*), SUM(dry) FROM visits").fetchone() == (3, 3)

    asyncio.run(scenario())


def test_the_invoice_link_reaches_the_visit_and_the_email(tmp_path):
    async def scenario():
        shop, ids, profiles = shop_with(tmp_path, FakeVerifier())
        made = []

        async def invoice(name, email, lines, currency, payment_id, shop_name, key):
            made.append((name, email, lines, currency, payment_id))
            return {"status": "created", "id": "INV2-1", "number": "0042", "url": "https://www.sandbox.paypal.com/invoice/p/#1", "message": ""}

        seen = []

        async def receipt(to, name, lines, payment, key):
            seen.append((to, payment["receipt_url"], payment["invoice"]["number"], payment["method"]))
            return {"status": "sent", "to": to, "id": "em", "message": ""}

        shop.pay.invoice, shop.mail.receipt = invoice, receipt
        await shop.session.finish()
        assert sorted(m[:4] for m in made) == [(f"Shopper {i}", f"s{i}@example.com", [("Can", 3.0, "")], "USD") for i in (0, 2, 4)]
        assert sorted(m[4] for m in made) == ["pi_1", "pi_2", "pi_3"]  # each invoice names the payment it is for
        assert ("s2@example.com", "https://www.sandbox.paypal.com/invoice/p/#1", "0042", "Visa ending 1881") in seen
        url = shop.book.db.execute("SELECT receipt_url FROM visits WHERE profile = ?", (profiles[2].id,)).fetchone()[0]
        assert url == "https://www.sandbox.paypal.com/invoice/p/#1"
        assert [e["status"] for e in shop.engine.events if e["type"] == "invoice"] == ["created"] * 3

    asyncio.run(scenario())


# ---------------------------------------------------------------- how the model's observations become a verdict


def test_the_verdict_comes_from_what_the_model_says_it_saw():
    import json

    import httpx

    from checkout.verify import Verifier

    replies = []

    def model(request: httpx.Request) -> httpx.Response:
        sent = json.loads(request.content)
        assert len([part for part in sent["messages"][1]["content"] if part["type"] == "image_url"]) == 3  # one call, every picture
        return httpx.Response(200, json={"choices": [{"message": {"content": replies.pop(0)}}]})

    async def scenario():
        verifier = Verifier({"LLM_URL": "http://model.test", "LLM_MODAL_KEY": "k", "LLM_MODAL_SECRET": "s"})
        await verifier.http.aclose()
        verifier.http = httpx.AsyncClient(transport=httpx.MockTransport(model))
        lines = [{"line": n, "name": name, "looks_like": "book"} for n, name in enumerate(["Kept", "Back", "Never", "Order", "Skipped"], 1)]
        pictures = [(float(t), PICTURE) for t in (5, 9, 19)]
        replies.append(json.dumps({"lines": [
            {"line": 1, "held_in": "A", "read": "KEPT", "put_down_in": None, "why": "in the stack"},
            {"line": 2, "held_in": "A", "read": "BACK", "put_down_in": "C", "why": "on the shelf again"},
            {"line": 3, "held_in": None, "read": "", "put_down_in": None, "why": "not in any picture"},
            {"line": 4, "held_in": "B", "read": "ORDER", "put_down_in": "A", "why": "on the shelf before they took it"},
            {"line": 9, "held_in": "A", "read": "", "put_down_in": None, "why": "a line nobody asked about"},
            {"line": 1, "held_in": None, "read": "", "put_down_in": None, "why": "a second answer for line 1"},
        ]}))
        out = await verifier.audit(lines, pictures)
        assert out[1] == {"verdict": "kept", "why": 'In hand in picture A: "KEPT"'}
        assert out[2] == {"verdict": "put_back", "why": "In hand in picture A, back on the shelf in picture C"}
        assert out[3] == {"verdict": "unclear", "why": "not in any picture"}
        assert out[4]["verdict"] == "kept"  # on the shelf before it was taken is not a put-back
        assert out[5]["verdict"] == "unclear" and set(out) == {1, 2, 3, 4, 5}  # a line the model skipped is not confirmed
        replies.append(json.dumps({"lines": [{"line": 1, "held_in": "Z", "read": "KEPT", "put_down_in": None}]}))
        assert (await verifier.audit(lines[:1], pictures))[1]["verdict"] == "unclear"  # a picture it was never shown
        replies.append("not json at all")
        assert await verifier.audit(lines, pictures) is None
        assert await verifier.audit(lines, []) is None and await verifier.audit([], pictures) is None  # nothing to ask
        await verifier.close()

    asyncio.run(scenario())


def test_unclear_is_charged_when_the_title_was_read_in_their_hands(tmp_path, monkeypatch):
    """Two books carried in a stack: the one underneath was read in their hands on the floor, but the final check
    cannot make its title out again. The readings stand and it is charged; the reason says so."""
    async def scenario():
        shop, ids, _ = shop_with(tmp_path, FakeVerifier({0: "unclear", 2: "unclear", 4: "put_back"}))
        # the can stands in for a titled product here: one whose confirmation took two readings of its title
        monkeypatch.setattr(shop.session.catalog, "family_of", lambda product: "any:can" if product == "can" else None)
        for i in (0, 4):
            line_ = next(e for e in shop.engine.entries if e.person == ids[i])
            line_.verified, line_.why = "confirmed", "Read in hand twice: CAN"
        await shop.session.finish()
        assert sorted(c["customer"] for c in shop.pay.charges) == ["cust_0"]
        kept = line(shop, ids[0])
        assert (kept["audit"], kept["charge"], bool(kept["paid"])) == ("kept", True, True) and "title not visible" in kept["audit_why"]
        # never read in their hands: still held for review
        assert line(shop, ids[2])["audit"] == "unclear" and not line(shop, ids[2])["paid"]
        # seen put back: never charged, whatever was read before
        assert line(shop, ids[4])["audit"] == "put_back" and not line(shop, ids[4])["paid"]

    asyncio.run(scenario())


def test_no_answer_from_the_check_holds_even_a_confirmed_title(tmp_path):
    async def scenario():
        shop, ids, _ = shop_with(tmp_path, FakeVerifier(answers=False))
        for e in shop.engine.entries:
            e.verified = "confirmed"
        await shop.session.finish()
        assert not shop.pay.charges and line(shop, ids[0])["audit"] == "unclear"

    asyncio.run(scenario())


def test_unclear_holds_a_product_without_a_title_even_when_confirmed(tmp_path):
    """A can is confirmed on a single look, which is not enough to overrule the final check."""
    async def scenario():
        shop, ids, _ = shop_with(tmp_path, FakeVerifier({0: "unclear"}))
        next(e for e in shop.engine.entries if e.person == ids[0]).verified = "confirmed"
        await shop.session.finish()
        assert "cust_0" not in [c["customer"] for c in shop.pay.charges] and line(shop, ids[0])["audit"] == "unclear"

    asyncio.run(scenario())


def test_a_title_first_read_at_the_door_does_not_overrule_the_final_check(tmp_path, monkeypatch):
    """A pick in their last second, named only from the door pictures: the final check saw those same pictures and
    could not confirm it, so it is held for review, not charged."""
    async def scenario():
        shop, ids, _ = shop_with(tmp_path, FakeVerifier({0: "unclear"}))
        monkeypatch.setattr(shop.session.catalog, "family_of", lambda product: "any:can" if product == "can" else None)

        async def door_look(pid):
            for e in shop.engine.entries:
                if e.person == pid:
                    e.verified, e.why = "confirmed", "Read in hand at exit: CAN"

        monkeypatch.setattr(shop.session, "_last_looks", door_look)
        await shop.session.finish()
        assert line(shop, ids[0])["verified"] == "confirmed"  # the door look did name it
        assert "cust_0" not in [c["customer"] for c in shop.pay.charges] and line(shop, ids[0])["audit"] == "unclear"
        assert "cust_2" in [c["customer"] for c in shop.pay.charges]  # kept by the final check itself

    asyncio.run(scenario())


def test_one_item_in_hand_never_pays_for_two_lines():
    import json

    import httpx

    from checkout.verify import Verifier

    replies = []

    def model(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": replies.pop(0)}}]})

    async def scenario():
        verifier = Verifier({"LLM_URL": "http://model.test", "LLM_MODAL_KEY": "k", "LLM_MODAL_SECRET": "s"})
        await verifier.http.aclose()
        verifier.http = httpx.AsyncClient(transport=httpx.MockTransport(model))
        bank = {"name": "Power bank", "looks_like": "grey slab"}
        lines = [{"line": 1, **bank}, {"line": 2, "name": "Tissue pack", "looks_like": "blue pack"}, {"line": 3, **bank}]
        pictures = [(1.0, PICTURE)]

        def said(count):
            row = lambda n: {"line": n, "held_in": "A", "read": "", "put_down_in": None, "why": "in hand", **({"count": count} if count is not None else {})}
            return json.dumps({"lines": [row(1), {**row(2), "count": 1}, row(3)]})

        replies.append(said(1))  # one power bank in their hand, two on the bill
        out = await verifier.audit(lines, pictures)
        assert [out[n]["verdict"] for n in (1, 2, 3)] == ["kept", "kept", "unclear"]
        assert out[3]["why"] == "Only 1 in hand at once, 2 on the bill"
        replies.append(said(2))  # two of them, really
        assert [v["verdict"] for v in (await verifier.audit(lines, pictures)).values()] == ["kept", "kept", "kept"]
        replies.append(said(None))  # no count given: the one seen held is billed, the second is not
        out = await verifier.audit(lines, pictures)
        assert [out[n]["verdict"] for n in (1, 2, 3)] == ["kept", "kept", "unclear"]
        await verifier.close()

    asyncio.run(scenario())


def test_the_check_in_steps_takes_off_what_went_back_and_keeps_doubt_for_what_left_unread():
    """The check before charging as separate questions about separate pictures. A book read in hand and then seen
    going onto the shelf comes off; one in hand earlier whose title cannot be made out as they leave is unclear (not
    absent), so a title confirmed on the floor still stands; a book read in their hands as they leave is kept."""
    import base64
    import json

    import cv2
    import httpx

    from checkout.verify import Verifier

    def picture(n):  # each picture is told apart by its height
        return np.zeros((20 + n, 30, 3), np.uint8)

    views = [(26.9 + 0.66 * n, picture(n)) for n in range(6)]  # pictures 0 to 5, the last of their hands
    asked = []

    def model(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        system, content = body["messages"][0]["content"], body["messages"][1]["content"]
        raw = base64.b64decode(content[1]["image_url"]["url"].split(",", 1)[1])
        n = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR).shape[0] - 20
        text = content[0]["text"]
        asked.append((system[:40], n))
        if "what is in the shopper's hands" in system:
            answer = {"hands": "visible", "items": [{"product": "deep-work", "read": "DEEP WORK"}, {"product": "unknown"}] if n == 4 else [{"product": "unknown"}, {"product": "unknown"}], "why": "test"}
        elif "Question: is the shopper holding" in system:
            answer = {"answer": "yes" if n >= 10 else "unknown", "read": "", "why": "test"}
        else:  # the shelf: picture 2 shows Shoe Dog going back
            answer = {"placing": "yes" if n == 2 else "no", "product": "shoe-dog" if n == 2 else "unknown", "read": "SHOE DOG" if n == 2 else "", "why": "test"}
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(answer)}}]})

    async def scenario(budget=12):
        verifier = Verifier({"LLM_URL": "http://model.test", "LLM_MODAL_KEY": "k", "LLM_MODAL_SECRET": "s"})
        await verifier.http.aclose()
        verifier.http = httpx.AsyncClient(transport=httpx.MockTransport(model))
        lines = [
            {"line": 1, "product": "shoe-dog", "name": "Shoe Dog", "looks_like": "black spine", "door": "put_back", "door_why": "Put down on the way out", "last_read_t": 27.5},
            {"line": 2, "product": "think-straight", "name": "Think Straight", "looks_like": "orange spine", "door": None, "door_why": "", "last_read_t": 25.4},
            {"line": 3, "product": "deep-work", "name": "Deep Work", "looks_like": "yellow spine", "door": None, "door_why": "", "last_read_t": 28.9},
        ]
        evidence = {"held": {2: (25.4, picture(12)), 3: (28.9, picture(13))}, "pick": {}, "views": views, "shelf": [], "scene": None}
        catalogue = [{"id": i, "name": n} for i, n in (("shoe-dog", "Shoe Dog"), ("think-straight", "Think Straight"), ("deep-work", "Deep Work"))]
        out = await verifier.check_visit(lines, evidence, catalogue, budget=budget)
        await verifier.close()
        return out

    out = asyncio.run(scenario())
    verdicts = out["verdicts"]
    assert verdicts[1]["verdict"] == "put_back" and "put on the shelf" in verdicts[1]["why"] and verdicts[1]["absent"]
    assert verdicts[2]["verdict"] == "unclear" and not verdicts[2]["absent"]  # in hand earlier, unreadable as they left
    assert verdicts[3]["verdict"] == "kept" and "as they left" in verdicts[3]["why"]
    assert out["calls"] <= 12 and len(out["steps"]) == out["calls"] == len(out["pictures"]) == len(asked)
    assert [s["ask"] for s in out["steps"]][:2] == ["leaving", "leaving"]
    asked.clear()
    few = asyncio.run(scenario(budget=2))  # out of questions: the rest is unclear, never guessed
    assert few["calls"] == 2 and len(asked) == 2 and all(v["verdict"] != "kept" for n, v in few["verdicts"].items() if n != 3)


def test_the_book_they_could_not_read_is_matched_by_its_look_among_the_titles_it_could_be():
    """The door counted one unreadable book in their hands, and two titles it could be: one on the bill (read last
    some time ago), one the floor saw go back after it was read (a track that slipped in a stack). Both of the last
    views say it looks like the second: that one goes back on the bill, the first comes off. Views that disagree
    decide nothing, and a title is put back on the bill only on both views."""
    import json

    import httpx

    from checkout.verify import Verifier

    def run(said, shelved=None):
        answers = list(said)

        def model(request: httpx.Request) -> httpx.Response:
            system = json.loads(request.content)["messages"][0]["content"]
            if "putting a book onto a shelf here" in system:
                answer = {"product": shelved, "why": "white book going back"}
            elif "Question: is a hand putting an item onto or into a shelf" in system:
                answer = {"placing": "yes" if shelved else "no", "product": "unknown", "why": "test"}
            elif "which of the candidate books" in system:
                answer = {"product": answers.pop(0), "why": "pale blue cover"}
            elif "what is in the shopper's hands" in system:
                answer = {"hands": "visible", "items": [{"product": "unknown"}], "why": "test"}
            elif "Question: is the shopper holding" in system:
                answer = {"answer": "unknown", "read": "", "why": "test"}
            else:
                answer = {"placing": "no", "product": "unknown", "why": "test"}
            return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(answer)}}]})

        async def scenario():
            verifier = Verifier({"LLM_URL": "http://model.test", "LLM_MODAL_KEY": "k", "LLM_MODAL_SECRET": "s"})
            await verifier.http.aclose()
            verifier.http = httpx.AsyncClient(transport=httpx.MockTransport(model))
            lines = [{"line": 1, "product": "deep-work", "name": "Deep Work", "looks_like": "white spine", "door": None, "door_why": "", "last_read_t": 27.6}]
            views = [(28.0 + n, PICTURE) for n in range(3)]
            doubt = {"spare": 1, "candidates": [{"product": "deep-work", "name": "Deep Work", "looks_like": "white spine", "line": 1, "sure": True},
                                                {"product": "ikigai", "name": "Ikigai", "looks_like": "pale blue, blossom", "line": None}]}
            catalogue = [{"id": "deep-work", "name": "Deep Work"}, {"id": "ikigai", "name": "Ikigai"}]
            out = await verifier.check_visit(lines, {"held": {1: (27.6, PICTURE)}, "pick": {}, "views": views, "shelf": [], "scene": None}, catalogue, doubt=doubt)
            await verifier.close()
            return out

        return asyncio.run(scenario())

    both = run(["ikigai", "ikigai"])
    assert [r["product"] for r in both["reinstate"]] == ["ikigai"]
    assert both["verdicts"][1]["verdict"] == "put_back" and both["verdicts"][1]["absent"] and "is Ikigai" in both["verdicts"][1]["why"]
    # one view is not enough to put a title back on the bill, nor to take one off; with nothing seen going onto a
    # shelf, the title on the bill is left to the floor's readings (not absent)
    one = run(["ikigai", "unknown"])
    assert one["reinstate"] == [] and one["verdicts"][1]["verdict"] == "unclear" and not one["verdicts"][1]["absent"]
    split = run(["ikigai", "deep-work"])  # the views disagree: neither is clearly the one
    assert split["reinstate"] == [] and split["verdicts"][1]["verdict"] == "unclear" and not split["verdicts"][1]["absent"]
    # a book seen going onto the shelf that cannot be told apart: the title on the bill is held, not charged
    went = run(["unknown", "unknown"], shelved="unknown")
    assert went["reinstate"] == [] and went["verdicts"][1]["verdict"] == "unclear" and went["verdicts"][1]["absent"]
    kept = run(["deep-work", "unknown"])  # it looks like the title on the bill: that one stays
    assert kept["reinstate"] == [] and kept["verdicts"][1]["verdict"] == "kept"
    # a white book seen going onto the shelf looks like Deep Work: it comes off, and the one book they still hold
    # unread can only be the other title, which goes back on the bill without asking which it looks like
    gone = run([], shelved="deep-work")
    assert gone["verdicts"][1]["verdict"] == "put_back" and "Seen put on the shelf" in gone["verdicts"][1]["why"]
    assert [r["product"] for r in gone["reinstate"]] == ["ikigai"] and "onto the shelf" in gone["reinstate"][0]["why"]
    assert "which" not in [s["ask"] for s in gone["steps"]] and "placed" in [s["ask"] for s in gone["steps"]]


def test_the_check_in_steps_keeps_a_product_without_a_title_held_and_not_seen_put_down():
    """A product told apart by its look, not a title (a tube of cream): seen in hand earlier and not seen going onto a
    shelf, it is kept even when it is not named in the pictures as they leave, as the one-look check did. One seen
    going back still comes off. A book the door found gone, with only another item seen going back, is left to the
    floor's readings."""
    import base64
    import json

    import cv2
    import httpx

    from checkout.verify import Verifier

    def picture(n):
        return np.zeros((20 + n, 30, 3), np.uint8)

    views = [(26.0 + 0.5 * n, picture(n)) for n in range(4)]

    def model(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        system, content = body["messages"][0]["content"], body["messages"][1]["content"]
        raw = base64.b64decode(content[1]["image_url"]["url"].split(",", 1)[1])
        n = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR).shape[0] - 20
        if "what is in the shopper's hands" in system:
            answer = {"hands": "hidden", "items": [], "why": "test"}
        elif "Question: is the shopper holding" in system:
            answer = {"answer": "yes" if n >= 10 else "unknown", "read": "", "why": "test"}
        else:  # picture 1 shows the earbuds going back
            answer = {"placing": "yes" if n == 1 else "no", "product": "earbuds" if n == 1 else "unknown", "read": "", "why": "test"}
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(answer)}}]})

    async def scenario():
        verifier = Verifier({"LLM_URL": "http://model.test", "LLM_MODAL_KEY": "k", "LLM_MODAL_SECRET": "s"})
        await verifier.http.aclose()
        verifier.http = httpx.AsyncClient(transport=httpx.MockTransport(model))
        lines = [
            {"line": 1, "product": "nivea", "name": "Nivea", "looks_like": "blue tube", "door": None, "door_why": "", "last_read_t": None, "titled": False},
            {"line": 2, "product": "earbuds", "name": "Earbuds", "looks_like": "white case", "door": None, "door_why": "", "last_read_t": None, "titled": False},
            # a book the door found gone (not read in its last pictures), with nothing but the earbuds seen going back
            {"line": 3, "product": "deep-work", "name": "Deep Work", "looks_like": "yellow spine", "door": "put_back", "door_why": "Put down on the way out", "last_read_t": 25.0, "titled": True},
        ]
        evidence = {"held": {1: (24.0, picture(12)), 2: (24.5, picture(13))}, "pick": {}, "views": views, "shelf": [], "scene": None}
        catalogue = [{"id": "nivea", "name": "Nivea"}, {"id": "earbuds", "name": "Earbuds"}, {"id": "deep-work", "name": "Deep Work"}]
        out = await verifier.check_visit(lines, evidence, catalogue)
        await verifier.close()
        return out

    verdicts = asyncio.run(scenario())["verdicts"]
    assert verdicts[1]["verdict"] == "kept" and "not seen put down" in verdicts[1]["why"]
    assert verdicts[2]["verdict"] == "put_back" and verdicts[2]["absent"]
    assert verdicts[3]["verdict"] == "unclear" and not verdicts[3]["absent"]  # a stack, not a put-back: the floor decides


def test_a_placing_named_as_the_book_still_in_hand_is_matched_among_the_others_or_not_believed():
    """The shelf question names the book in the other hand (read in hand as they leave). The placing is matched among
    the other titles by its look: one that fits comes off; when none fits, the placing is not believed, and a book the
    door found gone with nothing else seen going back is left to the floor's readings."""
    import json

    import httpx

    from checkout.verify import Verifier

    def run(placed):
        def model(request: httpx.Request) -> httpx.Response:
            system = json.loads(request.content)["messages"][0]["content"]
            if "putting a book onto a shelf here" in system:
                answer = {"product": placed, "why": "white book"}
            elif "Question: is a hand putting an item onto or into a shelf" in system:
                answer = {"placing": "yes", "product": "blue-ocean-strategy", "why": "test"}
            elif "what is in the shopper's hands" in system:
                answer = {"hands": "visible", "items": [{"product": "blue-ocean-strategy", "read": "BLUE OCEAN"}], "why": "test"}
            else:
                answer = {"answer": "unknown", "read": "", "why": "test"}
            return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(answer)}}]})

        async def scenario():
            verifier = Verifier({"LLM_URL": "http://model.test", "LLM_MODAL_KEY": "k", "LLM_MODAL_SECRET": "s"})
            await verifier.http.aclose()
            verifier.http = httpx.AsyncClient(transport=httpx.MockTransport(model))
            lines = [
                {"line": 1, "product": "blue-ocean-strategy", "name": "Blue Ocean Strategy", "looks_like": "blue", "door": None, "door_why": "", "last_read_t": 29.0, "titled": True},
                {"line": 2, "product": "the-psychology-of-money", "name": "The Psychology of Money", "looks_like": "white", "door": "put_back", "door_why": "Put down on the way out", "last_read_t": 27.6, "titled": True},
            ]
            views = [(28.0 + 0.5 * n, PICTURE) for n in range(4)]
            catalogue = [{"id": x["product"], "name": x["name"]} for x in lines]
            out = await verifier.check_visit(lines, {"held": {}, "pick": {}, "views": views, "shelf": [], "scene": None}, catalogue)
            await verifier.close()
            return out["verdicts"]

        return asyncio.run(scenario())

    matched = run("the-psychology-of-money")
    assert matched[2]["verdict"] == "put_back" and "Seen put on the shelf" in matched[2]["why"]
    assert matched[1]["verdict"] == "kept"
    unmatched = run("unknown")
    assert unmatched[2]["verdict"] == "unclear" and not unmatched[2]["absent"]

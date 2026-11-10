"""
The gate: a known face with a payment method is let in, a new face registers first, and only people the gate let in
are billed on the shop floor.

Faces here are made-up embeddings compared with numpy, and the payment provider is a stand-in, so no model file,
camera or network is needed.
"""
import time
import asyncio

import numpy as np

from checkout.gate import Gate
from checkout.identity import Face, Profiles
from checkout.payments import Card
from tests.test_crowd import Shop, play


class Book(Profiles):
    """Profiles whose faces are compared without the SFace model, and whose "camera" shows what a test puts in it."""

    def __init__(self, path, **kw):
        super().__init__(path, **kw)
        self.ready = True
        self.scene: list[tuple[Face, float, float]] = []  # (face, centre x, centre y) in view on the shop floor

    def score(self, profile, face):
        if not profile.templates:
            return 0.0
        return max(float(t.ravel() @ face.signature.ravel()) for t in profile.templates)

    def faces_in(self, frame):
        return [(np.array([i, 0, 80.0]), x, y) for i, (_, x, y) in enumerate(self.scene)]

    def read(self, frame, row):
        return self.scene[int(row[0])][0]


def face(seed: int, quality: float = 120.0) -> Face:
    """A face nobody else has: a random unit vector. Two seeds score about 0 against each other."""
    v = np.random.default_rng(seed).normal(size=(1, 128)).astype(np.float32)
    return Face(v / np.linalg.norm(v), quality, b"\xff\xd8jpeg")


class FakePay:
    enabled, problem, name = True, "", "paypal"

    def __init__(self):
        self.saved, self.fail = [], None

    async def save_card(self, name, test_card, shopper_id, email=None, details=None):
        if self.fail:
            raise self.fail
        self.saved.append(("card", name, test_card))
        return Card(f"cust_{shopper_id}", f"tok_{shopper_id}", "visa", "1881")

    async def begin_wallet(self, shopper_id, return_url, cancel_url, shop=""):
        self.saved.append(("begin", return_url))
        return {"id": "SETUP1", "approve_url": "https://www.sandbox.paypal.com/agreements/approve?approval_session_id=x"}

    async def finish_wallet(self, setup_token):
        if self.fail:
            raise self.fail
        return Card("cust_w", "tok_w", "paypal", "", "buyer@example.com")


def gate_at(tmp_path):
    book, pay = Book(tmp_path / "checkout.db"), FakePay()
    return Gate(book, pay, "Bookshop"), book, pay


def run(coro):
    return asyncio.run(coro)


def test_a_new_face_registers_with_a_card_and_is_let_in(tmp_path):
    async def scenario():
        gate, book, pay = gate_at(tmp_path)
        seen = await gate.see(face(1))
        assert seen["status"] == "new" and seen["face"].startswith("data:image/jpeg;base64,")
        assert (await gate.see(face(1)))["ticket"] == seen["ticket"]  # still the same person standing there
        assert (await gate.see(face(2)))["ticket"] != seen["ticket"]

        done = await gate.register(ticket=seen["ticket"], name="Ada Lovelace", email="ada@example.com", method="card", test_card="visa")
        assert done["status"] == "admitted" and done["new"] is True
        profile = next(iter(book.profiles.values()))
        assert (profile.name, profile.email, profile.brand, profile.last4, profile.provider) == ("Ada Lovelace", "ada@example.com", "visa", "1881", "paypal")
        assert len(profile.templates) == 1 and book.open_visit(profile) is not None
        assert done["profile"]["method"] == "Visa ending 1881" and done["profile"]["inside"] is True

        again = await gate.see(face(1))  # next time they are simply recognised
        assert again["status"] == "known" and again["ready"] is True and again["profile"]["name"] == "Ada Lovelace"

    run(scenario())


def test_a_known_face_with_a_method_on_file_is_let_in(tmp_path):
    async def scenario():
        gate, book, pay = gate_at(tmp_path)
        profile = book.add("Grace", face(3), "grace@example.com")
        book.set_card(profile, "cust", "tok", "mastercard", "9903")
        assert book.open_visit(profile) is None
        seen = await gate.see(face(3))
        assert seen["status"] == "known" and seen["ready"] is True
        assert gate.admit(seen["pass"])["status"] == "admitted"
        first = book.open_visit(profile)
        assert first is not None
        gate.admit(seen["pass"])  # walking through twice is still one visit
        assert book.db.execute("SELECT COUNT(*) FROM admissions").fetchone()[0] == 1
        book.close_visit(profile, "s1")
        assert book.open_visit(profile) is None
        assert gate.admit("not-a-pass")["status"] == "error"

    run(scenario())


def test_a_known_face_without_a_usable_method_must_add_one(tmp_path):
    async def scenario():
        gate, book, pay = gate_at(tmp_path)
        profile = book.add("Old Timer", face(4), "old@example.com")
        book.set_card(profile, "cus_1", "pm_1", "visa", "4242")
        profile.provider = "stripe"  # saved by the Stripe version: PayPal cannot charge it
        seen = await gate.see(face(4))
        assert seen["status"] == "known" and seen["ready"] is False and seen["profile"]["method"] == ""
        assert gate.admit(seen["pass"])["status"] == "needs_method" and book.open_visit(profile) is None
        done = await gate.register(token=seen["pass"], method="card", test_card="visa")
        assert done["status"] == "admitted" and done["new"] is False
        assert len(book.profiles) == 1 and profile.provider == "paypal" and book.open_visit(profile) is not None

    run(scenario())


def test_a_paypal_account_is_approved_before_anyone_is_let_in(tmp_path):
    async def scenario():
        gate, book, pay = gate_at(tmp_path)
        seen = await gate.see(face(5))
        begun = await gate.register(ticket=seen["ticket"], name="Lin", email="lin@example.com", method="paypal", return_base="http://localhost:8765")
        assert begun["status"] == "approve" and begun["approve_url"].startswith("https://www.sandbox.paypal.com/")
        assert pay.saved[-1] == ("begin", f"http://localhost:8765/gate/return?reg={begun['reg']}")
        assert gate.progress(begun["reg"]) == {"status": "waiting"} and not book.profiles  # nothing exists until PayPal says yes

        done = await gate.approved(begun["reg"])
        assert done["status"] == "admitted" and done["profile"]["method"] == "PayPal account buyer@example.com"
        assert gate.progress(begun["reg"])["status"] == "admitted"
        assert (await gate.approved(begun["reg"]))["status"] == "admitted" and len(book.profiles) == 1  # a reload of the return page changes nothing

        other = await gate.see(face(6))
        begun = await gate.register(ticket=other["ticket"], name="Sam", email="sam@example.com", method="paypal", return_base="http://localhost:8765")
        assert (await gate.approved(begun["reg"], cancelled=True))["status"] == "cancelled"
        assert len(book.profiles) == 1 and gate.progress("nope")["status"] == "error"

    run(scenario())


def test_registration_needs_a_name_an_email_and_a_method_that_saves(tmp_path):
    async def scenario():
        gate, book, pay = gate_at(tmp_path)
        assert (await gate.see(None))["status"] == "no_face"
        assert (await gate.see(face(7, quality=20)))["status"] == "closer"  # too far away to register on
        seen = await gate.see(face(7))
        assert (await gate.register(ticket=seen["ticket"], name="", email="a@example.com", method="card", test_card="visa"))["field"] == "name"
        assert (await gate.register(ticket=seen["ticket"], name="Zed", email="not an address", method="card", test_card="visa"))["field"] == "email"
        assert (await gate.register(ticket=seen["ticket"], name="Zed", email="abdullahweb24bd@gmail.com", method="card", test_card="visa"))["field"] == "email"
        pay.fail = RuntimeError("card refused")
        refused = await gate.register(ticket=seen["ticket"], name="Zed", email="zed@example.com", method="card", test_card="visa")
        assert refused["status"] == "error" and "card refused" in refused["message"]
        assert not book.profiles  # no method, no profile, no way in
        pay.enabled, pay.problem = False, "PAYPAL_ENV is not 'sandbox'"
        assert "sandbox" in (await gate.register(ticket=seen["ticket"], name="Zed", email="zed@example.com", method="card", test_card="visa"))["message"]
        assert (await gate.register(ticket="gone", name="Zed", email="zed@example.com"))["status"] == "error"

    run(scenario())


def test_the_gate_searches_the_vector_index_when_there_is_one(tmp_path):
    class Index:
        enabled = True

        def __init__(self):
            self.rows, self.asked = {}, 0

        def sync(self, rows):
            self.rows = {(pid, local): sig for local, pid, sig in rows}

        def add_later(self, profile, local_id, signature):
            self.rows[(profile, local_id)] = signature

        def search(self, signature, nearest=24):
            self.asked += 1
            best = {}
            for (pid, _), sig in self.rows.items():
                best[pid] = max(best.get(pid, -1.0), float(sig.ravel() @ signature.ravel()))
            return sorted(best.items(), key=lambda pair: -pair[1])

        def forget_all(self):
            self.rows = {}

    async def scenario():
        index = Index()
        book = Book(tmp_path / "checkout.db", index=index)
        gate = Gate(book, FakePay())
        ada = book.add("Ada", face(8), "ada@example.com")
        book.set_card(ada, "c", "t", "visa", "1881")
        assert len(index.rows) == 1  # the embedding went to the index as it was learnt
        seen = await gate.see(face(8))
        assert seen["status"] == "known" and seen["profile"]["name"] == "Ada" and index.asked == 1
        assert (await gate.see(face(9)))["status"] == "new" and index.asked == 2

        def down(signature, nearest=24):
            raise ConnectionError("database asleep")

        index.search = down  # the database is unreachable: the local copy answers
        assert (await gate.see(face(8)))["status"] == "known"
        assert len(Book(tmp_path / "checkout.db", index=Index()).index.rows) == 1  # a restart puts the local copies back
        book.forget_all()
        assert index.rows == {}

    run(scenario())


# ---------------------------------------------------------------- who the shop floor bills


def lookalike(of: Face, seed: int, score: float = 0.4) -> Face:
    """A face that scores `score` against `of`: like that person, but not sure enough (between MATCH and SURE)."""
    rest = np.random.default_rng(seed).normal(size=of.signature.shape).astype(np.float32)
    rest -= (rest.ravel() @ of.signature.ravel()) * of.signature
    rest /= np.linalg.norm(rest)
    return Face((score * of.signature + np.sqrt(1 - score**2) * rest).astype(np.float32), 120.0, b"\xff\xd8jpeg")


def in_view(shop, faces: dict[int, Face]) -> None:
    """Put each face on the head of the shopper at that place, left to right."""
    state = shop.engine.snapshot()
    ids = shop.ids()
    centre = {p["id"]: ((p["box"][0] + p["box"][2]) / 2, p["box"][1] + 0.05) for p in state["people"]}
    shop.book.scene = [(f, *centre[ids[i]]) for i, f in faces.items()]


def look(shop, times: int = 2) -> None:
    state = shop.engine.snapshot()
    for _ in range(times):
        shop.session._faces(None, state)


def registered(book, name: str, seed: int, customer: str):
    profile = book.add(name, face(seed), f"{name.split()[0].lower()}@example.com")
    book.set_card(profile, customer, f"tok_{customer}", "visa", "1881")
    return profile


def test_registered_shoppers_are_billed_with_or_without_the_gate(tmp_path):
    async def scenario():
        shop = Shop(tmp_path, book=Book(tmp_path / "checkout.db"))
        ids = play(shop)  # shoppers 0, 2 and 4 are holding an item each
        book, session = shop.book, shop.session
        admitted = registered(book, "Admitted", 10, "cust_a")
        book.admit(admitted)
        skipped = registered(book, "Skipped Gate", 11, "cust_s")  # has a method, did not come through the gate
        in_view(shop, {0: face(10), 2: face(11), 4: face(12)})
        look(shop)

        # the one who skipped the gate is still who they are, and is checked in on the floor; a face nobody knows is not
        assert session.who == {ids[0]: admitted, ids[2]: skipped} and session.walk_ins == {ids[2]}
        assert session.strangers == {ids[4]: ""}
        assert book.open_visit(skipped) is not None  # the floor opened their visit, as the gate would have
        checkins = {e["person"]: e["how"] for e in shop.engine.events if e["type"] == "checkin"}
        assert checkins == {ids[0]: "recognised", ids[2]: "floor"}
        people = session.meta()["people"]
        assert people[ids[2]]["status"] == "known" and people[ids[4]] == {"status": "not_registered", "name": "", "face": None}
        assert [e["person"] for e in shop.engine.events if e["type"] == "unregistered"] == [ids[4]]  # said once

        await session.finish()
        assert sorted(c["customer"] for c in shop.pay.charges) == ["cust_a", "cust_s"]
        assert session.payments[ids[2]][-1]["status"] == "succeeded" and session.payments[ids[4]][-1]["status"] == "not_registered"
        assert book.open_visit(admitted) is None and book.open_visit(skipped) is None  # both left
        assert book.summary(skipped)["visits"] == 1

    run(scenario())


def test_one_sure_look_is_not_enough_without_the_gate(tmp_path):
    shop = Shop(tmp_path, book=Book(tmp_path / "checkout.db"))
    ids = play(shop)
    skipped = registered(shop.book, "Skipped Gate", 11, "cust_s")
    in_view(shop, {0: face(11)})
    look(shop, 1)
    assert not shop.session.who and shop.book.open_visit(skipped) is None
    look(shop, 1)
    assert shop.session.who == {ids[0]: skipped}


def test_a_weak_resemblance_is_never_charged(tmp_path):
    async def scenario():
        shop = Shop(tmp_path, book=Book(tmp_path / "checkout.db"))
        ids = play(shop)
        book, session = shop.book, shop.session
        alike = registered(book, "Alike", 13, "cust_l")
        in_view(shop, {0: lookalike(face(13), 99)})
        look(shop, 4)
        assert not session.who and session.strangers[ids[0]] == "Alike" and book.open_visit(alike) is None
        await session.finish()
        assert shop.pay.charges == [] and session.payments[ids[0]][-1]["status"] == "not_registered"

    run(scenario())


def test_someone_flagged_is_picked_up_once_the_gate_lets_them_in(tmp_path):
    shop = Shop(tmp_path, book=Book(tmp_path / "checkout.db"))
    ids = play(shop)
    book, session = shop.book, shop.session
    late = registered(book, "Late", 13, "cust_l")
    in_view(shop, {0: lookalike(face(13), 98)})
    look(shop)
    assert session.strangers == {ids[0]: "Late"} and not session.who
    book.admit(late)  # the gate is sure of them; the floor needs only a likeness now
    look(shop)
    assert session.who == {ids[0]: late} and not session.strangers and not session.walk_ins


def test_a_rehearsal_checks_in_on_the_floor_but_opens_no_visit(tmp_path):
    async def scenario():
        shop = Shop(tmp_path, book=Book(tmp_path / "checkout.db"), dry=True)
        ids = play(shop)
        skipped = registered(shop.book, "Skipped Gate", 11, "cust_s")
        in_view(shop, {0: face(11)})
        look(shop)
        assert shop.session.who == {ids[0]: skipped} and shop.book.open_visit(skipped) is None
        await shop.session.finish()
        assert shop.pay.charges == [] and shop.session.payments[ids[0]][-1]["status"] == "dry"

    run(scenario())


def test_two_runs_watching_a_shopper_who_skipped_the_gate_charge_once(tmp_path):
    async def scenario():
        book = Book(tmp_path / "checkout.db")
        skipped = registered(book, "Skipped Gate", 11, "cust_s")
        first, second = Shop(tmp_path, book=book), Shop(tmp_path, book=book)
        play(first)
        pid = play(second)[0]
        in_view(first, {0: face(11)})
        look(first)
        await first.session.finish()
        assert [c["customer"] for c in first.pay.charges] == ["cust_s"]

        # the same walk-out, watched by another run a moment later: recognised, and not charged again
        in_view(second, {0: face(11)})
        look(second)
        assert second.session.who == {pid: skipped} and book.open_visit(skipped) is None
        await second.session.finish()
        assert second.pay.charges == [] and second.session.payments[pid][-1]["status"] == "settled"

    run(scenario())


def test_a_live_camera_bills_a_shopper_who_comes_back_after_paying(tmp_path):
    async def scenario():
        book = Book(tmp_path / "checkout.db")
        skipped = registered(book, "Skipped Gate", 11, "cust_s")
        first = Shop(tmp_path, book=book)
        first.session.live = True
        play(first)
        in_view(first, {0: face(11)})
        look(first)
        await first.session.finish()
        assert [c["customer"] for c in first.pay.charges] == ["cust_s"]

        # a page on the same camera that was already watching while that visit was settled: the same walk-out
        alongside = Shop(tmp_path, book=book)
        alongside.session.live, alongside.session.started = True, first.session.started
        pid = play(alongside)[0]
        in_view(alongside, {0: face(11)})
        look(alongside)
        await alongside.session.finish()
        assert alongside.pay.charges == [] and alongside.session.payments[pid][-1]["status"] == "settled"

        # a page reloaded straight after the charge, the items still in their hands: not a new visit either
        reloaded = Shop(tmp_path, book=book)
        reloaded.session.live, reloaded.session.started = True, time.time() + 1
        pid = play(reloaded)[0]
        in_view(reloaded, {0: face(11)})
        look(reloaded)
        await reloaded.session.finish()
        assert reloaded.pay.charges == [] and reloaded.session.payments[pid][-1]["status"] == "settled"

        # a run started a few minutes after that visit was settled: they came back and took more, a new visit
        book.db.execute("UPDATE admissions SET closed = closed - 300")
        again = Shop(tmp_path, book=book)
        again.session.live, again.session.started = True, time.time() + 1
        pid = play(again)[0]
        in_view(again, {0: face(11)})
        look(again)
        await again.session.finish()
        assert [c["customer"] for c in again.pay.charges] == ["cust_s"]

    run(scenario())

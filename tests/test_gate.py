"""
The gate: a known face with a payment method is let in, a new face registers first, and only people the gate let in
are billed on the shop floor.

Faces here are made-up embeddings compared with numpy, and the payment provider is a stand-in, so no model file,
camera or network is needed.
"""
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


# ---------------------------------------------------------------- the shop floor only bills open visits


def test_only_people_the_gate_let_in_are_billed(tmp_path):
    async def scenario():
        shop = Shop(tmp_path, book=Book(tmp_path / "checkout.db"))
        ids = play(shop)  # shoppers 0, 2 and 4 are holding an item each
        book, session = shop.book, shop.session
        admitted = book.add("Admitted", face(10), "in@example.com")
        book.set_card(admitted, "cust_a", "tok_a", "visa", "1881")
        book.admit(admitted)
        stayed_out = book.add("Stayed Out", face(11), "out@example.com")  # has a method, did not come through the gate
        book.set_card(stayed_out, "cust_o", "tok_o", "visa", "1881")

        state = shop.engine.snapshot()
        centre = {p["id"]: ((p["box"][0] + p["box"][2]) / 2, p["box"][1] + 0.05) for p in state["people"]}
        book.scene = [(face(10), *centre[ids[0]]), (face(11), *centre[ids[2]]), (face(12), *centre[ids[4]])]
        for _ in range(2):
            session._faces(None, state)

        assert session.who == {ids[0]: admitted}
        assert session.strangers == {ids[2]: "Stayed Out", ids[4]: ""}
        people = session.meta()["people"]
        assert people[ids[0]]["status"] == "known" and people[ids[2]] == {"status": "not_registered", "name": "Stayed Out"}
        assert people[ids[4]] == {"status": "not_registered", "name": ""}
        flagged = [e for e in shop.engine.events if e["type"] == "unregistered"]
        assert sorted(e["person"] for e in flagged) == sorted([ids[2], ids[4]])  # said once each, not on every frame

        await session.finish()
        assert [c["customer"] for c in shop.pay.charges] == ["cust_a"]  # one charge: the person the gate let in
        assert session.payments[ids[0]][-1]["status"] == "succeeded"
        assert session.payments[ids[2]][-1]["status"] == "not_registered" and session.payments[ids[4]][-1]["status"] == "not_registered"
        assert book.open_visit(admitted) is None  # they left: the next visit starts at the gate
        assert book.summary(stayed_out)["visits"] == 0

    run(scenario())


def test_someone_flagged_is_picked_up_once_the_gate_lets_them_in(tmp_path):
    shop = Shop(tmp_path, book=Book(tmp_path / "checkout.db"))
    ids = play(shop)
    book, session = shop.book, shop.session
    late = book.add("Late", face(13), "late@example.com")
    book.set_card(late, "cust_l", "tok_l", "visa", "1881")
    state = shop.engine.snapshot()
    box = next(p["box"] for p in state["people"] if p["id"] == ids[0])
    book.scene = [(face(13), (box[0] + box[2]) / 2, box[1] + 0.05)]
    for _ in range(2):
        session._faces(None, state)
    assert session.strangers == {ids[0]: "Late"} and not session.who
    book.admit(late)
    session._faces(None, state)
    assert session.who == {ids[0]: late} and not session.strangers

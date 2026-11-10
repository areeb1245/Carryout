"""
Five shoppers at once, and the rule that matters at the door: you pay for what you walk out with, nothing else.

The scene is a shelf along the top of the frame and five shoppers in a row below it, each reaching up.
The engine runs on synthetic masks; payments and email are stand-ins that record what they were asked to do.
"""
import asyncio

import pytest

from checkout.catalog import Catalog
from checkout.identity import Profiles
from checkout.payments import Card
from checkout.pipeline import Services, Session

CATALOG = Catalog.from_dict({"products": [{"id": "can", "name": "Can", "prompt": "drink can", "price": 3.0}, {"id": "box", "name": "Box", "prompt": "cereal box", "price": 10.0}], "zones": ["shelf"]})
DT = 1 / 3
XS = [0.1, 0.3, 0.5, 0.7, 0.9]
SHELF_Y, HOME_Y = 0.12, 0.45


def rect(label, cx, cy, w, h, score=0.9):
    x0, y0, x1, y1 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
    return {"label": label, "score": score, "box": [x0, y0, x1, y1], "polygons": [[[x0, y0], [x1, y0], [x1, y1], [x0, y1]]]}


def shopper(px, hand_y):
    """Body from y 0.4 down, one arm straight up to a hand at `hand_y` (the person mask includes arm and hand)."""
    body = rect("person", px, 0.7, 0.12, 0.6, 0.95)
    hand = rect("hand", px, hand_y, 0.04, 0.05, 0.85)
    body["polygons"].append([[px - 0.015, hand_y], [px + 0.015, hand_y], [px + 0.015, 0.42], [px - 0.015, 0.42]])
    body["polygons"].append(hand["polygons"][0])
    return [body, hand]


def item(px, y, label):
    return rect(label, px + 0.035, y, 0.03, 0.06, 0.8)


def lerp(a, b, k):
    return a + (b - a) * k


class FakePayments:
    enabled, problem = True, ""

    def __init__(self):
        self.charges = []

    async def save_card(self, name, test_card, shopper_id, email=None):
        return Card(f"cus_{shopper_id}", f"pm_{shopper_id}", "visa", "4242")

    async def charge(self, card, amount, currency, description, key, metadata=None):
        self.charges.append({"customer": card.customer, "amount": amount, "key": key, "description": description})
        return {"status": "succeeded", "id": f"pi_{len(self.charges)}", "amount": amount, "currency": currency.lower(), "receipt_url": None, "message": ""}


class FakeMailer:
    enabled = True

    def __init__(self):
        self.sent = []

    async def receipt(self, to, name, lines, payment, key):
        self.sent.append({"to": to, "lines": lines, "amount": payment["amount"]})
        return {"status": "sent", "to": to, "id": "em_1", "message": ""}


class Shop:
    def __init__(self, tmp_path, book=None, verifier=None, dry=False):
        self.pay, self.mail = FakePayments(), FakeMailer()
        self.book = book or Profiles(tmp_path / "checkout.db")
        self.session = Session(Services(None, verifier, self.book, self.pay, self.mail), CATALOG, dry=dry)
        self.engine = self.session.engine
        self.t = 0.0
        self.labels = ["drink can", "cereal box", "drink can", "cereal box", "drink can"]
        # where each shopper's hand and each item is; items start on the shelf
        self.hand = [HOME_Y] * 5
        self.item = [SHELF_Y] * 5
        self.item_seen = [True] * 5
        self.present = [True] * 5

    def frame(self):
        dets = [rect("shelf", 0.5, 0.12, 1.0, 0.2, 0.8)]
        for i, px in enumerate(XS):
            if self.present[i]:
                dets += shopper(px, self.hand[i])
            if self.item_seen[i]:
                dets.append(item(px, self.item[i], self.labels[i]))
        self.engine.step(self.t, {"image": {"w": 960, "h": 540}, "detections": dets})
        self.t += DT

    def run(self, seconds, move=None):
        n = max(2, round(seconds / DT))
        for k in range(n):
            if move:
                move(k / (n - 1))
            self.frame()

    def ids(self):
        """Person id of each of the five, left to right."""
        people = sorted(self.engine.snapshot()["people"], key=lambda p: p["box"][0])
        return [p["id"] for p in people]

    def cart(self, pid):
        return next(c for c in self.engine.carts() if c["person"] == pid)


def play(shop: Shop):
    """0, 2 and 4 take their item home. 1 takes it and puts it back. 3 only touches the shelf item."""
    shop.run(2.5)
    ids = shop.ids()
    assert len(ids) == 5

    def reach(k):
        shop.hand = [lerp(HOME_Y, SHELF_Y, k)] * 5

    def bring(k):
        for i in (0, 1, 2, 4):
            shop.hand[i] = lerp(SHELF_Y, HOME_Y, k)
            shop.item[i] = lerp(SHELF_Y, HOME_Y, k)
        shop.hand[3] = lerp(SHELF_Y, HOME_Y, k)

    shop.run(1.0, reach)
    shop.run(0.7)
    shop.run(1.0, bring)
    shop.run(1.5)

    def back(k):  # shopper 1 changes their mind
        shop.hand[1] = lerp(HOME_Y, SHELF_Y, k)
        shop.item[1] = lerp(HOME_Y, SHELF_Y, k)

    def let_go(k):
        shop.hand[1] = lerp(SHELF_Y, HOME_Y, k)

    shop.run(1.0, back)
    shop.run(0.7, let_go)
    shop.run(2.0)
    return ids


def test_five_shoppers_each_get_only_their_own_items(tmp_path):
    shop = Shop(tmp_path)
    ids = play(shop)
    picks = [e for e in shop.engine.events if e["type"] == "pick"]
    assert sorted(e["person"] for e in picks) == sorted(ids[i] for i in (0, 1, 2, 4))
    assert [shop.cart(ids[i])["total"] for i in range(5)] == [3.0, 0.0, 3.0, 0.0, 3.0]
    assert [line["state"] for line in shop.cart(ids[1])["lines"]] == ["returned"]
    assert shop.cart(ids[3])["lines"] == []
    assert len(shop.ids()) == 5 and shop.ids() == ids  # nobody swapped identity along the way


def test_walking_out_charges_what_was_kept_and_nothing_that_was_put_back(tmp_path):
    async def scenario():
        shop = Shop(tmp_path)
        ids = play(shop)
        profiles = []
        for i, pid in enumerate(ids):  # everyone has a profile and a card; shopper 0 also gave an email address
            profile = shop.book.add(f"Shopper {i}", None, "zero@example.com" if i == 0 else "")
            shop.book.set_card(profile, f"cus_{i}", f"pm_{i}", "visa", "4242")
            shop.book.admit(profile)
            shop.session.who[pid] = profile
            profiles.append(profile)
        # the three who kept an item walk out with it; the other two stay
        for i in (0, 2, 4):
            shop.present[i] = False
            shop.item_seen[i] = False
        first = len(shop.engine.events)
        for _ in range(14):
            shop.frame()
            for e in shop.engine.events[first:]:
                if e["type"] == "exit":
                    shop.session._spawn(shop.session._pay(e["person"]))
            first = len(shop.engine.events)
        while shop.session.jobs:
            await asyncio.wait(list(shop.session.jobs))
        assert sorted(c["customer"] for c in shop.pay.charges) == ["cus_0", "cus_2", "cus_4"]
        assert all(c["amount"] == 3.0 for c in shop.pay.charges)
        assert len({c["key"] for c in shop.pay.charges}) == 3
        assert shop.mail.sent == [{"to": "zero@example.com", "lines": [("Can", 3.0, "")], "amount": 3.0}]

        # closing time: the two still inside walk out. One put their item back, one never took any: no charge.
        await shop.session.finish()
        assert len(shop.pay.charges) == 3
        statuses = {pid: shop.session.payments[pid][-1]["status"] for pid in ids}
        assert [statuses[pid] for pid in ids] == ["succeeded", "nothing_due", "succeeded", "nothing_due", "succeeded"]
        # whoever walked out with something has left: their next visit starts at the gate again. The two who
        # left the picture with empty hands have not finished a visit, so theirs stays open.
        assert [shop.book.open_visit(p) is None for p in profiles] == [True, False, True, False, True]
        # the visit history says the same
        assert [shop.book.summary(p)["spent"] for p in profiles] == [3.0, 0.0, 3.0, 0.0, 3.0]
        assert shop.book.summary(profiles[0])["recent"] == ["Can"]
        # and a second closing does not charge anyone twice
        await shop.session.finish()
        assert len(shop.pay.charges) == 3

    asyncio.run(scenario())


def test_card_declined_leaves_the_lines_unpaid(tmp_path):
    async def scenario():
        shop = Shop(tmp_path)
        ids = play(shop)

        async def decline(card, amount, currency, description, key, metadata=None):
            return {"status": "declined", "id": "pi_x", "amount": amount, "currency": "usd", "receipt_url": None, "message": "Your card was declined."}

        shop.pay.charge = decline
        profile = shop.book.add("Declined", None)
        shop.book.set_card(profile, "cus_d", "pm_d", "visa", "0341")
        shop.book.admit(profile)  # the floor only links a shopper the gate has let in
        shop.session.who[ids[0]] = profile
        await shop.session.finish()
        assert shop.session.payments[ids[0]][-1]["status"] == "declined"
        assert shop.cart(ids[0])["due"] == 3.0 and not shop.cart(ids[0])["lines"][0]["paid"]
        assert shop.book.summary(profile)["unpaid"] == 3.0 and not shop.mail.sent
        assert shop.session.payments[ids[2]][-1]["status"] == "not_registered"  # walked out without coming through the gate

    asyncio.run(scenario())

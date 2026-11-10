"""
Left behind: a product on the bill is charged only when it is still linked to the shopper as they leave. Either the
floor saw it with them in their last few seconds in view, or the door look sees it in their last pictures. An item
that dropped out of sight long before they left, and is not with them at the door, is kept off the charge for review.
"""
import asyncio

import numpy as np

from checkout.pipeline import LINK_S
from tests.test_crowd import HOME_Y
from tests.test_last_check import PICTURE, FakeVerifier, line, shop_with


class DoorVerifier(FakeVerifier):
    """The last check says "kept" for everyone; the door look answers what a test gives each shopper."""

    def __init__(self, door=None):
        super().__init__()
        self.door, self.door_asks = door or {}, 0

    async def still_carried(self, frame, box, products, refs=None):
        self.door_asks += 1
        return self.door.get(round(float(frame[0, 0, 0])), {"products": [], "unknown": 0, "hands_seen": 1, "why": ""})


def last_frames(shop, ids, i, n=4):
    """Door pictures for shopper `i`, their first pixel set to `i` so the fake door look knows whose they are."""
    frame = np.full((40, 40, 3), i, np.uint8)
    shop.session.evidence[ids[i]]["last"] = [(shop.t - (n - k) * 0.45, frame, [0.1, 0.1, 0.4, 0.9], PICTURE) for k in range(n)]


def hide_then_stay(shop, i, seconds):
    """Shopper `i`'s item drops out of sight (put down out of view) while they stay in the picture with empty hands."""
    shop.item_seen[i] = False
    shop.hand[i] = HOME_Y
    shop.run(seconds)


def test_an_item_out_of_sight_long_before_they_leave_is_not_charged(tmp_path):
    async def scenario():
        verifier = DoorVerifier()
        shop, ids, _ = shop_with(tmp_path, verifier)
        hide_then_stay(shop, 0, LINK_S + 3)
        last_frames(shop, ids, 0)
        await shop.session.finish()

        held = line(shop, ids[0])
        assert (held["audit"], held["charge"], held["paid"]) == ("unclear", False, None)
        assert held["audit_why"].startswith("Not seen with the shopper as they left")
        assert sorted(c["customer"] for c in shop.pay.charges) == ["cust_2", "cust_4"]  # the others still pay
        assert shop.cart(ids[0])["held_back"] == 3.0  # on the dashboard for review, not dropped

    asyncio.run(scenario())


def test_the_door_look_seeing_it_with_them_keeps_it_on_the_bill(tmp_path):
    async def scenario():
        verifier = DoorVerifier({0: {"products": ["can"], "unknown": 0, "hands_seen": 1, "why": "holds a can"}})
        shop, ids, _ = shop_with(tmp_path, verifier)
        hide_then_stay(shop, 0, LINK_S + 3)
        last_frames(shop, ids, 0)
        await shop.session.finish()
        assert line(shop, ids[0])["paid"] and "cust_0" in [c["customer"] for c in shop.pay.charges]

    asyncio.run(scenario())


def test_something_unnamed_in_their_hands_gets_the_benefit_of_the_doubt(tmp_path):
    async def scenario():
        verifier = DoorVerifier({0: {"products": [], "unknown": 1, "hands_seen": 1, "why": "holds something dark"}})
        shop, ids, _ = shop_with(tmp_path, verifier)
        hide_then_stay(shop, 0, LINK_S + 3)
        last_frames(shop, ids, 0)
        await shop.session.finish()
        assert line(shop, ids[0])["paid"]

    asyncio.run(scenario())


def test_an_item_seen_with_them_moments_before_they_leave_is_charged(tmp_path):
    async def scenario():
        verifier = DoorVerifier()  # the door look sees one hand and nothing in it: not enough to take anything off
        shop, ids, _ = shop_with(tmp_path, verifier)
        last_frames(shop, ids, 0)
        await shop.session.finish()
        assert line(shop, ids[0])["paid"] and verifier.door_asks == 4

    asyncio.run(scenario())


def test_both_hands_empty_at_the_door_takes_it_off_even_when_seen_lately(tmp_path):
    async def scenario():
        empty = {"products": [], "unknown": 0, "hands_seen": 2, "why": "both hands empty"}
        verifier = DoorVerifier({0: empty})
        shop, ids, _ = shop_with(tmp_path, verifier)
        last_frames(shop, ids, 0)
        await shop.session.finish()
        held = line(shop, ids[0])
        assert (held["audit"], held["charge"]) == ("put_back", False)

    asyncio.run(scenario())


def test_without_the_second_model_an_item_long_out_of_sight_is_still_held(tmp_path):
    async def scenario():
        shop, ids, _ = shop_with(tmp_path, None)
        hide_then_stay(shop, 0, LINK_S + 3)
        await shop.session.finish()
        assert line(shop, ids[0])["charge"] is False and [c["customer"] for c in shop.pay.charges] == ["cust_2", "cust_4"]

    asyncio.run(scenario())


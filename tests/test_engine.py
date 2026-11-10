"""
Engine scenarios on synthetic masks: rectangles standing in for what SAM returns, at 3 frames a second.

The scene is a shelf on the right (x 0.70..1.0). Shoppers stand left of it and reach in with a hand.
"""
import numpy as np
import pytest

from checkout.catalog import Catalog
from checkout.engine import Engine

CATALOG = Catalog.from_dict(
    {
        "products": [{"id": "can", "name": "Can", "prompt": "drink can", "price": 3.0}, {"id": "box", "name": "Box", "prompt": "cereal box", "price": 10.0}],
        "belongings": ["mobile phone"],
        "zones": ["shelf"],
    }
)
DT = 1 / 3
SHELF = (0.70, 0.05, 1.0, 0.95)


def rect(label, cx, cy, w, h, score=0.9):
    x0, y0, x1, y1 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
    return {"label": label, "score": score, "box": [x0, y0, x1, y1], "polygons": [[[x0, y0], [x1, y0], [x1, y1], [x0, y1]]]}


def person(cx, cy=0.55, w=0.16, h=0.8):
    return rect("person", cx, cy, w, h, 0.95)


def hand(x, y):
    return rect("hand", x, y, 0.04, 0.05, 0.85)


def arm(px, hx, hy, w=0.16):
    """A shopper at `px` whose arm reaches to a hand at (hx, hy): body, arm (part of the person mask) and hand."""
    body = person(px)
    edge = px + w / 2 if hx > px else px - w / 2
    x0, x1 = sorted((edge, hx))
    body["polygons"].append([[x0, hy - 0.02], [x1, hy - 0.02], [x1, hy + 0.02], [x0, hy + 0.02]])
    own = hand(hx, hy)
    body["polygons"].append(own["polygons"][0])  # SAM's person mask includes the hand
    return [body, own]


def can(x, y, score=0.8, label="drink can"):
    return rect(label, x, y, 0.035, 0.07, score)


class Scene:
    def __init__(self):
        self.engine = Engine(CATALOG)
        self.t = 0.0
        self.events = []

    def frame(self, *dets, frame=None):
        flat = [rect("shelf", (SHELF[0] + SHELF[2]) / 2, (SHELF[1] + SHELF[3]) / 2, SHELF[2] - SHELF[0], SHELF[3] - SHELF[1], 0.8)]
        for d in dets:
            flat.extend(d if isinstance(d, list) else [d])
        state = self.engine.step(self.t, {"image": {"w": 960, "h": 540}, "detections": flat}, frame)
        self.events.extend(state["events"])
        self.t += DT
        return state

    def hold(self, seconds, *dets):
        state = None
        for _ in range(max(1, round(seconds / DT))):
            state = self.frame(*dets)
        return state

    def glide(self, seconds, make):
        """`make(k)` returns the detections for progress k in 0..1."""
        n = max(2, round(seconds / DT))
        state = None
        for i in range(n):
            state = self.frame(*make(i / (n - 1)))
        return state

    def of(self, kind):
        return [e for e in self.events if e["type"] == kind]

    def cart(self, pid):
        return next(c for c in self.engine.carts() if c["person"] == pid)


def lerp(a, b, k):
    return a + (b - a) * k


def pick_from_shelf(s: Scene, px=0.5, item=(0.75, 0.4), others=()):
    """Shopper at `px` reaches to the can resting at `item`, grips it and brings it to the chest."""
    ix, iy = item
    rest_hand = (px + 0.06, 0.6)
    s.glide(1.0, lambda k: [arm(px, lerp(rest_hand[0], ix - 0.03, k), lerp(rest_hand[1], iy, k)), can(ix, iy), *others])
    s.hold(0.4, arm(px, ix - 0.03, iy), can(ix, iy), *others)
    s.glide(1.0, lambda k: [arm(px, lerp(ix - 0.03, rest_hand[0], k), lerp(iy, rest_hand[1], k)), can(lerp(ix, rest_hand[0] + 0.03, k), lerp(iy, rest_hand[1], k)), *others])
    return rest_hand


def test_pick_from_shelf_goes_to_the_shopper():
    s = Scene()
    s.hold(2.5, arm(0.5, 0.56, 0.6), can(0.75, 0.4))
    assert not s.of("pick")
    hx, hy = pick_from_shelf(s)
    s.hold(1.0, arm(0.5, hx, hy), can(hx + 0.03, hy))
    picks = s.of("pick")
    assert len(picks) == 1 and picks[0]["person"] == 1 and picks[0]["origin"] == "shelf"
    cart = s.cart(1)
    assert cart["count"] == 1 and cart["total"] == 3.0 and cart["lines"][0]["name"] == "Can"


def test_touching_without_taking_is_not_a_pick():
    s = Scene()
    s.hold(2.5, arm(0.5, 0.56, 0.6), can(0.75, 0.4))
    s.glide(1.0, lambda k: [arm(0.5, lerp(0.56, 0.72, k), lerp(0.6, 0.4, k)), can(0.75, 0.4)])
    s.hold(2.0, arm(0.5, 0.72, 0.4), can(0.75, 0.4))
    s.glide(1.0, lambda k: [arm(0.5, lerp(0.72, 0.56, k), lerp(0.4, 0.6, k)), can(0.75, 0.4)])
    s.hold(2.0, arm(0.5, 0.56, 0.6), can(0.75, 0.4))
    assert not s.of("pick") and s.cart(1)["count"] == 0


def test_two_shoppers_at_one_shelf_the_one_whose_hand_took_it_pays():
    s = Scene()
    # shopper 1 stands at 0.35 with a hand hovering near the shelf, shopper 2 at 0.55 takes the can
    other = arm(0.3, 0.68, 0.62)
    s.hold(2.5, other, arm(0.56, 0.62, 0.6), can(0.75, 0.4))
    ids = {round(p["box"][0], 1): p["id"] for p in s.frame(other, arm(0.56, 0.62, 0.6), can(0.75, 0.4))["people"]}
    taker = ids[min(ids, key=lambda x: abs(x - 0.48))]
    hx, hy = pick_from_shelf(s, px=0.56, others=other)
    s.hold(1.0, other, arm(0.56, hx, hy), can(hx + 0.03, hy))
    picks = s.of("pick")
    assert len(picks) == 1 and picks[0]["person"] == taker
    assert picks[0]["confidence"] >= 0.6
    assert sum(c["count"] for c in s.engine.carts()) == 1


def test_putting_it_back_empties_the_cart():
    s = Scene()
    s.hold(2.5, arm(0.5, 0.56, 0.6), can(0.75, 0.4))
    hx, hy = pick_from_shelf(s)
    s.hold(1.5, arm(0.5, hx, hy), can(hx + 0.03, hy))
    assert s.cart(1)["count"] == 1
    # back to the shelf, let go, withdraw
    s.glide(1.0, lambda k: [arm(0.5, lerp(hx, 0.72, k), lerp(hy, 0.4, k)), can(lerp(hx + 0.03, 0.75, k), lerp(hy, 0.4, k))])
    s.glide(0.7, lambda k: [arm(0.5, lerp(0.72, hx, k), lerp(0.4, hy, k)), can(0.75, 0.4)])
    s.hold(2.0, arm(0.5, hx, hy), can(0.75, 0.4))
    assert len(s.of("return")) == 1
    cart = s.cart(1)
    assert cart["count"] == 0 and cart["total"] == 0 and cart["lines"][0]["state"] == "returned"


def test_item_put_in_a_bag_stays_in_the_cart():
    s = Scene()
    s.hold(2.5, arm(0.5, 0.56, 0.6), can(0.75, 0.4))
    hx, hy = pick_from_shelf(s)
    s.hold(1.0, arm(0.5, hx, hy), can(hx + 0.03, hy))
    s.hold(5.0, arm(0.5, hx, hy))  # the can is out of sight, the hands are empty, and nobody went near the shelf
    assert not s.of("return") and s.cart(1)["count"] == 1


def test_item_hidden_for_a_moment_is_not_counted_twice():
    s = Scene()
    s.hold(2.5, arm(0.5, 0.56, 0.6), can(0.75, 0.4))
    hx, hy = pick_from_shelf(s)
    s.hold(1.0, arm(0.5, hx, hy), can(hx + 0.03, hy))
    s.hold(2.5, arm(0.5, hx, hy))
    s.hold(2.0, arm(0.5, hx - 0.1, hy + 0.1), can(hx - 0.07, hy + 0.1))  # shows up again in the other hand
    assert len(s.of("pick")) == 1 and s.cart(1)["count"] == 1


def test_second_identical_item_is_a_second_line():
    s = Scene()
    two = can(0.80, 0.4)
    s.hold(2.5, arm(0.5, 0.56, 0.6), can(0.75, 0.4), two)
    hx, hy = pick_from_shelf(s, others=[two])
    s.hold(1.0, arm(0.5, hx, hy), can(hx + 0.03, hy), two)
    s.hold(2.5, arm(0.5, hx, hy), two)  # first one into the bag
    pick_from_shelf(s, item=(0.80, 0.4))
    s.hold(1.0, arm(0.5, hx, hy), can(hx + 0.03, hy))
    assert len(s.of("pick")) == 2 and s.cart(1)["count"] == 2 and s.cart(1)["total"] == 6.0


def test_item_first_seen_in_a_hand_at_the_shelf_is_a_pick():
    """Stock behind glass or packed too tight to segment: the item only becomes visible in the hand."""
    s = Scene()
    s.hold(2.0, arm(0.5, 0.56, 0.6))
    s.glide(1.0, lambda k: [arm(0.5, lerp(0.56, 0.74, k), lerp(0.6, 0.4, k))])
    s.glide(1.2, lambda k: [arm(0.5, lerp(0.74, 0.56, k), lerp(0.4, 0.6, k)), can(lerp(0.77, 0.59, k), lerp(0.4, 0.6, k), label="item held in hand")])
    s.hold(1.5, arm(0.5, 0.56, 0.6), can(0.59, 0.6, label="cereal box"), can(0.59, 0.6, 0.6, label="item held in hand"))
    picks = s.of("pick")
    assert len(picks) == 1 and picks[0]["origin"] == "hand"
    assert s.cart(1)["lines"][0]["name"] == "Box" and s.cart(1)["total"] == 10.0


def test_hand_over_moves_the_item_to_the_other_cart():
    s = Scene()
    two = arm(0.2, 0.26, 0.6)
    s.hold(2.5, two, arm(0.5, 0.56, 0.6), can(0.75, 0.4))
    hx, hy = pick_from_shelf(s, others=two)
    people = s.hold(1.0, two, arm(0.5, hx, hy), can(hx + 0.03, hy))["people"]
    giver = s.of("pick")[0]["person"]
    taker = next(p["id"] for p in people if p["id"] != giver)
    # the giver holds it out to the left, the other shopper takes it and pulls it in
    s.glide(1.0, lambda k: [arm(0.2, 0.26, 0.6), arm(0.5, lerp(hx, 0.38, k), 0.6), can(lerp(hx + 0.03, 0.35, k), 0.6)])
    s.glide(0.7, lambda k: [arm(0.2, lerp(0.26, 0.32, k), 0.6), arm(0.5, 0.38, 0.6), can(0.35, 0.6)])
    s.glide(1.0, lambda k: [arm(0.2, lerp(0.32, 0.27, k), 0.6), arm(0.5, lerp(0.42, 0.44, k), 0.6), can(lerp(0.34, 0.30, k), 0.6)])
    s.hold(2.0, arm(0.2, 0.27, 0.6), arm(0.5, 0.44, 0.6), can(0.30, 0.6))
    assert len(s.of("handover")) == 1
    assert s.cart(giver)["count"] == 0 and s.cart(taker)["count"] == 1


def test_shelf_item_that_vanishes_under_a_hand_is_an_inferred_pick():
    s = Scene()
    s.hold(4.0, arm(0.5, 0.56, 0.6), can(0.75, 0.4))
    s.glide(1.0, lambda k: [arm(0.5, lerp(0.56, 0.73, k), lerp(0.6, 0.4, k)), can(0.75, 0.4)])
    s.hold(0.7, arm(0.5, 0.75, 0.4))  # the hand covers it
    s.glide(0.7, lambda k: [arm(0.5, lerp(0.75, 0.56, k), lerp(0.4, 0.75, k))])  # and comes away closed, the can never seen again
    s.hold(3.0, arm(0.5, 0.56, 0.75))
    picks = s.of("pick")
    assert len(picks) == 1 and picks[0]["origin"] == "inferred" and s.cart(1)["count"] == 1
    assert picks[0]["confidence"] < 0.75


def test_own_phone_is_never_billed():
    s = Scene()
    s.hold(1.0, person(0.5))
    s.hold(3.0, arm(0.5, 0.56, 0.6), can(0.59, 0.6, 0.6, label="mobile phone"), can(0.59, 0.6, 0.5, label="item held in hand"))
    cart = s.cart(1)
    assert cart["count"] == 0 and cart["total"] == 0 and cart["review"] == 0


def test_leaving_closes_the_cart_and_coming_back_reopens_it():
    def picture(shirt, px):
        img = np.full((270, 480, 3), 200, np.uint8)
        img[40:260, int((px - 0.08) * 480) : int((px + 0.08) * 480)] = shirt
        return img

    s = Scene()
    red, blue = (40, 40, 220), (220, 80, 30)
    for _ in range(9):
        s.frame(person(0.3), person(0.6), frame=picture(red, 0.3) | 0 if False else _two(picture, red, 0.3, blue, 0.6))
    ids = sorted(p["id"] for p in s.engine.snapshot()["people"])
    assert ids == [1, 2]
    who_red = next(p["id"] for p in s.engine.snapshot()["people"] if p["box"][0] < 0.4)
    for _ in range(12):  # red walks out, blue stays
        s.frame(person(0.6), frame=_two(picture, None, 0, blue, 0.6))
    assert [e["person"] for e in s.of("exit")] == [who_red]
    for _ in range(6):  # red walks back in on the other side
        s.frame(person(0.85), person(0.6), frame=_two(picture, red, 0.85, blue, 0.6))
    assert [e["person"] for e in s.of("reenter")] == [who_red]
    assert sorted(p["id"] for p in s.engine.snapshot()["people"]) == [1, 2]


def _two(picture, a, ax, b, bx):
    img = np.full((270, 480, 3), 200, np.uint8)
    for colour, px in ((a, ax), (b, bx)):
        if colour is not None:
            img[30:265, int((px - 0.08) * 480) : int((px + 0.08) * 480)] = colour
    return img


@pytest.mark.parametrize("fps", [2, 4])
def test_pick_holds_at_other_frame_rates(fps, monkeypatch):
    import tests.test_engine as me

    monkeypatch.setattr(me, "DT", 1 / fps)
    s = Scene()
    s.hold(2.5, arm(0.5, 0.56, 0.6), can(0.75, 0.4))
    hx, hy = pick_from_shelf(s)
    s.hold(1.5, arm(0.5, hx, hy), can(hx + 0.03, hy))
    assert len(s.of("pick")) == 1 and s.cart(1)["count"] == 1


def test_walking_in_front_of_the_shelf_takes_nothing():
    s = Scene()
    stock = [can(0.75, 0.4), can(0.82, 0.4), can(0.9, 0.4, label="cereal box")]
    s.hold(3.0, *stock)
    # a shopper crosses the whole frame, hiding each item in turn, hands at their side
    n = 18
    for i in range(n):
        px = lerp(-0.05, 1.05, i / (n - 1))
        hidden = [d for d in stock if abs((d["box"][0] + d["box"][2]) / 2 - px) > 0.1]
        s.frame(person(px), hand(px + 0.05, 0.6), *hidden)
    s.hold(4.0, *stock)
    assert not s.of("pick") and all(c["count"] == 0 for c in s.engine.carts())


def test_two_shoppers_picking_at_the_same_moment_each_get_their_own():
    s = Scene()
    a, b = (0.75, 0.3), (0.75, 0.7)

    def both(k_reach, k_back):
        out = []
        for px, (ix, iy), home in ((0.35, a, (0.41, 0.3)), (0.55, b, (0.61, 0.7))):
            hx = lerp(lerp(home[0], ix - 0.03, k_reach), home[0], k_back)
            out.append(arm(px, hx, iy))
            out.append(can(lerp(ix, home[0] + 0.03, k_back), iy, label="drink can" if iy < 0.5 else "cereal box"))
        return out

    s.hold(2.5, *both(0, 0))
    people = {round(p["box"][0], 2): p["id"] for p in s.frame(*both(0, 0))["people"]}
    left, right = people[min(people)], people[max(people)]
    s.glide(1.0, lambda k: both(k, 0))
    s.hold(0.4, *both(1, 0))
    s.glide(1.0, lambda k: both(1, k))
    s.hold(1.0, *both(1, 1))
    assert len(s.of("pick")) == 2
    assert [line["name"] for line in s.cart(left)["lines"]] == ["Can"]
    assert [line["name"] for line in s.cart(right)["lines"]] == ["Box"]

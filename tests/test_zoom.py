"""The closer look round the hands: where it goes, how its outlines come back into the frame, and the frame loop
sending it beside the frame's own call."""
import asyncio

import cv2
import numpy as np

from checkout import zoom
from checkout.catalog import Catalog
from checkout.pipeline import Services, Session

SHOP = Catalog.from_dict({"products": [{"id": "tissue", "name": "Tissue pack", "prompt": "tissue pack", "price": 10.0}], "belongings": ["mobile phone"]})


def test_a_closer_look_only_while_the_hands_are_small():
    near = zoom.regions([[0.45, 0.5, 0.52, 0.7]])  # a few steps back: one hand, small
    assert len(near) == 1
    x0, y0, x1, y1 = near[0]
    assert abs((x1 - x0) - (y1 - y0)) < 1e-9 and x1 - x0 <= zoom.WIDEST  # the frame's shape, blown up
    assert x0 <= 0.45 and x1 >= 0.52 and y0 < 0.5 - 0.1  # takes in the hand and what it holds above it
    assert zoom.regions([[0.0, 0.3, 0.4, 1.0], [0.6, 0.3, 1.0, 1.0]]) == []  # close to the camera: no need
    apart = zoom.regions([[0.2, 0.4, 0.27, 0.6], [0.7, 0.4, 0.77, 0.6]])  # arms out to the sides: one each
    assert len(apart) == 2
    assert zoom.regions([]) == []
    for x0, y0, x1, y1 in apart + near:
        assert 0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1


def test_what_the_closer_look_finds_lands_where_it_is_in_the_frame():
    found = [{"label": "tissue pack", "score": 0.5, "box": [0.5, 0.5, 1.0, 1.0], "polygons": [[[0.5, 0.5], [1.0, 0.5], [1.0, 1.0]]]}]
    (d,) = zoom.back(found, [0.2, 0.4, 0.6, 0.8])
    assert d["box"] == [0.4, 0.6000000000000001, 0.6, 0.8] and d["zoom"] and d["label"] == "tissue pack"
    assert d["polygons"][0][1] == [0.6, 0.6000000000000001]


class Sam:
    """Answers the whole frame with a hand and nothing in it; a cut-out with the tissue pack."""

    def __init__(self):
        self.asked = []

    async def segment(self, jpeg, prompts, threshold=0.3, exemplars=None):
        image = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        self.asked.append((image.shape[1], list(prompts)))
        if "person" in prompts:
            return {"image": {"w": 960, "h": 540}, "detections": [{"label": "hand", "score": 0.9, "box": [0.45, 0.5, 0.52, 0.7], "polygons": []}], "ms": 1}
        return {"image": {"w": 960, "h": 540}, "detections": [{"label": "tissue pack", "score": 0.5, "box": [0.4, 0.2, 0.6, 0.4], "polygons": []}], "ms": 1}


def test_the_frame_loop_looks_closer_beside_the_frame_call():
    async def scenario():
        sam = Sam()
        session = Session(Services(sam), SHOP)
        jpeg = cv2.imencode(".jpg", np.full((540, 960, 3), 128, np.uint8))[1].tobytes()
        first, _ = await session._ask(jpeg, SHOP.frame_prompts())
        assert len(sam.asked) == 1 and "zoom" not in first  # no hands known yet: the frame only
        second, _ = await session._ask(jpeg, SHOP.frame_prompts())
        assert len(sam.asked) == 3 and sam.asked[2] == (zoom.SIDE, ["tissue pack", "mobile phone", "item held in hand"])
        added = [d for d in second["detections"] if d.get("zoom")]
        assert len(added) == 1 and 0 < added[0]["box"][0] < added[0]["box"][2] < 1
        assert session.zoomed == 1

    asyncio.run(scenario())


def test_books_are_read_from_the_whole_frame_only():
    books = Catalog.from_dict({"products": [{"id": "a", "name": "A", "prompt": "book", "price": 1}, {"id": "b", "name": "B", "prompt": "book", "price": 1}]})
    assert not Session(Services(Sam()), books).zooming


def test_two_hands_close_together_get_one_closer_look():
    assert len(zoom.regions([[0.25, 0.5, 0.35, 0.62], [0.47, 0.48, 0.57, 0.6]])) == 1


def test_a_sharper_frame_allows_a_closer_look():
    hand = [[0.48, 0.55, 0.5, 0.6]]  # a hand far back: tiny in the frame
    (wide,) = zoom.regions(hand, 960)
    (sharp,) = zoom.regions(hand, 1600)
    assert abs((wide[2] - wide[0]) - 300 / 960) < 1e-9  # never fewer than 300 of the frame's own pixels across
    assert sharp[2] - sharp[0] < 0.25 < wide[2] - wide[0]  # closer: down to the room a held item needs round the hand
    assert zoom.narrowest(4000) == zoom.TIGHTEST  # however sharp, not past this


def test_with_no_hands_found_the_closer_look_goes_round_a_far_off_shopper():
    far = [0.45, 0.3, 0.55, 0.7]  # a shopper a few steps back, hands not outlined
    (area,) = zoom.regions([], 1600, [far])
    assert area[0] < far[0] and area[2] > far[2]  # arms held out to the sides are in it
    assert area[1] > far[1] and area[3] < far[3] + 0.1  # head and feet are not the point
    assert zoom.regions([], 1600, [[0.2, 0.0, 0.8, 1.0]]) == []  # close to the camera: no need
    assert zoom.regions([], 1600, []) == []
    two = zoom.regions([], 1600, [[0.1, 0.3, 0.2, 0.7], [0.75, 0.3, 0.85, 0.7], [0.4, 0.5, 0.42, 0.6]])
    assert len(two) == 2  # the two biggest, each on its own


class FarSam(Sam):
    """A whole frame with a shopper far back and no hand; a cut-out with the tissue pack."""

    async def segment(self, jpeg, prompts, threshold=0.3, exemplars=None):
        image = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        self.asked.append((image.shape[1], list(prompts)))
        if "person" in prompts:
            return {"image": {"w": 1600, "h": 900}, "detections": [{"label": "person", "score": 0.9, "box": [0.45, 0.3, 0.55, 0.7], "polygons": []}], "ms": 1}
        return {"image": {"w": 1600, "h": 900}, "detections": [{"label": "tissue pack", "score": 0.5, "box": [0.4, 0.4, 0.6, 0.6], "polygons": []}], "ms": 1}


def test_the_frame_loop_looks_closer_at_a_far_off_shopper():
    async def scenario():
        sam = FarSam()
        session = Session(Services(sam), SHOP)
        jpeg = cv2.imencode(".jpg", np.full((900, 1600, 3), 128, np.uint8))[1].tobytes()
        await session._ask(jpeg, SHOP.frame_prompts())
        second, _ = await session._ask(jpeg, SHOP.frame_prompts())
        assert session.frame_w == 1600 and len(sam.asked) == 3
        assert [d for d in second["detections"] if d.get("zoom")]

    asyncio.run(scenario())

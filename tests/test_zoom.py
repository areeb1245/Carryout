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

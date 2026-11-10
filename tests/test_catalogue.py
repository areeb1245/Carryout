"""The catalogue screen's pieces: example pictures kept through a save, clips cut into frames, phrases ranked by
whether SAM outlines the chosen object, and the likeness check as one vote among the looks."""
import asyncio
import json

import numpy as np

from checkout import clips
from checkout.catalog import Catalog
from checkout.likeness import MARGIN, SURE, Likeness
from checkout.sam_client import ROOT


def shop(**extra) -> dict:
    return {
        "name": "Corner shop",
        "currency": "USD",
        "products": [
            {"id": "tissue", "name": "Tissue rolls, pack of two", "prompt": "tissue roll pack", "price": 4.5, "examples": ["ex-0123456789abcdef.jpg", "../../.env", "ex-ABC.jpg"], **extra},
            {"id": "cream", "name": "Face cream", "prompt": "cream jar", "price": 12.0},
        ],
    }


def test_examples_survive_a_save_and_bad_names_are_dropped():
    catalog = Catalog.from_dict(shop())
    assert catalog.by_id["tissue"].examples == ["ex-0123456789abcdef.jpg"]
    again = Catalog.from_dict(json.loads(json.dumps(catalog.to_dict())))
    assert again.by_id["tissue"].examples == ["ex-0123456789abcdef.jpg"]
    assert "examples" not in again.to_dict()["products"][1]


def test_at_most_four_examples():
    names = [f"ex-{i:016x}.jpg" for i in range(7)]
    catalog = Catalog.from_dict({"products": [{"name": "A", "prompt": "a", "examples": names}]})
    assert catalog.products[0].examples == names[:4]


def test_loading_a_shop_file_remembers_which_shop():
    assert Catalog.load(ROOT / "shops" / "supermarket.json").shop == "supermarket"


def test_a_phrase_wins_by_outlining_the_chosen_object():
    box = [0.4, 0.4, 0.6, 0.6]
    answers = {
        3: {"detections": [
            {"label": "tissue roll", "box": [0.4, 0.4, 0.5, 0.6], "score": 0.8},  # one roll of the pair
            {"label": "tissue roll", "box": [0.5, 0.4, 0.6, 0.6], "score": 0.8},
            {"label": "tissue roll pack", "box": [0.41, 0.4, 0.6, 0.61], "score": 0.6},
        ]},
        4: {"detections": [{"label": "tissue roll pack", "box": [0.43, 0.41, 0.63, 0.62], "score": 0.55}]},
    }
    ranked = clips.rank_phrases(3, box, answers, ["tissue roll", "tissue roll pack", "toilet paper"])
    assert [r["phrase"] for r in ranked][:2] == ["tissue roll pack", "tissue roll"]
    assert ranked[0]["fits"] and ranked[0]["frames"] == 2
    assert not ranked[1]["fits"]  # half the pack is not the pack
    assert ranked[2]["score"] == 0.0


def test_nearby_frames_stay_inside_the_clip():
    assert clips.nearby(0, 10) == [0, 1, 2]
    assert clips.nearby(9, 10) == [7, 8, 9]


def test_a_clip_is_cut_into_frames(tmp_path, monkeypatch):
    import cv2

    monkeypatch.setattr(clips, "CLIPS", tmp_path)
    video = tmp_path / "in.mp4"
    out = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10, (320, 240))
    for i in range(30):
        out.write(np.full((240, 320, 3), i * 8, np.uint8))
    out.release()
    meta = clips.cut(video, "Test", every_s=0.5)
    assert meta["frames"] == 6 and meta["width"] == 320
    assert clips.frame_path(meta["clip"], 5) is not None
    assert clips.frame_path(meta["clip"], 6) is None
    assert clips.frame_path("../" + meta["clip"], 0) is None
    assert clips.info(meta["clip"])["times"][:2] == [0.0, 0.5]


def test_likeness_votes_only_when_clear():
    assert Likeness.verdict([("cream", SURE + 0.1), ("tissue", SURE - 0.2)])[0] == "cream"
    assert Likeness.verdict([("cream", SURE + 0.02), ("tissue", SURE + 0.02 - MARGIN / 2)])[0] is None  # two look alike
    assert Likeness.verdict([("cream", SURE - 0.1)])[0] is None  # like nothing in the catalogue
    assert Likeness.verdict([]) == (None, "")


class FakeSam:
    def __init__(self, vectors):
        self.vectors = vectors
        self.calls = 0

    async def embed(self, pictures):
        self.calls += 1
        return [self.vectors.pop(0) for _ in pictures]


def test_likeness_compares_with_the_pictures(tmp_path, monkeypatch):
    import checkout.likeness as lk

    monkeypatch.setattr(lk, "SHOPS", tmp_path / "shops")
    monkeypatch.setattr(lk, "CACHE", tmp_path / "cache")
    folder = tmp_path / "shops" / "corner" / "examples"
    folder.mkdir(parents=True)
    (folder / "ex-0123456789abcdef.jpg").write_bytes(b"\xff\xd8\xff picture")
    catalog = Catalog.from_dict(shop())
    catalog.shop = "corner"
    sam = FakeSam([[1.0, 0.0], [0.8, 0.6]])
    like = Likeness(sam, catalog)
    asyncio.run(like.prepare())
    assert like.ready and sam.calls == 1
    scores = asyncio.run(like.compare(b"\xff\xd8\xff item"))
    assert scores[0][0] == "tissue" and abs(scores[0][1] - 0.8) < 1e-6
    # the vector is kept: the next visit does not ask for it again
    again = Likeness(FakeSam([]), catalog)
    asyncio.run(again.prepare())
    assert again.ready


def test_without_an_embed_call_likeness_stays_off(tmp_path, monkeypatch):
    import checkout.likeness as lk

    class Old:
        async def embed(self, pictures):
            raise RuntimeError("endpoint answered 404")

    monkeypatch.setattr(lk, "SHOPS", tmp_path / "shops")
    monkeypatch.setattr(lk, "CACHE", tmp_path / "cache")
    folder = tmp_path / "shops" / "corner" / "examples"
    folder.mkdir(parents=True)
    (folder / "ex-0123456789abcdef.jpg").write_bytes(b"\xff\xd8\xff")
    catalog = Catalog.from_dict(shop())
    catalog.shop = "corner"
    like = Likeness(Old(), catalog)
    asyncio.run(like.prepare())
    assert not like.ready and "404" in like.failed


def test_a_camera_capture_adds_frames_one_by_one(tmp_path, monkeypatch):
    import cv2

    monkeypatch.setattr(clips, "CLIPS", tmp_path)
    meta = clips.start_live("Camera")
    jpeg = cv2.imencode(".jpg", np.full((720, 1280, 3), 90, np.uint8))[1].tobytes()
    assert clips.add_frame(meta["clip"], jpeg, 0.0) == 0
    assert clips.add_frame(meta["clip"], jpeg, 0.7) == 1
    assert clips.add_frame(meta["clip"], b"not a picture", 1.4) is None
    after = clips.info(meta["clip"])
    assert after["frames"] == 2 and after["width"] == 960 and after["times"] == [0.0, 0.7]
    # an imported clip is not a camera capture: nothing can be added to it
    (tmp_path / "abcdefabcdef").mkdir()
    (tmp_path / "abcdefabcdef" / "clip.json").write_text(json.dumps({"clip": "abcdefabcdef", "frames": 1, "times": [0]}))
    assert clips.add_frame("abcdefabcdef", jpeg, 0) is None


def test_the_most_different_views_are_kept():
    front, front2, side, back = [1.0, 0.0, 0.0], [0.99, 0.14, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]
    assert clips.diverse([front, front2, side, back], 3, 0) == [0, 2, 3]
    assert clips.diverse([front], 4, 0) == [0]
    assert clips.spread(10, 4) == [0, 3, 6, 9]
    assert clips.spread(3, 4) == [0, 1, 2]


def test_each_camera_view_is_judged_by_its_own_box():
    answers = {
        0: {"detections": [{"label": "cream jar", "box": [0.4, 0.4, 0.6, 0.6], "score": 0.7}]},
        5: {"detections": [{"label": "cream jar", "box": [0.1, 0.1, 0.3, 0.3], "score": 0.6}]},  # moved, and found there
    }
    ranked = clips.rank_phrases(0, [0.4, 0.4, 0.6, 0.6], answers, ["cream jar"], {5: [0.1, 0.1, 0.3, 0.3]})
    assert ranked[0]["frames"] == 2


class Looks:
    """The second model's look at a pick: always the same answer."""

    enabled = True

    def __init__(self, holding):
        self.holding = holding

    async def held_item(self, frame, box, products, ocr_text="", refs=None):
        return {"holding": self.holding, "product": None, "why": "In hand" if self.holding else "Still on the shelf"}


class AlwaysAlike:
    """A likeness that is sure the item is the can: what a pick that never left the shelf would give."""

    ready = wanted = True

    async def compare(self, picture, among=None):
        return [("can", 0.8), ("box", 0.2)]


def _looked_at(tmp_path, holding, looks=3):
    from tests.test_crowd import Shop, play

    async def scenario():
        shop = Shop(tmp_path, verifier=Looks(holding))
        ids = play(shop)
        session = shop.session
        session.likeness, session.likeness_job = AlwaysAlike(), None
        entry = next(e for e in shop.engine.entries if e.person == ids[0] and e.state == "in_cart")
        frame = np.zeros((540, 960, 3), np.uint8)
        for _ in range(looks):
            await session._look(entry.id, entry.item.id, frame, [], [0.1, 0.1, 0.2, 0.2], ids[0])
        return entry, shop.cart(ids[0])["lines"][0]

    return asyncio.run(scenario())


def test_a_likeness_never_outvotes_not_held(tmp_path):
    entry, line = _looked_at(tmp_path, holding=False)
    assert entry.verified == "disputed" and not line["charge"]
    assert line["likeness"]["product"] == "can"  # still shown, as what it resembles
    assert sum(1 for look in entry.looks if look[3]) == 0  # no vote at all when the model saw nothing held


def test_a_likeness_names_an_item_seen_held_and_votes_once(tmp_path):
    entry, line = _looked_at(tmp_path, holding=True)
    assert entry.verified == "confirmed" and line["charge"]
    assert sum(1 for look in entry.looks if look[3]) == 1


def test_the_item_in_a_hand_is_what_touches_the_hand():
    hand = {"label": "hand", "box": [0.52, 0.55, 0.83, 1.0], "score": 0.97}
    bank = {"label": "phone", "box": [0.43, 0.52, 0.63, 0.81], "score": 0.88}  # a power bank, held up: what SAM gave
    shelf = {"label": "box", "box": [0.0, 0.0, 0.2, 0.3], "score": 0.95}  # sure, but nowhere near the hand
    wall = {"label": "device", "box": [0.0, 0.0, 1.0, 1.0], "score": 0.99}  # the whole frame
    assert clips.held([hand, bank, shelf, wall]) is bank
    assert clips.held([bank, shelf]) is None  # no hand, nothing held
    named = {"label": "item held in hand", "box": [0.1, 0.1, 0.2, 0.2], "score": 0.5}
    assert clips.held([named]) is named


# Two items in one shopper's hands: SAM's phrase for one product can outline the other too, and outscore that one's
# own phrase (a power bank's phrase on a tissue pack, 0.36 against 0.30). The looks and the pictures then name it.

WITH_PHONE = Catalog.from_dict({
    "products": [{"id": "can", "name": "Can", "prompt": "drink can", "price": 3.0}, {"id": "box", "name": "Box", "prompt": "cereal box", "price": 10.0}],
    "belongings": ["mobile phone"],
    "zones": ["shelf"],
})


class Names(Looks):
    """The second model's look: held, and always the same product."""

    def __init__(self, product):
        super().__init__(True)
        self.product = product

    async def held_item(self, frame, box, products, ocr_text="", refs=None):
        return {"holding": True, "product": self.product, "why": f"Hand holds the {self.product}"}


class Alike(AlwaysAlike):
    def __init__(self, product):
        self.product = product

    async def compare(self, picture, among=None):
        return [(self.product, 0.8), ("x", 0.1)]


class NoPictures:
    ready = wanted = False


def _named(tmp_path, monkeypatch, seen_as_label, looked, liked, looks=3):
    import tests.test_crowd as crowd
    from tests.test_crowd import Shop, play

    monkeypatch.setattr(crowd, "CATALOG", WITH_PHONE)

    async def scenario():
        shop = Shop(tmp_path, verifier=Names(looked))
        shop.labels[0] = seen_as_label
        ids = play(shop)
        session = shop.session
        session.likeness, session.likeness_job = Alike(liked) if liked else NoPictures(), None
        entry = next(e for e in shop.engine.entries if e.person == ids[0] and e.state == "in_cart")
        frame = np.zeros((540, 960, 3), np.uint8)
        for _ in range(looks):
            await session._look(entry.id, entry.item.id, frame, [], [0.1, 0.1, 0.2, 0.2], ids[0])
        return entry, shop.cart(ids[0])["lines"][0]

    return asyncio.run(scenario())


def test_the_looks_overrule_a_phrase_that_outlined_the_wrong_product(tmp_path, monkeypatch):
    entry, line = _named(tmp_path, monkeypatch, "drink can", looked="box", liked="box")
    assert line["product"] == "box" and line["charge"] and line["price"] == 10.0


def test_a_product_taken_for_a_shoppers_phone_is_billed_once_looked_at(tmp_path, monkeypatch):
    entry, line = _named(tmp_path, monkeypatch, "mobile phone", looked="can", liked="can")
    assert line["product"] == "can" and line["charge"]


def test_a_real_phone_that_only_looks_like_a_product_is_never_billed(tmp_path, monkeypatch):
    entry, line = _named(tmp_path, monkeypatch, "mobile phone", looked="own0", liked="can")
    assert not line["charge"]


def test_the_pictures_alone_never_rename_an_item(tmp_path, monkeypatch):
    # the second model names nothing; the likeness alone says box: it stays a can, held back until looked at again
    entry, line = _named(tmp_path, monkeypatch, "drink can", looked=None, liked="box")
    assert line["product"] == "can"


def test_a_line_the_looks_dispute_the_name_of_is_held_back(tmp_path, monkeypatch):
    # one look says box, nothing else agrees: neither name is sure, so it is not billed as a can on SAM's word
    entry, line = _named(tmp_path, monkeypatch, "drink can", looked="box", liked=None, looks=1)
    assert not line["charge"] and line["contested"] == "Box"
    from checkout.pipeline import Session

    assert Session._why_held(line) == "Seen in hand as another product: Box"


def test_a_phrase_spilling_onto_the_item_beside_its_own_does_not_name_it():
    from checkout.engine import Engine
    from tests.test_crowd import CATALOG, rect

    engine = Engine(CATALOG)
    # the scores SAM gave a power bank (here the box) and a tissue pack (the can) in one shopper's two hands
    dets = [
        rect("cereal box", 0.8, 0.3, 0.12, 0.2, 0.84),
        rect("cereal box", 0.3, 0.3, 0.12, 0.2, 0.36),  # the box's phrase, faintly, on the can
        rect("drink can", 0.3, 0.3, 0.12, 0.2, 0.30),
    ]
    for i in range(3):
        engine.step(i / 3, {"image": {"w": 960, "h": 540}, "detections": dets})
    left_to_right = [i.seen_as for i in sorted(engine.items.values(), key=lambda i: float(i.c[0]))]
    assert left_to_right == ["can", "box"]


def test_two_of_one_product_keep_their_name():
    from checkout.engine import Engine
    from tests.test_crowd import CATALOG, rect

    engine = Engine(CATALOG)
    dets = [rect("cereal box", 0.8, 0.3, 0.12, 0.2, 0.84), rect("cereal box", 0.3, 0.3, 0.12, 0.2, 0.7)]
    for i in range(3):
        engine.step(i / 3, {"image": {"w": 960, "h": 540}, "detections": dets})
    assert sorted(i.seen_as for i in engine.items.values()) == ["box", "box"]

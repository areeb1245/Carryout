"""
A shop where many products look the same to SAM (books): one phrase finds them, reading tells them apart.
"""
import pytest

from checkout.catalog import Catalog
from checkout.engine import Engine
from checkout.reader import Reader
from tests.test_engine import DT, arm, lerp, rect

BOOKS = Catalog.from_dict(
    {
        "products": [
            {"id": "deep-work", "name": "Deep Work", "prompt": "book", "price": 59, "author": "Cal Newport"},
            {"id": "shoe-dog", "name": "Shoe Dog", "prompt": "book", "price": 65, "author": "Phil Knight"},
            {"id": "do-it-today", "name": "Do It Today", "prompt": "book", "price": 49, "author": "Darius Foroux"},
            {"id": "think-straight", "name": "Think Straight", "prompt": "book", "price": 49, "author": "Darius Foroux"},
            {"id": "pen", "name": "Pen", "prompt": "pen", "price": 5},
        ],
        "belongings": ["mobile phone"],
        "zones": ["shelf"],
    }
)


def book(x, y):
    return rect("book", x, y, 0.03, 0.2, 0.8)


class Shelf:
    """Two books standing at x 0.75 and 0.80; a shopper at 0.5 can take the first."""

    def __init__(self):
        self.engine = Engine(BOOKS)
        self.t = 0.0

    def frame(self, *dets):
        flat = [rect("shelf", 0.85, 0.5, 0.3, 0.9, 0.8)]
        for d in dets:
            flat.extend(d if isinstance(d, list) else [d])
        state = self.engine.step(self.t, {"image": {"w": 960, "h": 540}, "detections": flat})
        self.t += DT
        return state

    def settle_in(self):
        for _ in range(8):
            state = self.frame(arm(0.5, 0.56, 0.6), book(0.75, 0.4), book(0.80, 0.4))
        return {round(i["box"][0], 2): i["id"] for i in state["items"] if i["state"] == "rest"}

    def take_first(self):
        steps = [(lerp(0.56, 0.72, k), lerp(0.6, 0.4, k), 0.75, 0.4) for k in (0, 0.5, 1)] + [(0.72, 0.4, 0.75, 0.4)]
        steps += [(lerp(0.72, 0.56, k), lerp(0.4, 0.6, k), lerp(0.75, 0.59, k), lerp(0.4, 0.6, k)) for k in (0, 0.5, 1)]
        steps += [(0.56, 0.6, 0.59, 0.6)] * 4
        for hx, hy, bx, by in steps:
            self.frame(arm(0.5, hx, hy), book(bx, by), book(0.80, 0.4))

    def line(self):
        return self.engine.carts()[0]["lines"][0]


def test_products_sharing_a_phrase_are_asked_for_once():
    assert BOOKS.frame_prompts().count("book") == 1
    assert BOOKS.by_prompt["book"].head and len(BOOKS.members["any:book"]) == 4
    assert not BOOKS.by_prompt["pen"].head
    assert BOOKS.same_kind("any:book", "deep-work") and not BOOKS.same_kind("deep-work", "shoe-dog") and not BOOKS.same_kind("any:book", "pen")


def test_name_from_the_shelf_is_shown_but_billed_only_once_read_in_the_hand():
    s = Shelf()
    ids = s.settle_in()
    first = ids[min(ids)]
    assert s.engine.settle(first, "deep-work", 'Reads "DEEP WORK Cal Newport"')
    assert not s.engine.settle(first, "pen", "nonsense")  # a pen is not a kind of book
    s.take_first()
    line = s.line()
    assert line["name"] == "Deep Work" and not line["charge"] and line["verified"] is None
    assert s.engine.carts()[0]["total"] == 0 and s.engine.carts()[0]["review"] == 1
    # two looks at the shopper's hands read the same title: confirmed, billed
    assert s.engine.name_line(1, "deep-work", "Read in their hands on two looks")
    line = s.line()
    assert line["charge"] and line["verified"] == "confirmed" and s.engine.carts()[0]["total"] == 59


def test_hands_reading_another_title_renames_the_pick():
    """The outline that followed the hand out was a neighbour of the one that had been read."""
    s = Shelf()
    ids = s.settle_in()
    s.engine.settle(ids[min(ids)], "deep-work", 'Reads "DEEP WORK"')
    s.take_first()
    assert s.engine.name_line(1, "shoe-dog", "Read in their hands on two looks")
    cart = s.engine.carts()[0]
    assert [line["name"] for line in cart["lines"]] == ["Shoe Dog"] and cart["total"] == 65


def test_book_nobody_could_read_is_not_charged():
    s = Shelf()
    s.settle_in()
    s.take_first()
    line, cart = s.line(), s.engine.carts()[0]
    assert line["unread"] and not line["charge"]
    assert cart["total"] == 0 and cart["review"] == 1


def test_same_title_is_never_charged_twice():
    s = Shelf()
    s.settle_in()
    s.take_first()
    assert s.engine.name_line(1, "elon" if False else "shoe-dog", "two looks")
    assert not s.engine.name_line(1, "shoe-dog", "two looks again")  # already on the cart
    assert not s.engine.add_seen(1, "shoe-dog", "and again")
    # even if two lines somehow end up with one title, only the first is billed
    assert s.engine.add_seen(1, "deep-work", "two looks")
    twin = s.engine.entries[-1]
    twin.item.settled = "shoe-dog"
    cart = s.engine.carts()[0]
    assert [line["charge"] for line in cart["lines"]] == [True, False] and cart["lines"][1]["repeat"]
    assert cart["total"] == 65 and len(s.engine.unpaid(1)) == 1


def test_title_read_in_the_hands_adds_a_line_the_tracking_missed_and_it_comes_off_when_put_down():
    s = Shelf()
    s.settle_in()
    s.take_first()
    s.engine.name_line(1, "shoe-dog", "two looks")
    assert s.engine.add_seen(1, "deep-work", "Read in their hands on two looks")  # a second book the outlines never showed
    assert s.engine.carts()[0]["total"] == 124
    assert s.engine.drop_title(1, "deep-work", "no longer in their hands")
    cart = s.engine.carts()[0]
    assert cart["total"] == 65 and [line["state"] for line in cart["lines"]] == ["in_cart", "returned"]


def test_stray_outlines_are_folded_once_the_hands_are_fully_read():
    s = Shelf()
    s.settle_in()
    s.take_first()
    assert s.engine.add_seen(1, "shoe-dog", "two looks")  # the reading went to a line of its own
    assert len(s.engine.carts()[0]["lines"]) == 2  # the unnamed pick and the named line
    assert s.engine.void_unnamed(1, s.t) == 1
    cart = s.engine.carts()[0]
    assert [line["name"] for line in cart["lines"]] == ["Shoe Dog"] and cart["review"] == 0


def test_reading_survives_a_flickering_outline():
    s = Shelf()
    ids = s.settle_in()
    first = ids[min(ids)]
    s.engine.settle(first, "deep-work", 'Reads "DEEP WORK"')
    for _ in range(20):  # the first book's outline drops out for over six seconds, nobody near it
        s.frame(arm(0.5, 0.56, 0.6), book(0.80, 0.4))
    for _ in range(6):
        state = s.frame(arm(0.5, 0.56, 0.6), book(0.75, 0.4), book(0.80, 0.4))
    back = min((i for i in state["items"] if i["state"] == "rest"), key=lambda i: i["box"][0])
    assert back["product"] == "deep-work" and not back["unread"]
    assert not [e for e in s.engine.events if e["type"] == "pick"]


@pytest.mark.parametrize(
    "text, title, sure",
    [
        ("DEEPWORK | Cal Newport", "Deep Work", True),
        ("THERICHESTMANINBABYION", None, False),  # not on this shelf
        ("DO IT TODAY | DARNS FOROUX", "Do It Today", True),
        ("DARIUS FOROUX", None, False),  # the author alone: two of his books are on the shelf
        ("SHOE DOG   PHIL KNIGHT", "Shoe Dog", True),
        ("", None, False),
    ],
)
def test_matching_ocr_text_to_titles(text, title, sure):
    members = BOOKS.members["any:book"]
    reading = Reader().match(text, members)
    assert reading.sure == sure
    if sure:
        assert reading.product.name == title


def test_a_title_wrongly_seen_put_back_returns_to_the_cart_when_read_in_the_hands_twice(tmp_path):
    """The outlines can mistake a book held next to the shelf for a book put back. If the second model then reads
    that title in the shopper's hands on two more looks, it is on the cart again, once."""
    import asyncio

    from checkout.pipeline import Services, Session

    class Hands:
        enabled = True

        async def carried(self, frame, box, products):
            return {"products": ["deep-work"], "unreadable": 0, "hands_visible": True, "why": 'read "DEEP WORK"'}

    async def scenario():
        session = Session(Services(None, Hands()), BOOKS)
        engine = session.engine
        shelf = Shelf()
        shelf.engine = engine
        shelf.settle_in()
        pid = engine.snapshot()["people"][0]["id"]

        async def look(t):
            await session._ask_hand(pid, t, None, [0, 0, 1, 1])

        await look(1.0)
        assert not engine.has_title(pid, "deep-work")  # one look names nothing
        await look(2.0)
        assert engine.has_title(pid, "deep-work")
        assert engine.drop_title(pid, "deep-work", "put back on the shelf")  # the outlines say it went back
        assert not engine.has_title(pid, "deep-work") and engine.carts()[0]["total"] == 0
        await look(3.0)
        assert not engine.has_title(pid, "deep-work")  # one reading does not undo a return
        await look(4.0)
        assert engine.has_title(pid, "deep-work") and engine.carts()[0]["total"] == 59
        lines = [(l["name"], l["state"]) for l in engine.carts()[0]["lines"]]
        assert lines == [("Deep Work", "returned"), ("Deep Work", "in_cart")]
        await look(5.0)
        assert engine.carts()[0]["total"] == 59  # and never twice

    asyncio.run(scenario())

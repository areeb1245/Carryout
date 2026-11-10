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


class Looks:
    """A second model whose answers a test sets: what it reads in the shopper's hands on the next look."""

    enabled = True

    def __init__(self):
        self.reads, self.unreadable = ["deep-work"], 0

    async def carried(self, frame, box, products):
        return {"products": list(self.reads), "unreadable": self.unreadable, "hands_visible": True, "why": "what the test says"}


def test_a_book_still_in_the_hand_is_not_taken_for_put_down_when_its_title_stops_showing(tmp_path):
    """A live shopper turns a book over, or lays another on top of it. The looks stop reading its title, but it has
    not gone anywhere: its outline is still in their hand, or the look says there is something there it cannot read."""
    import asyncio

    from checkout.pipeline import Services, Session

    async def scenario():
        hands = Looks()
        session = Session(Services(None, hands), BOOKS)
        engine = session.engine
        shelf = Shelf()
        shelf.engine = engine
        shelf.settle_in()
        shelf.take_first()  # a book in the hand, outline in view
        pid = engine.snapshot()["people"][0]["id"]

        async def look(t):
            await session._ask_hand(pid, t, None, [0, 0, 1, 1])

        await look(4.0)
        await look(5.0)
        assert engine.carts()[0]["total"] == 59  # read twice: Deep Work is on the bill
        hands.reads = []  # turned away: the title is not read any more
        for t in (6.0, 7.0, 8.0, 9.0, 10.0, 11.0):
            await look(t)
        assert engine.has_title(pid, "deep-work") and engine.carts()[0]["total"] == 59  # its outline never left the hand

        engine.entries[0].item.visible = False  # now hidden under another book, which the look cannot read
        hands.unreadable = 1
        for t in (12.0, 13.0, 14.0, 15.0, 16.0):
            await look(t)
        assert engine.has_title(pid, "deep-work")
        hands.unreadable = 0  # hands in view with nothing in them that could be it: it was put down
        for t in (17.0, 18.0, 19.0):
            await look(t)
        assert not engine.has_title(pid, "deep-work") and engine.carts()[0]["total"] == 0

    asyncio.run(scenario())


def test_a_book_taken_in_the_last_seconds_is_read_from_the_last_pictures_at_the_door(tmp_path):
    """Someone takes a book and walks straight out: no look made in time. The pictures kept of them leaving are read
    again at the door, and the title needs two of them, as it always does."""
    import asyncio

    import numpy as np

    from checkout.pipeline import Services, Session

    async def scenario(pictures):
        session = Session(Services(None, Looks()), BOOKS)
        engine = session.engine
        shelf = Shelf()
        shelf.engine = engine
        shelf.settle_in()
        shelf.take_first()
        pid = engine.snapshot()["people"][0]["id"]
        assert engine.unnamed(pid) == 1 and engine.carts()[0]["total"] == 0
        picture = np.zeros((40, 40, 3), np.uint8)
        session.evidence[pid] = {"read": {}, "recent": [], "last": [(5.0 + n, picture, [0, 0, 1, 1], picture) for n in range(pictures)]}
        await session._last_looks(pid)
        return engine.carts()[0], session.evidence[pid]["read"]

    cart, read = asyncio.run(scenario(2))
    assert cart["total"] == 59 and [(l["name"], l["verified"]) for l in cart["lines"]] == [("Deep Work", "confirmed")]
    assert "at exit" in cart["lines"][0]["why"] and "deep-work" in read  # with the picture it was read in
    cart, _ = asyncio.run(scenario(1))
    assert cart["total"] == 0 and cart["lines"][0]["unread"]  # one picture is one look: not enough


def test_a_title_not_in_the_hands_as_they_leave_is_not_charged(tmp_path):
    """Kept, or back on the shelf? The last pictures of a leaving shopper decide for every title on their bill: read
    in their hands, or one of the things in them that could not be read, or gone."""
    import asyncio

    import numpy as np

    from checkout.pipeline import Services, Session

    async def scenario(reads, unreadable, pictures=3):
        hands = Looks()
        session = Session(Services(None, hands), BOOKS)
        engine = session.engine
        shelf = Shelf()
        shelf.engine = engine
        shelf.settle_in()
        pid = engine.snapshot()["people"][0]["id"]
        for t in (1.0, 2.0):  # Deep Work, read twice in their hands, is on the bill
            await session._ask_hand(pid, t, None, [0, 0, 1, 1])
        assert engine.carts()[0]["total"] == 59
        hands.reads, hands.unreadable = reads, unreadable
        picture = np.zeros((40, 40, 3), np.uint8)
        session.evidence[pid]["last"] = [(8.0 + n, picture, [0, 0, 1, 1], picture) for n in range(pictures)]
        await session._last_looks(pid)
        return engine.carts()[0]

    kept = asyncio.run(scenario(["deep-work"], 0))
    assert kept["total"] == 59 and kept["lines"][0]["audit"] is None  # read as they left: stays, for the last check
    gone = asyncio.run(scenario([], 0))  # hands in view, nothing in them: it went back on the shelf
    assert gone["total"] == 0 and gone["lines"][0]["audit"] == "put_back" and "last 3 pictures" in gone["lines"][0]["audit_why"]
    hidden = asyncio.run(scenario([], 1))  # something in their hands that could not be read: the benefit of the doubt
    assert hidden["total"] == 59 and hidden["lines"][0]["audit"] is None
    other = asyncio.run(scenario(["shoe-dog"], 0))  # they left with another book, not this one
    assert [(l["name"], l["charge"]) for l in other["lines"]] == [("Deep Work", False), ("Shoe Dog", True)] and other["total"] == 65
    one = asyncio.run(scenario([], 0, pictures=1))  # a single picture takes nothing off a bill
    assert one["total"] == 59

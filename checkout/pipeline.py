"""
One shop visit, start to finish: frames in, checkout states out, with the side jobs that hang off them.

The frame loop    `submit` queues a frame and asks SAM at once; `next_state` hands back engine states strictly in the
                  order the frames were submitted, because the engine's logic is a timeline and answers can overtake
                  each other on the network.
Beside the loop   Faces       each shopper's face is read locally and matched to the saved profiles. Someone the
                              gate let in (gate.py) has an open visit: their profile and payment method are pulled
                              up. Anyone else is flagged as not registered, and is never charged.
                  Reading     where a shop sells many things that look alike to SAM (books), the text on each one
                              is read locally by OCR, on the shelf and again in the hand, to tell which it is.
                  Second look every pick is shown to a vision language model on up to three frames; together the
                              looks name the product or dispute the pick.
                  Last check  when a shopper leaves, one call shows the second model their bill and the pictures
                              of them carrying things. Line by line it answers kept, put back or unclear.
                  Payment     the lines it answered "kept" for are charged to the saved method in one payment. An
                              invoice is written with the provider and emailed to the address on their profile.

Side jobs never block the frame loop. What they find goes to `outbox` as `update` messages.
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass

import cv2
import numpy as np

from .catalog import Catalog
from .engine import Config, Engine
from .identity import MATCH, Profile, Profiles
from .mailer import Mailer
from .payments import Card, Payments
from .reader import Reader, squash
from .sam_client import SamClient, load_env
from .verify import Verifier

log = logging.getLogger("checkout")
ZONE_EVERY_S = 3.0
LOOKS = 3  # second-model looks per pick
MORE_LOOKS = 7  # for an item that still has no name after those
LOOK_GAP_S = 0.7
LOOK_WINDOW_S = 8.0  # stop waiting for the item to come back into view after this long
SIGHT_GAP_S = 0.7  # how often the text on a held family item is read again
HAND_ASKS = 80  # whole-hand questions per shopper per visit
HAND_ASK_GAP_S = 0.9
HAND_MISSES = 3  # looks in a row without a title before it counts as put down
HAND_GONE_S = 4.0
READ_TRIES = 3  # attempts at reading a shelf item's text
READ_RETRY_S = 2.5
VERIFY_WAIT_S = 10.0
VIEW_GAP_S = 1.0  # how often a carrying shopper's picture is kept for the check before charging
AUDIT_PICTURES = 5


@dataclass
class Services:
    """Shared by every session of the process."""

    sam: SamClient
    verifier: Verifier | None = None
    faces: Profiles | None = None
    payments: Payments | None = None
    mailer: Mailer | None = None
    reader: Reader | None = None
    demo_email: str = ""  # prefilled in the gate's registration form

    def __post_init__(self):
        self.demo_email = self.demo_email or load_env().get("DEMO_EMAIL", "")

    def status(self) -> dict:
        return {
            "verifier": bool(self.verifier and self.verifier.enabled),
            "faces": bool(self.faces and self.faces.ready),
            "face_search": "pgvector" if self.faces and self.faces.index else "local",
            "payments": bool(self.payments and self.payments.enabled),
            "provider": getattr(self.payments, "name", "") if self.payments else "",
            "payments_note": self.payments.problem if self.payments else "payments are not set up",
            "email": bool(self.mailer and self.mailer.enabled),
            "reader": bool(self.reader and self.reader.ready),
            "demo_email": self.demo_email,
        }


class Session:
    def __init__(self, services: Services, catalog: Catalog, outbox: asyncio.Queue | None = None, dry: bool = False, cfg: Config | None = None, threshold: float = 0.3):
        self.id = uuid.uuid4().hex[:12]
        self.services = services
        self.sam = services.sam
        self.catalog = catalog
        self.engine = Engine(catalog, cfg)
        self.threshold = threshold
        self.dry = dry  # a rehearsal: everything runs except the charge, the invoice and the email
        self.outbox = outbox or asyncio.Queue()
        self.queue: asyncio.Queue = asyncio.Queue()
        self.last_zone_t: float | None = None
        self.who: dict[int, Profile] = {}  # person track -> the profile of someone the gate let in
        self.looks: dict[int, int] = {}  # how many times a face was read without matching an open visit
        self.strangers: dict[int, str] = {}  # not registered: "" for an unknown face, else the name of a profile that did not come through the gate
        self.evidence: dict[int, dict] = {}  # per shopper: pictures of them carrying things, for the last check
        self.checking: set[int] = set()  # shoppers whose bill is with the second model right now
        self.shown: dict[int, list[bytes]] = {}  # the pictures the last check was shown, as JPEGs, for the page
        self.jobs: set[asyncio.Task] = set()
        self.checks: dict[int, set[asyncio.Task]] = {}  # second looks still running, by person
        self.payments: dict[int, list[dict]] = {}
        self.watch: dict[int, dict] = {}  # cart lines the second model still owes looks, by entry id
        self.reading: set[int] = set()  # shelf items whose text is being read right now
        self.sighting: set[int] = set()  # held items being read right now
        self.hand_asks: dict[int, tuple[int, float]] = {}  # per shopper: whole-hand questions asked, time of the last
        self.carried: dict[int, dict[str, dict]] = {}  # per shopper and title: looks that read it, looks that did not
        self.hand_seen_t: dict[int, float] = {}
        self.all_read: dict[int, int] = {}
        self.unreadable: dict[int, int] = {}  # per shopper: carried items the last look could not read
        self.asking: dict[int, int] = {}  # whole-hand questions in flight, per shopper
        self.sighted_t: dict[int, float] = {}
        self.naming = asyncio.Semaphore(4)
        self.last_view: tuple | None = None
        self.read_tries: dict[int, tuple[int, float]] = {}  # item id -> (attempts, time of the last one)
        self.frames_seen = 0
        self.closed = False

    # ------------------------------------------------------------------ frame loop

    def submit(self, seq: int, t: float, jpeg: bytes) -> None:
        prompts = self.catalog.frame_prompts()
        zone_due = self.catalog.zones and (self.last_zone_t is None or t - self.last_zone_t >= ZONE_EVERY_S or t < self.last_zone_t)
        if zone_due:
            self.last_zone_t = t
            prompts = prompts + self.catalog.zone_prompts()
        task = asyncio.ensure_future(self._ask(jpeg, prompts))
        self.queue.put_nowait((seq, t, jpeg, bool(zone_due), task))

    @property
    def in_flight(self) -> int:
        return self.queue.qsize()

    def cancel(self) -> None:
        self.closed = True
        while not self.queue.empty():
            self.queue.get_nowait()[-1].cancel()
        for job in self.jobs:
            job.cancel()

    async def _ask(self, jpeg: bytes, prompts: list[str]) -> tuple[dict, float]:
        t0 = time.perf_counter()
        result = await self.sam.segment(jpeg, prompts, self.threshold)
        return result, (time.perf_counter() - t0) * 1000

    async def next_state(self) -> dict:
        """The state for the oldest frame still in flight, or `{seq, t, error}` if the model call failed."""
        seq, t, jpeg, with_zone, task = await self.queue.get()
        try:
            result, round_trip = await task
        except Exception as e:  # this frame is lost; the timeline simply has one sample fewer
            return {"seq": seq, "t": t, "error": str(e) or type(e).__name__}
        frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        state = self.engine.step(t, result, frame)
        self.frames_seen += 1
        first = len(self.engine.events)
        self._faces(frame, state)
        self._keep_views(t, frame, state)
        for event in state["events"]:
            if event["type"] in ("pick", "handover"):
                picked = self.engine.items.get(event["item"])
                if picked is not None and self.catalog.family_of(picked.seen_as) and self.services.verifier and self.services.verifier.enabled:
                    continue  # one of a family (a book): named by the whole-hand looks, not one outline at a time
                self.watch[event["entry"]] = {"person": event["person"], "item": event["item"], "looks": 0, "next_t": t}
            elif event["type"] == "exit":
                self._spawn(self._pay(event["person"]))
        self._second_looks(t, frame, state)
        self._read_hands(t, frame, state)
        self._ask_hands(t, frame, state)
        self._read_shelf(t, frame, state)
        state["events"] = state["events"] + self.engine.events[first:]
        state["seq"] = seq
        state["ms"] = {"model": result.get("ms"), "round_trip": round(round_trip)}
        state["meta"] = self.meta()
        if with_zone:
            state["zone"] = self.engine.zone_outline()
        return state

    # ------------------------------------------------------------------ side jobs

    def _spawn(self, coro) -> asyncio.Task:
        task = asyncio.ensure_future(coro)
        self.jobs.add(task)
        task.add_done_callback(self._done)
        return task

    def _done(self, task: asyncio.Task) -> None:
        self.jobs.discard(task)
        if not task.cancelled() and task.exception():
            log.error("side job failed", exc_info=task.exception())

    def _update(self, first_event: int) -> None:
        """Tell the page what a side job changed."""
        if not self.closed:
            self.outbox.put_nowait({"type": "update", "session": self.id, "t": round(self.engine.t, 3), "events": self.engine.events[first_event:], "carts": self.engine.carts(), "meta": self.meta()})

    def meta(self) -> dict:
        """Who each tracked person is, for the page: their profile, or that they are not registered."""
        book = self.services.faces
        people = {}
        for person in self.engine.people.people.values():
            pid = person.id
            if pid in self.who:
                people[pid] = {"status": "known", "profile": book.summary(self.who[pid])}
            elif pid in self.strangers:
                people[pid] = {"status": "not_registered", "name": self.strangers[pid]}
            else:
                people[pid] = {"status": "no_face"}
            if pid in self.checking:
                people[pid]["checking"] = True
        stock = [i for i in self.engine.items.values() if i.state == "rest" and self.catalog.family_of(i.seen_as)]
        shelf = {"items": len(stock), "read": sum(1 for i in stock if i.settled)} if stock else None
        return {"people": people, "payments": {pid: made[-1] for pid, made in self.payments.items() if made}, "shelf": shelf, "dry": self.dry}

    # faces: who came through the gate

    def _faces(self, frame: np.ndarray, state: dict) -> None:
        book = self.services.faces
        if book is None or not book.ready:
            return
        refresh = self.frames_seen % 8 == 0  # now and then, look again at known shoppers to learn another angle
        wanted = [p for p in state["people"] if p["id"] not in self.who or refresh]
        if not wanted:
            return
        faces = book.faces_in(frame)
        taken = {profile.id for profile in self.who.values()}
        for person in wanted:
            x0, y0, x1, y1 = person["box"]
            head = [f for f in faces if x0 <= f[1] <= x1 and y0 <= f[2] <= y0 + 0.45 * (y1 - y0)]
            if not head:
                continue
            face = book.read(frame, max(head, key=lambda f: f[0][2])[0])
            pid = person["id"]
            known = self.who.get(pid)
            if known:
                if not known.templates or MATCH <= book.score(known, face) < 0.7:
                    book.learn(known, face)  # a new angle
                continue
            match = book.who(face, skip=taken)
            if match and book.open_visit(match[0]) is not None:
                taken.add(match[0].id)
                self.strangers.pop(pid, None)
                self._link(pid, match[0], match[1])
                continue
            # No open visit: an unknown face, or a profile that has not come through the gate this time. Two looks
            # before saying so, as one poor view of a registered face can miss.
            self.looks[pid] = self.looks.get(pid, 0) + 1
            name = match[0].name if match else ""
            if self.looks[pid] >= 2 and self.strangers.get(pid) != name:
                first_time = pid not in self.strangers
                self.strangers[pid] = name
                if first_time:
                    self.engine.note("unregistered", person=pid, name=name)

    def _link(self, pid: int, profile: Profile, score: float | None = None) -> None:
        self.who[pid] = profile
        extra = {} if score is None else {"score": round(score, 2)}
        method = self.services.faces.summary(profile)["method"]
        self.engine.note("checkin", person=pid, name=profile.name, how="recognised", method=method, **extra)

    # pictures for the last check

    @staticmethod
    def _crop(frame: np.ndarray, box: list[float], grow: float = 0.0) -> np.ndarray:
        h, w = frame.shape[:2]
        bw, bh = box[2] - box[0], box[3] - box[1]
        x0, y0 = int(max(0.0, box[0] - grow * bw) * w), int(max(0.0, box[1] - grow * bh) * h)
        x1, y1 = int(min(1.0, box[2] + grow * bw) * w), int(min(1.0, box[3] + grow * bh) * h)
        return frame[y0:max(y1, y0 + 2), x0:max(x1, x0 + 2)].copy()

    def _carrying_view(self, pid: int, frame: np.ndarray, state: dict, person_box: list[float]) -> np.ndarray:
        """A picture of what this shopper has in their hands: their hands and the items in them with room around,
        or the whole shopper when no hand is outlined in this frame."""
        boxes = [h["box"] for h in state["hands"] if h.get("person") == pid]
        boxes += [i["box"] for i in state["items"] if i["state"] == "held" and i.get("holder") == pid]
        if not boxes:
            return self._crop(frame, person_box, 0.05)
        x0, y0 = min(b[0] for b in boxes), min(b[1] for b in boxes)
        x1, y1 = max(b[2] for b in boxes), max(b[3] for b in boxes)
        # never tighter than a good part of the shopper: the model should see whose hands these are
        pw, ph = person_box[2] - person_box[0], person_box[3] - person_box[1]
        need_w, need_h = max(0.0, 0.6 * pw - (x1 - x0)) / 2, max(0.0, 0.45 * ph - (y1 - y0)) / 2
        return self._crop(frame, [x0 - need_w, y0 - need_h, x1 + need_w, y1 + need_h], 0.35)

    def _keep_views(self, t: float, frame: np.ndarray, state: dict) -> None:
        """Keep the latest couple of pictures of every shopper who has something on their bill: the last check
        wants to see what they still had on the way out."""
        verifier = self.services.verifier
        if verifier is None or not verifier.enabled:
            return
        for person in state["people"]:
            pid = person["id"]
            memo = self.evidence.setdefault(pid, {"read": {}, "recent": []})
            if memo["recent"] and 0 <= t - memo["recent"][-1][0] < VIEW_GAP_S:
                continue
            if not (self.engine.holding(pid) or self.engine.unpaid(pid)):
                continue
            memo["recent"].append((t, self._carrying_view(pid, frame, state, person["box"])))
            del memo["recent"][:-2]
            memo["scene"] = (t, frame)  # and the whole picture, shelves included, the last time they were seen

    def _pictures(self, pid: int, entries: list) -> list[tuple[float, np.ndarray]]:
        """What the last check is shown, oldest first: for each line the picture in which that item was read in
        their hands, then the latest close view of their hands, then the whole scene the last time they were seen."""
        memo = self.evidence.get(pid)
        if not memo:
            return []
        chosen = {}
        for entry in entries:
            hit = memo["read"].get(entry.item.cls)
            if hit:
                chosen[hit[0]] = hit[1]
        ending = [view for view in memo["recent"] if view[0] not in chosen][-1:]
        if memo.get("scene"):
            ending.append(memo["scene"])
        read = sorted(chosen.items())[-(AUDIT_PICTURES - len(ending)):]
        return read + ending

    # second look
    # second look

    def _could_be(self, item) -> tuple[list, list]:
        """(the products an item could be, the family members among them). SAM saw it as a single product or as a
        family ("a book"); for a family only its members, plus shoppers' own things, are candidates."""
        own = [p for p in self.catalog.products if not p.billable]
        members = self.catalog.members.get(self.catalog.family_of(item.seen_as), [])
        return (members + own if members else list(self.catalog.products)), members

    def _second_looks(self, t: float, frame: np.ndarray, state: dict) -> None:
        """Up to three looks at each new pick, on frames a little apart, while the item is in view in the hand."""
        verifier, reader = self.services.verifier, self.services.reader
        if not (verifier and verifier.enabled) and not (reader and reader.ready):
            return
        in_view = {i["id"]: i for i in state["items"] if i["state"] == "held"}
        for entry_id, job in list(self.watch.items()):
            item = in_view.get(job["item"])
            # three quick looks for every pick; a few more, further apart, while the item still has no name
            limit = MORE_LOOKS if item and item.get("unread") else LOOKS
            if job["looks"] >= limit or t - job["next_t"] > LOOK_WINDOW_S:
                del self.watch[entry_id]
            elif item and t >= job["next_t"]:
                job["looks"] += 1
                job["next_t"] = t + (LOOK_GAP_S if job["looks"] < LOOKS else 2 * LOOK_GAP_S)
                task = self._spawn(self._look(entry_id, job["item"], frame, item["polygons"], item["box"], job["person"], t))
                pending = self.checks.setdefault(job["person"], set())
                pending.add(task)
                task.add_done_callback(pending.discard)

    async def _look(self, entry_id: int, item_id: int, frame: np.ndarray, polygons: list, box: list[float], pid: int | None = None, t: float = 0.0) -> None:
        verifier, reader = self.services.verifier, self.services.reader
        item = self.engine.items.get(item_id)
        if item is None:
            return
        candidates, members = self._could_be(item)
        reading = None
        if members and reader and reader.ready:
            reading = await asyncio.to_thread(reader.read, frame, polygons, box, members)
        verdict = await verifier.held_item(frame, box, candidates, reading.text if reading else "") if verifier and verifier.enabled else None
        first = len(self.engine.events)
        if verdict is not None:
            agree = bool(reading and reading.product and reading.score >= 0.6 and verdict["product"] == reading.product.id)
            why = f'Reads "{reading.quote()}", and the second model agrees' if agree else verdict["why"]
            self.engine.apply_look(entry_id, verdict["holding"], verdict["product"], why, weight=2 if agree else 1)
            if verdict["holding"] and verdict["product"] and pid is not None:
                # a picture in which this item was seen in their hand, for the check before charging
                self.evidence.setdefault(pid, {"read": {}, "recent": []})["read"][verdict["product"]] = (t, self._crop(frame, box, 1.5))
        elif reading and reading.sure:
            self.engine.apply_look(entry_id, True, reading.product.id, f'Reads "{reading.quote()}"', weight=2)
        self._update(first)

    def _read_hands(self, t: float, frame: np.ndarray, state: dict) -> None:
        """Keep reading what family items in shoppers' hands say, about once a second each. A clear reading names the
        item, and tells the engine when a track has slipped from one item to another."""
        reader, verifier = self.services.reader, self.services.verifier
        if not self.catalog.members or reader is None or not reader.ready:
            return
        if verifier and verifier.enabled:
            return  # the whole-hand looks do this better; the OCR alone is the fallback without a second model
        for shown in state["items"]:
            if shown["state"] != "held" or len(self.sighting) >= 4:
                continue
            item = self.engine.items.get(shown["id"])
            if item is None or not self.catalog.family_of(item.seen_as) or shown["id"] in self.sighting:
                continue
            if t - self.sighted_t.get(shown["id"], -9.0) < SIGHT_GAP_S:
                continue
            self.sighted_t[shown["id"]] = t
            self.sighting.add(shown["id"])
            self._spawn(self._sight(shown["id"], frame, shown["polygons"], shown["box"]))

    async def _sight(self, item_id: int, frame: np.ndarray, polygons: list, box: list[float]) -> None:
        try:
            item = self.engine.items.get(item_id)
            if item is None:
                return
            reading = await asyncio.to_thread(self.services.reader.read, frame, polygons, box, self._could_be(item)[1])
            if reading.sure:
                first = len(self.engine.events)
                self.engine.name_held(item_id, reading.product.id, f'Reads "{reading.quote()}" in the hand')
                if len(self.engine.events) > first:
                    self._update(first)
        finally:
            self.sighting.discard(item_id)

    def _ask_hands(self, t: float, frame: np.ndarray, state: dict) -> None:
        """For shops with look-alike products (books): once the frame loop is sure a shopper has taken something,
        the second model is shown that shopper every second or two and asked which titles they are carrying.

        Their cart follows those answers. A title read on two looks goes on the cart: onto a pick that has no name
        yet, or as a line of its own if the frame loop missed that pick. A title that has dropped out of three looks
        in a row, with the hands in view, comes off again: it was put down. Following single outlines through a
        stack of books in a moving hand is fragile; what is legible in the hands, look after look, is not."""
        verifier = self.services.verifier
        if not self.catalog.members or verifier is None or not verifier.enabled:
            return
        for person in state["people"]:
            pid = person["id"]
            asked, last = self.hand_asks.get(pid, (0, -9.0))
            # twice as often, two at a time, while a pick of theirs still has no confirmed title
            eager = self.engine.unnamed(pid) > 0
            if asked >= HAND_ASKS or t - last < (HAND_ASK_GAP_S / 2 if eager else HAND_ASK_GAP_S) or self.asking.get(pid, 0) >= (2 if eager else 1):
                continue
            carrying = any(v["on_cart"] for v in self.carried.get(pid, {}).values())
            if not (self.engine.holding(pid) or carrying or self.engine.unnamed(pid)):
                continue  # nothing taken yet: nothing to ask about
            self.hand_asks[pid] = (asked + 1, t)
            self.asking[pid] = self.asking.get(pid, 0) + 1
            task = self._spawn(self._ask_hand(pid, t, frame, person["box"], self._carrying_view(pid, frame, state, person["box"])))
            pending = self.checks.setdefault(pid, set())
            pending.add(task)
            task.add_done_callback(pending.discard)

    async def _ask_hand(self, pid: int, t: float, frame: np.ndarray, box: list[float], view: np.ndarray | None = None) -> None:
        try:
            books = [p for group in self.catalog.members.values() for p in group]
            answer = await self.services.verifier.carried(frame, box, books)
            if not answer:
                return
            in_order = t >= self.hand_seen_t.get(pid, -1.0)  # answers can overtake each other
            self.hand_seen_t[pid] = max(t, self.hand_seen_t.get(pid, -1.0))
            engine, first = self.engine, len(self.engine.events)
            seen = self.carried.setdefault(pid, {})
            kept = self.evidence.setdefault(pid, {"read": {}, "recent": []})["read"]
            for product in answer["products"]:
                if view is not None and t >= kept.get(product, (-1.0,))[0]:
                    kept[product] = (t, view)  # the latest picture in which this title was read in their hands
                memo = seen.setdefault(product, {"votes": 0, "misses": 0, "last_t": t, "on_cart": False})
                if memo["on_cart"] and not engine.has_title(pid, product):
                    # The outlines said it went back on the shelf, yet here it is, read in their hands. One look
                    # is not enough to undo that: it comes back onto the cart only if a second look reads it too.
                    memo.update(votes=0, on_cart=False)
                memo["votes"], memo["misses"], memo["last_t"] = memo["votes"] + 1, 0, max(t, memo["last_t"])
                if memo["votes"] >= 2 and not memo["on_cart"]:
                    why = f"Read in their hands on two looks: {answer['why']}"
                    memo["on_cart"] = engine.name_line(pid, product, why) or engine.add_seen(pid, product, why) or True
            if answer["hands_visible"] and in_order:
                for product, memo in seen.items():
                    if product in answer["products"]:
                        continue
                    memo["misses"] += 1
                    if memo["on_cart"] and memo["misses"] >= HAND_MISSES and t - memo["last_t"] >= HAND_GONE_S:
                        engine.drop_title(pid, product, "no longer in their hands")
                        memo.update(votes=0, misses=0, on_cart=False)
                    elif not memo["on_cart"] and memo["misses"] >= HAND_MISSES:
                        memo["votes"] = 0  # a one-off reading that never came back
                # with nothing unreadable in the hands on two looks running, picks still without a name are stray outlines
                self.unreadable[pid] = answer["unreadable"]
                self.all_read[pid] = self.all_read.get(pid, 0) + 1 if answer["unreadable"] == 0 and answer["products"] else 0
                if self.all_read[pid] >= 2:
                    engine.void_unnamed(pid, t - 2.0)
            if len(engine.events) > first:
                self._update(first)
        finally:
            self.asking[pid] = max(0, self.asking.get(pid, 0) - 1)

    # reading the stock

    def _read_shelf(self, t: float, frame: np.ndarray, state: dict) -> None:
        """Work out which family member each resting item is, before anyone picks it up. Items not tried yet go
        first, so one unreadable spine cannot hold up the rest."""
        reader = self.services.reader
        if not self.catalog.members or reader is None or not reader.ready:
            return
        self.last_view = (t, frame, state)
        unread = [i for i in state["items"] if i["state"] == "rest" and i.get("unread") and i["id"] not in self.reading]
        unread.sort(key=lambda i: self.read_tries.get(i["id"], (0, 0.0))[0])
        for shown in unread:
            if len(self.reading) >= 12:
                break
            tries, last = self.read_tries.get(shown["id"], (0, -99.0))
            if tries >= READ_TRIES or t - last < READ_RETRY_S:
                continue
            self.read_tries[shown["id"]] = (tries + 1, t)
            self.reading.add(shown["id"])
            self._spawn(self._read_item(shown["id"], frame, shown["polygons"], shown["box"]))

    async def read_shelf_now(self, rounds: int = READ_TRIES) -> None:
        """For footage that is run faster than real time: keep reading the last frame until every resting item has
        had its tries. A live camera never needs this, the frames keep coming."""
        for k in range(rounds):
            if self.last_view is None:
                return
            t, frame = self.last_view[:2]
            self._read_shelf(t + READ_RETRY_S * (1 + k), frame, self.engine.snapshot())
            while self.jobs:
                await asyncio.wait(list(self.jobs))

    async def _read_item(self, item_id: int, frame: np.ndarray, polygons: list, box: list[float]) -> None:
        reader, verifier = self.services.reader, self.services.verifier
        try:
            item = self.engine.items.get(item_id)
            if item is None or item.settled:
                return
            candidates, members = self._could_be(item)
            reading = await asyncio.to_thread(reader.read, frame, polygons, box, members)
            first = len(self.engine.events)
            if reading.sure:
                self.engine.settle(item_id, reading.product.id, f'Reads "{reading.quote()}"')
            elif verifier and verifier.enabled:
                async with self.naming:  # a few at a time: each answer takes a couple of seconds
                    answer = await verifier.name_item(frame, box, members, reading.text)
                named = self.catalog.by_id.get(answer["product"]) if answer and answer["product"] else None
                if named is not None:
                    # the model's answer stands when the OCR text leans the same way, or when there was no usable
                    # OCR text at all (stacked or decorative lettering) and the model quotes what it read
                    # The model's answer stands when the OCR text leans the same way. With no OCR text at all (stacked
                    # or decorative lettering) it stands only if the item is big enough in the frame to be read.
                    fh, fw = frame.shape[:2]
                    legible = min((box[2] - box[0]) * fw, (box[3] - box[1]) * fh) >= 22
                    backed = reader.support(reading.text, named) >= 0.5 or (len(squash(reading.text)) < 4 and legible)
                    # And never for a title already settled on another item in view: a model shown a tight row of
                    # spines sometimes reads the neighbour's.
                    taken = any(o.settled == named.id and o.state != "gone" for o in self.engine.items.values() if o.id != item_id)
                    if backed and not taken:
                        self.engine.settle(item_id, named.id, f"Second model read it: {answer['why']}", sure=False)
            if len(self.engine.events) > first:
                self._update(first)
        finally:
            self.reading.discard(item_id)

    # the last check, then payment

    async def _audit(self, pid: int) -> dict:
        """One call to the second model before any money moves: is each line on this bill something the shopper
        really left with? Lines it does not answer "kept" for come off the charge. Returns what the page shows:
        {status: checked | failed | skipped | nothing, kept, held}."""
        engine, verifier = self.engine, self.services.verifier
        entries = engine.unpaid(pid)
        if not entries:
            return {"status": "nothing", "kept": 0, "held": 0}
        if verifier is None or not verifier.enabled:
            return {"status": "skipped", "kept": len(entries), "held": 0}
        self.checking.add(pid)
        self._update(len(engine.events))
        try:
            pictures = self._pictures(pid, entries)
            self.shown[pid] = [Verifier.lettered(image, chr(65 + i)) for i, (_, image) in enumerate(pictures)]
            asked = []
            for n, entry in enumerate(entries, 1):
                product = self.catalog.by_id[entry.item.cls]
                asked.append({"line": n, "name": product.name, **({"by": product.author} if product.author else {}), "looks_like": product.cues or product.prompt})
            answer = None
            for _ in range(2 if pictures else 0):
                answer = await verifier.audit(asked, pictures)
                if answer is not None:
                    break
            first = len(engine.events)
            missing = "the check before charging got no answer" if pictures else "no picture of them carrying it was kept"
            for n, entry in enumerate(entries, 1):
                verdict = answer[n] if answer else {"verdict": "unclear", "why": missing}
                engine.apply_audit(entry.id, verdict["verdict"], verdict["why"])
            kept = sum(1 for e in entries if e.audit == "kept")
            self._update(first)
            return {"status": "checked" if answer else "failed", "kept": kept, "held": len(entries) - kept, "pictures": len(pictures), "session": self.id, "person": pid}
        finally:
            self.checking.discard(pid)

    async def _pay(self, pid: int) -> None:
        # do not bill a line the second model is still looking at
        pending = list(self.checks.get(pid, ()))
        if pending:
            await asyncio.wait(pending, timeout=VERIFY_WAIT_S)
        engine, pay, book = self.engine, self.services.payments, self.services.faces
        first = len(engine.events)
        if self.hand_asks.get(pid, (0, 0.0))[0] >= 3:
            # Their hands were read several times. A pick that never got a confirmed title from those readings is a
            # stray outline, unless the last reading said there was something it could not read.
            engine.void_unnamed(pid, engine.t, keep=self.unreadable.get(pid, 0))
        shopper = self.who.get(pid)
        # someone who is not registered is never charged, so there is nothing to check for them
        check = await self._audit(pid) if shopper else {"status": "skipped", "kept": 0, "held": 0}
        entries = engine.unpaid(pid)
        lines = [engine.catalog.by_id[e.item.cls] for e in entries]
        amount = round(sum(p.price for p in lines), 2)
        billed = [(p.id, p.name, p.price) for p in lines]
        detail = [(p.name, p.price, " · ".join(x for x in (p.category, p.section) if x)) for p in lines]
        base = {"amount": amount, "currency": self.catalog.currency.lower(), "items": len(lines), "id": None, "receipt_url": None, "message": "", "check": check}
        if amount <= 0:
            payment = {**base, "status": "nothing_due"}
        elif shopper is None:
            payment = {**base, "status": "not_registered", "message": "Not registered at the gate, so nothing was charged."}
        elif self.dry:
            payment = {**base, "status": "dry", "message": "Rehearsal: nothing was charged and no email was sent."}
        elif not book.can_pay(shopper):
            payment = {**base, "status": "no_card", "message": "Left with unpaid items and no payment method on file."}
        elif pay is None or not pay.enabled:
            payment = {**base, "status": "off", "message": pay.problem if pay else "Payments are not set up."}
        else:
            visit = len(self.payments.get(pid, []))
            description = "Walk-out checkout: " + ", ".join(p.name for p in lines)
            card = Card(shopper.customer, shopper.payment_method, shopper.brand or "", shopper.last4 or "")
            extra = {"lines": detail} if getattr(pay, "name", "") == "paypal" else {}
            result = await pay.charge(card, amount, self.catalog.currency, description[:200], f"{self.id}-{pid}-{visit}", {"app": "sam3-supermarket", "session": self.id, "shopper": shopper.id}, **extra)
            payment = {**base, **result}
            if result["status"] == "succeeded":
                engine.mark_paid(entries, result["id"])
        if shopper:
            payment.update(brand=shopper.brand, last4=shopper.last4, name=shopper.name, method=book.method_name(shopper.brand, shopper.last4, shopper.payer) if book.can_pay(shopper) else "")
        self.payments.setdefault(pid, []).append(payment)
        engine.note("payment", person=pid, payment=payment["id"], **{k: payment.get(k) for k in ("status", "amount", "currency", "items", "method", "message")})
        self._update(first)
        if payment["status"] == "succeeded" and callable(getattr(pay, "invoice", None)):
            first = len(engine.events)
            made = await pay.invoice(shopper.name, shopper.email or "", detail, self.catalog.currency, payment["id"], self.catalog.name, f"{self.id}-{pid}-{len(self.payments[pid])}")
            payment["invoice"] = made
            payment["receipt_url"] = made.get("url") or payment["receipt_url"]
            engine.note("invoice", person=pid, status=made["status"], number=made.get("number"), url=made.get("url"), message=made.get("message", ""))
            self._update(first)
        if shopper and not self.dry and payment["status"] != "off":  # a rehearsal, or payments not set up: nothing happened worth keeping
            book.record_visit(shopper, self.id, payment, billed)
            book.close_visit(shopper, self.id)  # they have left: the next visit starts at the gate
        mailer = self.services.mailer
        if payment["status"] == "succeeded" and shopper and shopper.email and mailer:
            first = len(engine.events)
            sent = await mailer.receipt(shopper.email, shopper.name, detail, payment, f"receipt-{payment['id']}")
            payment["email"] = sent
            engine.note("email", person=pid, status=sent["status"], to=sent["to"], message=sent["message"])
            self._update(first)

    async def finish(self) -> None:
        """The footage is over: whoever is still inside walks out now, and every open job is seen through."""
        first = len(self.engine.events)
        for pid in self.engine.close():
            self._spawn(self._pay(pid))
        self._update(first)
        while self.jobs:
            await asyncio.wait(list(self.jobs))
        if not self.closed:
            self.outbox.put_nowait({"type": "finished", "session": self.id, "t": round(self.engine.t, 3), "events": [], "carts": self.engine.carts(), "meta": self.meta()})

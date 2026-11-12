"""
One shop visit, start to finish: frames in, checkout states out, with the side jobs that hang off them.

The frame loop    `submit` queues a frame and asks SAM at once; `next_state` hands back engine states strictly in the
                  order the frames were submitted, because the engine's logic is a timeline and answers can overtake
                  each other on the network.
Beside the loop   Faces       each shopper's face is read locally and matched to the saved profiles. Someone the
                              gate let in (gate.py) has an open visit: their profile and payment method are pulled
                              up. A registered shopper who skipped the gate is checked in on the floor instead,
                              on two sure looks. Anyone else is flagged as not registered, and is never charged.
                  Reading     where a shop sells many things that look alike to SAM (books), the text on each one
                              is read locally by OCR, on the shelf and again in the hand, to tell which it is.
                  Second look every pick is shown to a vision language model on up to three frames; together the
                              looks name the product or dispute the pick. Where products have example
                              pictures, each look also compares the item with them (likeness.py): a clear
                              likeness is one more vote, and the model is shown the pictures too.
                  Last check  when a shopper leaves, one call shows the second model their bill and the pictures
                              of them carrying things. Line by line it answers kept, put back or unclear.
                  Payment     the lines it answered "kept" for are charged to the saved method in one payment. An
                              invoice is written with the provider and emailed to the address on their profile.
                  Ledger      the walk-out is written down with its steps, its pictures and its numbers (ledger.py),
                              so the bill can be explained afterwards and refunded if it was wrong.

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

from . import zoom
from .catalog import HELD, Catalog
from .engine import Config, Engine
from .identity import SURE, Profile, Profiles
from .ledger import Ledger
from .likeness import Likeness
from .likeness import crop as cut_out
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
HAND_ASKS = 120  # whole-hand questions per shopper per visit (about a minute of carrying, looked at twice a second)
HAND_ASK_GAP_S = 0.9
EAGER_GAP_S = 0.25  # while eager: every frame a live camera sends (about three a second)
HAND_MISSES = 3  # looks in a row without a title before it counts as put down
HAND_GONE_S = 4.0
PAIR_S = 6.0  # two readings of a title further apart than this are not two readings of one book
VOTE_LAPSE_S = 2.0  # a single reading of a title not on the cart lapses after three looks without it, and this long
IN_FLIGHT = 2  # whole-hand looks at once per shopper
EAGER_IN_FLIGHT = 3  # whole-hand looks at once per shopper while eager
HAND_BACKLOG = 8  # frames queued for a look while those are out, oldest dropped first
READ_TRIES = 3  # attempts at reading a shelf item's text
READ_RETRY_S = 2.5
VERIFY_WAIT_S = 10.0
VIEW_GAP_S = 1.0  # how often a carrying shopper's picture is kept for the check before charging
AUDIT_PICTURES = 5
LAST_LOOKS = 6  # pictures of a leaving shopper that are read once more at the door
LAST_GAP_S = 0.45  # how far apart those are taken
LINK_S = 6.0  # a product not seen with the shopper in their last this-many seconds in view is not linked to them any more
PROOF_SIDE = 896  # longer side of the pictures kept in the ledger


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
    def __init__(self, services: Services, catalog: Catalog, outbox: asyncio.Queue | None = None, dry: bool = False, cfg: Config | None = None, threshold: float = 0.3, live: bool = False):
        self.id = uuid.uuid4().hex[:12]
        self.services = services
        self.sam = services.sam
        self.catalog = catalog
        self.engine = Engine(catalog, cfg or Config(exit_after_s=catalog.leave_after_s))
        self.threshold = threshold
        self.dry = dry  # a rehearsal: everything runs except the charge, the invoice and the email
        self.live = live  # a camera, not a clip: the footage clock is the wall clock
        self.wall0: float | None = None  # the time of day at second 0 of the footage
        self.started = time.time()  # when this run began, wall clock
        self.model_ms = 0.0  # SAM's own time over all frames, for the ledger
        self.pick_shots: dict[int, tuple[float, bytes]] = {}  # cart line -> the frame in which it left the shelf, as a JPEG
        self.shown_notes: dict[int, list[tuple[float, str]]] = {}  # what each picture in `shown` is, in words
        self.matches: dict[int, dict[str, tuple[int, float]]] = {}  # per person track and profile: looks that matched, best score
        self.scores: dict[int, float] = {}  # how well each linked shopper's face matched
        self.stranger_ids: dict[int, str] = {}  # the profile a not-registered person looks like
        self.face_shots: dict[int, tuple[float, bytes]] = {}  # the floor camera's view of a shopper when it recognised them
        self.outbox = outbox or asyncio.Queue()
        self.queue: asyncio.Queue = asyncio.Queue()
        self.last_zone_t: float | None = None
        self.who: dict[int, Profile] = {}  # person track -> the profile of someone the gate let in, or the floor checked in
        self.walk_ins: set[int] = set()  # person tracks checked in on the shop floor, not at the gate
        self.looks: dict[int, int] = {}  # how many times a face was read without matching an open visit
        self.strangers: dict[int, str] = {}  # not charged: "" for an unknown face, else the registered shopper it resembles too weakly to be sure
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
        self.hand_backlog: dict[int, list] = {}  # frames waiting for a whole-hand look, per shopper, oldest first
        self.doubt: dict[int, dict] = {}  # per shopper: titles the door look could not tell apart, for the final check
        self.sighted_t: dict[int, float] = {}
        self.naming = asyncio.Semaphore(4)
        self.likeness = Likeness(self.sam, catalog)
        self.likeness_job: asyncio.Task | None = None
        self.liked: set[int] = set()  # cart lines the likeness has voted on (once each)
        self.last_view: tuple | None = None
        self.read_tries: dict[int, tuple[int, float]] = {}  # item id -> (attempts, time of the last one)
        self.frames_seen = 0
        self.closed = False
        # a closer look round the hands while they are small in the frame (zoom.py); not where products are told
        # apart by reading them (books), whose path reads the whole frame and is rehearsed as it is
        self.zooming = not catalog.members and any(p.billable for p in catalog.products)
        self.zoom_prompts = [*catalog.by_prompt, HELD]
        self.hand_boxes: list[list[float]] = []  # the hands in the latest answer, for where to look closer
        self.people_boxes: list[list[float]] = []  # and the shoppers, for when no hand was found
        self.frame_w = 960  # how wide the frames come, in pixels: how close a look their pixels allow
        self.zoomed = 0  # frames that had a closer look

    # ------------------------------------------------------------------ frame loop

    def submit(self, seq: int, t: float, jpeg: bytes) -> None:
        prompts = self.catalog.frame_prompts()
        zone_due = self.catalog.zones and (self.last_zone_t is None or t - self.last_zone_t >= ZONE_EVERY_S or t < self.last_zone_t)
        if zone_due:
            self.last_zone_t = t
            prompts = prompts + self.catalog.zone_prompts()
        if self.wall0 is None:
            self.wall0 = time.time() - t
        if self.likeness_job is None and self.likeness.wanted:
            self.likeness_job = self._spawn(self.likeness.prepare())
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
        areas = zoom.regions(self.hand_boxes, self.frame_w, self.people_boxes) if self.zooming else []
        calls = [self.sam.segment(jpeg, prompts, self.threshold)]
        if areas:
            frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
            calls += [self.sam.segment(zoom.cut(frame, area), self.zoom_prompts, self.threshold) for area in areas]
        answers = await asyncio.gather(*calls, return_exceptions=True)
        result = answers[0]
        if isinstance(result, BaseException):
            raise result
        self.hand_boxes, self.people_boxes = zoom.hands_in(result), zoom.people_in(result)
        self.frame_w = int((result.get("image") or {}).get("w") or self.frame_w)
        closer = [zoom.back(near["detections"], area) for area, near in zip(areas, answers[1:]) if isinstance(near, dict)]
        if closer:  # a copy: the client may keep the frame's own answer in its cache
            result = {**result, "detections": result["detections"] + [d for found in closer for d in found], "zoom": areas}
            self.zoomed += 1
        return result, (time.perf_counter() - t0) * 1000

    async def next_state(self) -> dict:
        """The state for the oldest frame still in flight, or `{seq, t, error}` if the model call failed."""
        seq, t, jpeg, with_zone, task = await self.queue.get()
        try:
            result, round_trip = await task
        except Exception as e:  # this frame is lost; the timeline simply has one sample fewer
            return {"seq": seq, "t": t, "error": str(e) or type(e).__name__}
        work0 = time.perf_counter()
        frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        state = self.engine.step(t, result, frame)
        self.frames_seen += 1
        self.model_ms += float(result.get("ms") or 0.0)
        first = len(self.engine.events)
        self._faces(frame, state)
        self._keep_views(t, frame, state, jpeg)
        for event in state["events"]:
            if event["type"] in ("pick", "handover"):
                self._pick_shot(event, frame, state)
                picked = self.engine.items.get(event["item"])
                if picked is not None and self.catalog.family_of(picked.seen_as) and self.services.verifier and self.services.verifier.enabled:
                    continue  # one of a family (a book): named by the whole-hand looks, not one outline at a time
                self.watch[event["entry"]] = {"person": event["person"], "item": event["item"], "looks": 0, "next_t": t}
            elif event["type"] == "return":
                self._shelf_view(event, frame, state)
            elif event["type"] == "exit":
                self._spawn(self._pay(event["person"]))
        self._second_looks(t, frame, state)
        self._read_hands(t, frame, state)
        self._ask_hands(t, frame, state)
        self._read_shelf(t, frame, state)
        state["events"] = state["events"] + self.engine.events[first:]
        state["seq"] = seq
        # model: SAM on the GPU; round_trip: the call to it, network included; work: this frame's tracking here after
        state["ms"] = {"model": result.get("ms"), "round_trip": round(round_trip), "work": round((time.perf_counter() - work0) * 1000)}
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
                people[pid] = {"status": "known", "profile": book.summary(self.who[pid]), "score": self.scores.get(pid)}
            elif pid in self.strangers:
                people[pid] = {"status": "not_registered", "name": self.strangers[pid], "face": self.stranger_ids.get(pid)}
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
        if not faces:
            return
        in_view = {p["id"] for p in state["people"]}
        inside = book.inside()
        for person in wanted:
            x0, y0, x1, y1 = person["box"]
            head = [f for f in faces if x0 <= f[1] <= x1 and y0 <= f[2] <= y0 + 0.45 * (y1 - y0)]
            if not head:
                continue
            face = book.read(frame, max(head, key=lambda f: f[0][2])[0])
            pid = person["id"]
            known = self.who.get(pid)
            if known:
                if not known.templates or SURE <= book.score(known, face) < 0.7:
                    book.learn(known, face)  # a new angle of someone it is sure of
                continue
            # Who could this be? Only people the gate has let in, plus this run's own shoppers (their visit closes
            # when they are charged, and they may still be here). Not someone who is in view under another outline.
            mine = {profile.id: q for q, profile in self.who.items()}
            elsewhere = {profile.id for q, profile in self.who.items() if q in in_view and q != pid}
            match = book.who(face, among=(inside | set(mine)) - elsewhere)
            walked_in = False
            if not match:
                # Nobody who is inside. A registered shopper who skipped the gate is still in the shop, and is
                # still that shopper: the floor checks them in. With every profile to choose from, only a sure
                # match counts, as at the gate.
                match = book.who(face, skip=elsewhere, floor=SURE)
                walked_in = match is not None
            if match:
                # Two looks naming the same person before anything is put on their bill, unless one look is sure
                # and they came through the gate.
                profile, score = match
                n, best = self.matches.setdefault(pid, {}).get(profile.id, (0, 0.0))
                self.matches[pid][profile.id] = (n + 1, max(best, score))
                if n + 1 < 2 and (score < SURE or walked_in):
                    continue
                self.strangers.pop(pid, None)
                self.stranger_ids.pop(pid, None)
                earlier = mine.get(profile.id)
                if earlier is not None and self._same_shopper(pid, earlier, state):
                    in_view = {p["id"] for p in state["people"]}
                    continue
                if frame is not None:  # what the floor camera saw of them, for the ledger
                    self.face_shots[pid] = (self.engine.t, self._jpeg(self._crop(frame, [x0, y0, x1, y0 + 0.5 * (y1 - y0)], 0.05), 480))
                if walked_in:
                    self.walk_ins.add(pid)
                    if not self.dry:  # a rehearsal opens no visit, as it closes none
                        book.walk_in(profile, self.id, max(best, score), live_since=self.started if self.live else None)
                self._link(pid, profile, max(best, score), how="floor" if walked_in else "recognised")
                continue
            # Not sure enough of anyone to charge them. Does the face look like someone registered? Two looks before
            # saying so, as one poor view of a registered face can miss.
            other = book.who(face, skip=elsewhere)
            self.looks[pid] = self.looks.get(pid, 0) + 1
            name = other[0].name if other else ""
            if self.looks[pid] >= 2 and self.strangers.get(pid) != name:
                first_time = pid not in self.strangers
                self.strangers[pid] = name
                if other:
                    self.stranger_ids[pid] = other[0].id
                if first_time:
                    self.engine.note("unregistered", person=pid, name=name)

    def _link(self, pid: int, profile: Profile, score: float | None = None, how: str = "recognised") -> None:
        """`how` is "recognised" for someone the gate let in, "floor" for someone checked in on the shop floor."""
        self.who[pid] = profile
        extra = {} if score is None else {"score": round(score, 2)}
        if score is not None:
            self.scores[pid] = round(score, 2)
        method = self.services.faces.summary(profile)["method"]
        self.engine.note("checkin", person=pid, name=profile.name, how=how, method=method, **extra)

    def _same_shopper(self, new: int, old: int, state: dict) -> bool:
        """The outline tracked as `new` has the face of this run's shopper `old`, who is not in view under their
        old outline: it is them again. Everything kept about `new` moves over to `old`."""
        if not self.engine.same_person(new, old):
            return False
        for shown in state["people"]:
            if shown["id"] == new:
                shown["id"] = old
        for hand in state["hands"]:
            if hand.get("person") == new:
                hand["person"] = old
        for item in state["items"]:
            if item.get("holder") == new:
                item["holder"] = old
        state["carts"] = self.engine.carts()
        for kept in (self.looks, self.strangers, self.stranger_ids, self.matches, self.all_read, self.unreadable, self.asking, self.face_shots, self.scores):
            kept.pop(new, None)
        memo = self.evidence.pop(new, None)
        if memo:
            mine = self.evidence.setdefault(old, {"read": {}, "recent": []})
            mine["read"].update(memo["read"])
            mine["recent"] = sorted(mine["recent"] + memo["recent"], key=lambda view: view[0])[-2:]
            mine["last"] = sorted(mine.get("last", []) + memo.get("last", []), key=lambda kept: kept[0])[-LAST_LOOKS:]
            if memo.get("scene"):
                mine["scene"] = memo["scene"]
        asked, last = self.hand_asks.pop(new, (0, -9.0))
        before = self.hand_asks.get(old, (0, -9.0))
        self.hand_asks[old] = (before[0] + asked, max(before[1], last))
        for title, memo in self.carried.pop(new, {}).items():
            self.carried.setdefault(old, {}).setdefault(title, memo)
        if new in self.hand_seen_t:
            self.hand_seen_t[old] = max(self.hand_seen_t.pop(new), self.hand_seen_t.get(old, -1.0))
        self.checks.setdefault(old, set()).update(self.checks.pop(new, set()))
        for job in self.watch.values():
            if job["person"] == new:
                job["person"] = old
        return True

    # pictures for the last check

    @staticmethod
    def _image(kept) -> np.ndarray:
        """A kept picture as an image: whole frames are kept as JPEG bytes."""
        return cv2.imdecode(np.frombuffer(kept, np.uint8), cv2.IMREAD_COLOR) if isinstance(kept, (bytes, bytearray)) else kept

    @staticmethod
    def _shrink(image: np.ndarray, side: int = PROOF_SIDE) -> np.ndarray:
        """No bigger than the model and the ledger are shown it."""
        h, w = image.shape[:2]
        k = side / max(h, w)
        return cv2.resize(image, (max(2, round(w * k)), max(2, round(h * k))), interpolation=cv2.INTER_AREA) if k < 1 else image

    @staticmethod
    def _crop(frame: np.ndarray, box: list[float], grow: float = 0.0) -> np.ndarray:
        h, w = frame.shape[:2]
        bw, bh = box[2] - box[0], box[3] - box[1]
        x0, y0 = int(max(0.0, box[0] - grow * bw) * w), int(max(0.0, box[1] - grow * bh) * h)
        x1, y1 = int(min(1.0, box[2] + grow * bw) * w), int(min(1.0, box[3] + grow * bh) * h)
        return frame[y0:max(y1, y0 + 2), x0:max(x1, x0 + 2)].copy()

    @staticmethod
    def _hands_box(pid: int, state: dict) -> list[float] | None:
        """The part of the frame with this shopper's outlined hands and the items held in them, or None."""
        boxes = [h["box"] for h in state["hands"] if h.get("person") == pid]
        boxes += [i["box"] for i in state["items"] if i["state"] == "held" and i.get("holder") == pid]
        if not boxes:
            return None
        return [min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)]

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

    def _keep_views(self, t: float, frame: np.ndarray, state: dict, jpeg: bytes | None = None) -> None:
        """Keep the latest couple of pictures of every shopper who has something on their bill: the last check
        wants to see what they still had on the way out. Whole frames are kept as the JPEG the camera sent (a raw
        frame 1600 wide is 4 MB, its JPEG about a twentieth of that) and close views at the size the model is shown."""
        verifier = self.services.verifier
        if verifier is None or not verifier.enabled:
            return
        for person in state["people"]:
            pid = person["id"]
            memo = self.evidence.setdefault(pid, {"read": {}, "recent": []})
            if not (self.engine.holding(pid) or self.engine.unpaid(pid) or self.engine.unnamed(pid)):
                continue
            last = memo.setdefault("last", [])  # their last few seconds, closely spaced: read again at the door
            if not last or not 0 <= t - last[-1][0] < LAST_GAP_S:
                last.append((t, jpeg if jpeg is not None else frame, person["box"], self._shrink(self._carrying_view(pid, frame, state, person["box"]))))
                del last[:-LAST_LOOKS]
            if memo["recent"] and 0 <= t - memo["recent"][-1][0] < VIEW_GAP_S:
                continue
            memo["recent"].append((t, last[-1][3] if last[-1][0] == t else self._shrink(self._carrying_view(pid, frame, state, person["box"]))))
            del memo["recent"][:-2]
            memo["scene"] = (t, jpeg if jpeg is not None else frame)  # and the whole picture, shelves included, the last time they were seen

    def _shelf_view(self, event: dict, frame: np.ndarray, state: dict) -> None:
        """The floor saw something of theirs go back: keep a picture of that moment for the check before charging."""
        verifier, pid = self.services.verifier, event.get("person")
        person = next((p for p in state["people"] if p["id"] == pid), None)
        if verifier is None or not verifier.enabled or person is None or frame is None:
            return
        shelf = self.evidence.setdefault(pid, {"read": {}, "recent": []}).setdefault("shelf", [])
        if not shelf or shelf[-1][0] != event["t"]:
            shelf.append((event["t"], self._shrink(self._carrying_view(pid, frame, state, person["box"]))))
            del shelf[:-LAST_LOOKS]

    def _pictures(self, pid: int, entries: list) -> list[tuple[float, np.ndarray]]:
        """What the last check is shown, oldest first: for each line the picture in which that item was read in
        their hands, then the latest close view of their hands, then the whole scene the last time they were seen."""
        return [(t, image) for t, image, _ in self._evidence(pid, entries)]

    def _evidence(self, pid: int, entries: list) -> list[tuple[float, np.ndarray, str]]:
        """The same pictures, each with what it is in words, for the ledger."""
        memo = self.evidence.get(pid)
        if not memo:
            return []
        chosen: dict[float, tuple[np.ndarray, list[str]]] = {}
        for entry in entries:
            hit = memo["read"].get(entry.item.cls)
            if hit:
                chosen.setdefault(hit[0], (hit[1], []))[1].append(self.catalog.by_id[entry.item.cls].name)
        ending = [(t, image, "last close view of hands") for t, image in memo["recent"] if t not in chosen][-1:]
        if memo.get("scene"):
            ending.append((memo["scene"][0], self._image(memo["scene"][1]), "last full view"))
        read = [(t, image, " and ".join(names) + " in hand") for t, (image, names) in sorted(chosen.items())]
        return read[-(AUDIT_PICTURES - len(ending)):] + ending

    @staticmethod
    def _jpeg(image: np.ndarray, side: int = PROOF_SIDE) -> bytes:
        h, w = image.shape[:2]
        k = min(1.0, side / max(h, w))
        small = cv2.resize(image, (max(2, round(w * k)), max(2, round(h * k))), interpolation=cv2.INTER_AREA) if k < 1 else image
        return cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()

    def _pick_shot(self, event: dict, frame: np.ndarray | None, state: dict) -> None:
        """Keep the frame in which an item was taken, cropped to the shopper, with the item outlined: the first
        picture behind that line of the bill."""
        shown = next((i for i in state["items"] if i["id"] == event.get("item")), None)
        if frame is None or shown is None or not shown["polygons"]:
            return
        h, w = frame.shape[:2]
        marked = frame.copy()
        outline = [np.array([[x * w, y * h] for x, y in poly], np.int32) for poly in shown["polygons"]]
        cv2.polylines(marked, outline, True, (0, 255, 255), max(2, w // 400), cv2.LINE_AA)
        box = list(shown["box"])
        person = next((p for p in state["people"] if p["id"] == event.get("person")), None)
        if person:
            box = [min(box[0], person["box"][0]), min(box[1], person["box"][1]), max(box[2], person["box"][2]), max(box[3], person["box"][3])]
        self.pick_shots[event["entry"]] = (event["t"], self._jpeg(self._crop(marked, box, 0.06)))

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
        if not (verifier and verifier.enabled) and not (reader and reader.ready) and not self.likeness.wanted:
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
        asking = bool(verifier and verifier.enabled)
        liking = asyncio.ensure_future(self._liken(entry_id, frame, box, candidates))  # beside the second model, not before it
        refs = self._references(candidates) if asking else []
        try:
            verdict = await verifier.held_item(frame, box, candidates, reading.text if reading else "", **({"refs": refs} if refs else {})) if asking else None
        finally:
            liked, liked_why = await liking
        first = len(self.engine.events)
        # A likeness says what an item is, not that it left the shelf: a pick that never did still looks exactly like
        # its product. So it is one vote per line at most, and only when the second model's look at this same moment
        # sees the item held (or there is no second model). It can then help name the item, never outvote "not held".
        if liked and entry_id not in self.liked and (verdict["holding"] if verdict is not None else not asking):
            self.liked.add(entry_id)
            self.engine.apply_look(entry_id, True, liked, f"{liked_why}: {self.catalog.by_id[liked].name}", names_only=True)
        if verdict is not None:
            agree = bool(reading and reading.product and reading.score >= 0.6 and verdict["product"] == reading.product.id)
            why = f'Reads "{reading.quote()}", confirmed by the vision model' if agree else verdict["why"]
            self.engine.apply_look(entry_id, verdict["holding"], verdict["product"], why, weight=2 if agree else 1)
            if verdict["holding"] and verdict["product"] and pid is not None:
                # a picture in which this item was seen in their hand, for the check before charging
                self.evidence.setdefault(pid, {"read": {}, "recent": []})["read"][verdict["product"]] = (t, self._shrink(self._crop(frame, box, 1.5)))
        elif reading and reading.sure:
            self.engine.apply_look(entry_id, True, reading.product.id, f'Reads "{reading.quote()}"', weight=2)
        self._update(first)

    async def _liken(self, entry_id: int, frame: np.ndarray, box: list[float], candidates: list) -> tuple[str | None, str]:
        """Compare the picked item with the catalogue's pictures. Returns the product a clear likeness votes for."""
        if self.likeness_job is not None:
            await asyncio.wait([self.likeness_job], timeout=4)  # the example vectors, made once a visit
        if not self.likeness.ready:
            return None, ""
        picture = cut_out(frame, box)
        if picture is None:
            return None, ""
        try:
            scores = await self.likeness.compare(picture, [p.id for p in candidates if p.billable])
        except Exception as e:
            log.warning("likeness failed: %s", e)
            return None, ""
        if not scores:
            return None, ""
        product, why = Likeness.verdict(scores)
        self.engine.note_likeness(entry_id, scores[0][0], scores[0][1], why)
        return product, why

    def _references(self, candidates: list) -> list[tuple[str, bytes]]:
        """The first example picture of each candidate that has one, for the second model: (product id, JPEG)."""
        from .likeness import examples_dir

        if not self.catalog.shop:
            return []
        out = []
        for p in candidates:
            if p.examples and p.billable:
                path = examples_dir(self.catalog.shop) / p.examples[0]
                if path.exists():
                    out.append((p.id, path.read_bytes()))
            if len(out) >= 4:
                break
        return out

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
            # A look takes a second or two, and a title is often legible for a second or less as a book turns in the
            # hand: asking only when the last look is back steps right over it. So the frames are taken at a steady
            # pace whatever the model's speed, looked at two at a time, the rest queued in order. Every frame, three
            # at a time, while a pick of theirs still has no confirmed title, the last look saw something in their
            # hands it could not read, or a title has been read once and waits for its second reading.
            single = any(v["votes"] == 1 and not v["on_cart"] for v in self.carried.get(pid, {}).values())  # one reading, waiting for its second
            eager = self.engine.unnamed(pid) > 0 or self.unreadable.get(pid, 0) > 0 or single
            if asked >= HAND_ASKS or t - last < (EAGER_GAP_S if eager else HAND_ASK_GAP_S):
                continue
            busy = self.asking.get(pid, 0) >= (EAGER_IN_FLIGHT if eager else IN_FLIGHT)
            carrying = any(v["on_cart"] for v in self.carried.get(pid, {}).values())
            if not (self.engine.holding(pid) or carrying or self.engine.unnamed(pid)):
                continue  # nothing taken yet: nothing to ask about
            self.hand_asks[pid] = (asked + 1, t)
            look = (t, frame, person["box"], self._shrink(self._carrying_view(pid, frame, state, person["box"])), self._hands_box(pid, state))
            if busy:
                backlog = self.hand_backlog.setdefault(pid, [])
                backlog.append((t, self._jpeg(frame, 1600), *look[2:]))  # a JPEG: a raw frame is 4 MB
                del backlog[:-HAND_BACKLOG]
                continue
            self._start_look(pid, *look)

    @staticmethod
    def _paired(memo: dict) -> bool:
        """Two readings of a title close enough to count as two readings of one book: a few seconds apart at most, with
        fewer than three looks between them (by frame time) in which the hands were in view and it was not read."""
        reads, missed = sorted(memo.get("reads", [])), memo.get("missed", [])
        return any(b - a <= PAIR_S and sum(1 for m in missed if a < m < b) < HAND_MISSES for a, b in zip(reads, reads[1:]))

    def _start_look(self, pid: int, t: float, frame, box: list[float], view: np.ndarray, hands: list[float] | None = None) -> None:
        self.asking[pid] = self.asking.get(pid, 0) + 1
        task = self._spawn(self._ask_hand(pid, t, self._image(frame), box, view, hands))
        pending = self.checks.setdefault(pid, set())
        pending.add(task)
        task.add_done_callback(pending.discard)

    def _next_look(self, pid: int) -> None:
        """A look is back: the oldest queued frame of this shopper goes next."""
        backlog = self.hand_backlog.get(pid)
        if backlog and not self.closed:
            self._start_look(pid, *backlog.pop(0))

    async def _ask_hand(self, pid: int, t: float, frame: np.ndarray, box: list[float], view: np.ndarray | None = None, hands: list[float] | None = None) -> None:
        try:
            books = [p for group in self.catalog.members.values() for p in group]
            # the close-up is of their hands and what is in them, enlarged, when those are outlined: a title on a book
            # in a hand is a few pixels tall in the whole shopper
            answer = await (self.services.verifier.carried(frame, box, books, hands) if hands else self.services.verifier.carried(frame, box, books))
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
                    memo.update(votes=0, on_cart=False, reads=[], missed=[])
                memo.setdefault("reads", []).append(t)
                memo["votes"], memo["misses"], memo["last_t"] = memo["votes"] + 1, 0, max(t, memo["last_t"])
                # Two readings put a title on the bill: two looks at their hands, or one look and a line of theirs
                # that already carries this title from the shelf it was taken from. Two looks pair up by when their
                # frames were taken, not when the answers came back (answers overtake each other).
                named = engine.awaiting(pid, product)
                paired = memo["votes"] >= 2 or self._paired(memo)
                if (paired or named) and not memo["on_cart"]:
                    why = f"Identified on the shelf, confirmed in hand: {answer['why']}" if not paired else f"Read in hand twice: {answer['why']}"
                    memo["on_cart"] = engine.name_line(pid, product, why) or engine.add_seen(pid, product, why) or True
            if answer["hands_visible"] and not in_order:
                # an answer that came back late still says which titles that frame did not show, for pairing readings
                for product, memo in seen.items():
                    if product not in answer["products"] and not memo["on_cart"]:
                        memo.setdefault("missed", []).append(t)
            if answer["hands_visible"] and in_order:
                # A title on the cart that this look did not read. If the look saw at least as many things in the
                # hands that it could not read, the title is most likely one of those (turned away, or under
                # another book), so it is not a sign of anything having been put down.
                unlisted = [p for p, memo in seen.items() if memo["on_cart"] and p not in answer["products"]]
                hidden = answer["unreadable"] >= len(unlisted) > 0
                for product, memo in seen.items():
                    if product in answer["products"] or (hidden and memo["on_cart"]):
                        continue
                    memo.setdefault("missed", []).append(t)
                    memo["misses"] += 1
                    if memo["on_cart"] and memo["misses"] >= HAND_MISSES and t - memo["last_t"] >= HAND_GONE_S:
                        # the engine refuses while that line's own outline is still in their hand
                        if engine.drop_title(pid, product, "no longer in their hands"):
                            memo.update(votes=0, misses=0, on_cart=False, reads=[], missed=[])
                    elif not memo["on_cart"] and memo["misses"] >= HAND_MISSES and t - memo["last_t"] >= VOTE_LAPSE_S:
                        memo["votes"] = 0  # a one-off reading that never came back (by looks and by time: looks come faster while eager)
                # with nothing unreadable in the hands on two looks running, picks still without a name are stray outlines
                self.unreadable[pid] = answer["unreadable"]
                self.all_read[pid] = self.all_read.get(pid, 0) + 1 if answer["unreadable"] == 0 and answer["products"] else 0
                if self.all_read[pid] >= 2:
                    engine.void_unnamed(pid, t - 2.0)
            if len(engine.events) > first:
                self._update(first)
        finally:
            self.asking[pid] = max(0, self.asking.get(pid, 0) - 1)
            self._next_look(pid)

    async def _last_looks(self, pid: int) -> None:
        """What did this shopper have with them as they left? Their last few pictures, about half a second apart,
        are read once more, all at once, and two things are settled from them before anything is charged.

        Which titles.   A pick that never got a confirmed title (taken in their last seconds in view, or not legible
                        at the time) gets one on the rule the cart always follows, two readings: two of these
                        pictures, or one of them with an earlier look or with the name the line carries from the
                        shelf.
        Still with them. A title on the bill that none of these pictures shows in their hands, while the pictures do
                        show their hands and nothing in them is left unread, went back on a shelf at some point the
                        outlines missed. It comes off the charge. Things in their hands that no picture could read
                        are given the benefit of the doubt: first to picks that have no title, then to the titles
                        read most recently."""
        verifier, engine = self.services.verifier, self.engine
        if not self.catalog.members or verifier is None or not verifier.enabled:
            return
        family = lambda e: e.person == pid and e.state == "in_cart" and not e.paid and bool(self.catalog.family_of(e.item.seen_as))
        if not any(family(e) for e in engine.entries):
            return
        memo = self.evidence.get(pid) or {}
        frames = [kept for kept in memo.get("last", []) if kept[1] is not None]
        if not frames:
            return
        books = [p for group in self.catalog.members.values() for p in group]
        answers = await asyncio.gather(*(verifier.carried(self._image(frame), box, books) for _, frame, box, _ in frames), return_exceptions=True)
        read: dict[str, list] = {}
        clear = []  # (time, answer) for the pictures in which their hands could be seen, oldest first
        for (t, _, _, view), answer in zip(frames, answers):
            if not isinstance(answer, dict):
                continue
            self.hand_asks[pid] = (self.hand_asks.get(pid, (0, -9.0))[0] + 1, t)
            if answer["hands_visible"]:
                clear.append((t, answer))
            for product in answer["products"]:
                read.setdefault(product, []).append((t, view, answer["why"]))
        first = len(engine.events)
        seen = self.carried.setdefault(pid, {})
        for product, looks in read.items():
            t, view, why = looks[-1]
            memo["read"][product] = (t, view)  # the picture of them leaving with it, for the check before charging
            note = seen.setdefault(product, {"votes": 0, "misses": 0, "last_t": t, "on_cart": False})
            note["last_t"] = max(note["last_t"], t)
            if note["on_cart"] and engine.has_title(pid, product):
                continue
            # two readings: these pictures, an earlier look, or the name the line brought from the shelf
            if len(looks) + (1 if note["votes"] else 0) + (1 if engine.awaiting(pid, product) else 0) < 2:
                continue
            why = f"Read in hand at exit: {why}"
            note["on_cart"] = engine.name_line(pid, product, why) or engine.add_seen(pid, product, why) or True
        if len(clear) >= 2:  # one picture is no basis for taking anything off a bill
            counts = sorted(answer["unreadable"] for _, answer in clear)
            self.unreadable[pid] = counts[(len(counts) - 1) // 2]  # for keeping an untitled pick, what the pictures agree on
            # Gone, in the order the pictures were taken. A title is gone when its last reading in their hands (here,
            # or on the floor just before) is followed by three of these pictures that do not show it: the same three
            # missing readings that take a title off on the floor, but now even within their last seconds, so a book
            # put back on the way out comes off. A title none of them reads, read longer ago, needs two.
            last_read = {p: looks[-1][0] for p, looks in read.items()}
            since = lambda p: max(last_read.get(p, -1.0), seen.get(p, {}).get("last_t", -1.0))
            later = lambda p: sum(1 for t, _ in clear if t > since(p))
            billed = [e for e in engine.unpaid(pid) if family(e) and e.item.settled]
            gone = {e.item.settled for e in billed if later(e.item.settled) >= HAND_MISSES}
            # The benefit of the doubt: things in their hands that a picture could not read may be titles on the bill.
            # Per picture, the unreadable things there are first those titles still with them (read in some picture
            # and not gone since) that this picture did not read, then picks without a title made by then; what is
            # left over may be a title that is missing. The most any picture leaves over is what can be given.
            present = set(last_read) - gone
            untitled = [e for e in engine.unpaid(pid) if family(e) and e.verified is None and e.item.settled is None]
            spare = max((answer["unreadable"] - sum(1 for p in present if p not in answer["products"]) - sum(1 for e in untitled if e.picked_t <= t) for t, answer in clear), default=0)
            unread = sorted((e for e in billed if e.item.settled not in last_read and e.item.settled not in gone), key=lambda e: -since(e.item.settled))
            # Titles read twice in their hands whose outline the floor then saw go back after that reading: in a stack
            # a track can slip onto the book next to it, and that may have been the one put back. They too may be the
            # unreadable thing in their hands.
            on_cart = {e.item.settled for e in engine.entries if e.person == pid and e.state == "in_cart"}
            lost = [e for e in engine.entries if e.person == pid and e.state == "returned" and not e.paid and e.verified == "confirmed" and self.catalog.family_of(e.item.seen_as)
                    and e.item.settled and e.item.settled not in on_cart and e.returned_t is not None and e.returned_t > seen.get(e.item.settled, {}).get("last_t", e.returned_t)]
            lost = list({e.item.settled: e for e in lost}.values())
            missing = [e for e in billed if e.item.settled in gone]
            if spare > 0 and (len(unread) > spare or lost or missing):
                # More titles than unreadable things, or titles that went missing while something stayed unreadable:
                # which ones the unreadable things are is not for counting to say. The check before charging decides,
                # from what it sees go onto the shelf and what the book in their hands looks like
                # (Verifier.check_visit, "placed" and "which"); a title gone here needs both of the last views.
                self.doubt[pid] = {"spare": spare, "on_bill": [e.id for e in unread], "gone": [e.id for e in missing], "lost": [e.id for e in lost]}
                absent = []
            else:
                absent = unread[max(0, spare):] + missing
            for entry in absent:
                product = entry.item.settled
                why = (f"Put down on the way out: last in hand at {since(product):.1f} s, not in the {later(product)} pictures after it" if product in last_read
                       else f"Not in hand in any of the last {len(clear)} pictures")
                engine.apply_audit(entry.id, "put_back", why)
                seen.get(product, {}).update(on_cart=False)
        if len(engine.events) > first:
            self._update(first)

    async def _last_goods(self, pid: int) -> None:
        """Is each product on the bill still with the shopper as they leave? For products told apart by their look,
        not by a title (books have their own door reading, `_last_looks`).

        A line is charged only while something still links the item to the shopper at the end:

        Seen lately.    The floor saw the item with them in their last `LINK_S` seconds in view.
        Seen at the door. The second model, shown their last few pictures, sees it on them.

        Neither, and the line is held back for review ("unclear"): the item dropped out of sight well before they left
        and is not with them now, which is what an item put down out of the camera's view looks like. Things in their
        hands at the door that the model could not match to the catalogue are given the benefit of the doubt, the
        items seen most recently first. And an item seen lately is still taken off ("put back") when the door pictures
        show both of their hands, twice or more, with nothing in them the model cannot name and not this item."""
        engine, verifier = self.engine, self.services.verifier
        lines = [e for e in engine.unpaid(pid) if not self.catalog.family_of(e.item.seen_as)]
        if not lines:
            return
        person = engine.people.people.get(pid)
        left_t = person.last_t if person is not None else engine.t
        answers = []
        memo = self.evidence.setdefault(pid, {"read": {}, "recent": []})
        frames = [kept for kept in memo.get("last", []) if kept[1] is not None]
        if verifier is not None and verifier.enabled and frames:
            goods = [p for p in self.catalog.products if not self.catalog.family_of(p.id)]
            billed = list(dict.fromkeys(e.item.cls for e in lines))
            refs = self._references([self.catalog.by_id[c] for c in billed] + [p for p in goods if p.id not in billed])
            got = await asyncio.gather(*(verifier.still_carried(self._image(frame), box, goods, refs) for _, frame, box, _ in frames), return_exceptions=True)
            answers = [(t, view, a) for (t, _, _, view), a in zip(frames, got) if isinstance(a, dict)]
        with_them: set[str] = set()
        for t, view, answer in answers:
            for product in answer["products"]:
                with_them.add(product)
                memo["read"][product] = (t, view)  # the picture of them leaving with it, for the check before charging
        both = [a for _, _, a in answers if a["hands_seen"] >= 2]
        unknown = max((a["unknown"] for _, _, a in answers), default=0)
        first = len(engine.events)
        unlinked = []
        for entry in lines:
            if entry.item.cls in with_them:
                continue
            gap = max(0.0, left_t - entry.item.last_t)
            if gap > LINK_S:
                unlinked.append((gap, entry))
            elif len(both) >= 2 and unknown == 0:
                engine.apply_audit(entry.id, "put_back", f"Not in either hand in the last {len(both)} pictures")
        unlinked.sort(key=lambda x: x[0])
        for gap, entry in unlinked[unknown:]:
            engine.apply_audit(entry.id, "unclear", f"Not seen with the shopper as they left (last in hand {gap:.0f} s earlier)")
        if len(engine.events) > first:
            self._update(first)

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
                        self.engine.settle(item_id, named.id, f"Vision model: {answer['why']}", sure=False)
            if len(self.engine.events) > first:
                self._update(first)
        finally:
            self.reading.discard(item_id)

    # the last check, then payment

    async def _audit(self, pid: int, floor: set[int] = frozenset()) -> dict:
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
            if callable(getattr(verifier, "check_visit", None)):
                return await self._check_steps(pid, entries, floor)
            for entry in entries:  # no stepwise check to settle the door's doubt: a title that went missing stays off
                if entry.id in self.doubt.get(pid, {}).get("gone", []):
                    engine.apply_audit(entry.id, "put_back", "Not in hand in their last pictures")
            self.doubt.pop(pid, None)
            entries = engine.unpaid(pid)
            if not entries:
                return {"status": "nothing", "kept": 0, "held": 0}
            noted = self._evidence(pid, entries)
            pictures = [(t, image) for t, image, _ in noted]
            self.shown[pid] = [Verifier.lettered(image, chr(65 + i)) for i, (_, image) in enumerate(pictures)]
            self.shown_notes[pid] = [(t, f"Final check, picture {chr(65 + i)}: {what}") for i, (t, _, what) in enumerate(noted)]
            asked = []
            for n, entry in enumerate(entries, 1):
                # two lines for one product are two items: the check counts how many it sees in hand (verify.audit)
                product = self.catalog.by_id[entry.item.cls]
                asked.append({"line": n, "name": product.name, **({"by": product.author} if product.author else {}), "looks_like": product.cues or product.prompt})
            answer = None
            for _ in range(2 if pictures else 0):
                answer = await verifier.audit(asked, pictures)
                if answer is not None:
                    break
            first = len(engine.events)
            missing = "The final check got no answer" if pictures else "No picture of it in hand was kept"
            for n, entry in enumerate(entries, 1):
                verdict = answer[n] if answer else {"verdict": "unclear", "why": missing}
                if answer and verdict["verdict"] == "unclear" and entry.id in floor and entry.verified == "confirmed" and self.catalog.family_of(entry.item.seen_as):
                    # A titled item (a book) is confirmed only on two readings of its title in their hands, this one
                    # on the floor before they left, and the door look did not find it gone. The final check only
                    # failed to make it out again: a book under another in a stack, or one of several books without a
                    # picture of its own. The readings stand. Not for a title first read at the door: those readings
                    # come from the very pictures the final check has just looked at, so they add nothing. Nor for
                    # products without a title, which can be confirmed on one look.
                    verdict = {"verdict": "kept", "why": "Confirmed in hand in store; title not visible at checkout"}
                engine.apply_audit(entry.id, verdict["verdict"], verdict["why"])
            kept = sum(1 for e in entries if e.audit == "kept")
            self._update(first)
            return {"status": "checked" if answer else "failed", "kept": kept, "held": len(entries) - kept, "pictures": len(pictures), "session": self.id, "person": pid}
        finally:
            self.checking.discard(pid)

    async def _check_steps(self, pid: int, entries: list, floor: set[int]) -> dict:
        """The check before charging as several questions (Verifier.check_visit). Every step goes into the visit's
        events, and every picture it looked at into its proofs, so the bill shows why each line was charged or not.
        Lines the door look already found gone are shown to it too: what goes back on a shelf is matched to them,
        and they can only stay off the bill."""
        engine, verifier, catalog = self.engine, self.services.verifier, self.catalog
        memo = self.evidence.get(pid) or {}
        gone = [e for e in engine.entries if e.person == pid and e.state == "in_cart" and not e.paid and e.audit == "put_back"
                and e.item.cls in catalog.by_id and catalog.by_id[e.item.cls].billable and e not in entries]
        everything = entries + gone
        lines, held, pick = [], {}, {}
        for n, entry in enumerate(everything, 1):
            product = catalog.by_id[entry.item.cls]
            hit = memo.get("read", {}).get(entry.item.cls)
            lines.append({"line": n, "product": product.id, "name": product.name, **({"by": product.author} if product.author else {}), "looks_like": product.cues or product.prompt,
                          "door": "put_back" if entry in gone else None, "door_why": entry.audit_why if entry in gone else "", "last_read_t": hit[0] if hit else None,
                          "titled": bool(catalog.family_of(product.id))})
            if hit:
                held[n] = (hit[0], self._image(hit[1]))
            shot = self.pick_shots.get(entry.id)
            if shot:
                pick[n] = (shot[0], self._image(shot[1]))
        views = [(t, view) for t, _, _, view in memo.get("last", [])] or list(memo.get("recent", []))
        scene = (memo["scene"][0], self._image(memo["scene"][1])) if memo.get("scene") else None
        family = {catalog.family_of(e.item.seen_as) for e in everything} - {None}
        pool = [p for f in family for p in catalog.members.get(f, [])] + [p for p in catalog.products if not p.family and not p.head and (not family or not p.billable)]
        catalogue = [{"id": p.id, "name": p.name, **({"by": p.author} if p.author else {}), "looks_like": p.cues or p.prompt} for p in {p.id: p for p in pool}.values()]
        doubt, candidates = self.doubt.pop(pid, None), []
        if doubt:
            ids = {e.id: n for n, e in enumerate(everything, 1)}
            for entry in engine.entries:
                if entry.id in doubt["on_bill"] + doubt["gone"] + doubt["lost"] and entry.item.settled in catalog.by_id:
                    product = catalog.by_id[entry.item.settled]
                    candidates.append({"product": product.id, "name": product.name, "looks_like": product.cues or product.prompt, "line": ids.get(entry.id),
                                       "sure": entry.id in doubt["on_bill"],  # one view is enough only for a title not seen to go
                                       "last_read_t": self.carried.get(pid, {}).get(product.id, {}).get("last_t")})
        result = await verifier.check_visit(lines, {"held": held, "pick": pick, "views": views, "shelf": list(memo.get("shelf", [])), "scene": scene}, catalogue,
                                            doubt={"spare": doubt["spare"], "candidates": candidates} if candidates else None)
        first = len(engine.events)
        by_line = {n: entry for n, entry in enumerate(everything, 1)}
        if result:
            self.shown[pid] = [self._jpeg(image) for _, image, _ in result["pictures"]]
            self.shown_notes[pid] = [(t, f"Final check, {caption}") for t, _, caption in result["pictures"]]
            for step in result["steps"]:
                entry = by_line.get(step["line"])
                engine.note("check", person=pid, entry=entry.id if entry else None, name=step.get("name"), step=step["step"], ask=step["ask"], at=step["t"], answer=step["answer"], why=step["why"])
        for n, entry in by_line.items():
            verdict = (result or {}).get("verdicts", {}).get(n) or {"verdict": "unclear", "why": "The final check got no answer", "absent": False}
            if entry in gone:
                if verdict["verdict"] == "put_back":
                    engine.apply_audit(entry.id, "put_back", verdict["why"])
                elif verdict["verdict"] == "kept":  # the door and the check disagree: held for review, not guessed
                    engine.apply_audit(entry.id, "unclear", f"Found gone at the door, seen in hand by the final check: {verdict['why']}")
                elif not verdict.get("absent") and entry.id in floor and entry.verified == "confirmed" and catalog.family_of(entry.item.seen_as):
                    # gone from the door's pictures but not seen going onto a shelf: a title read twice in their hands
                    # on the floor stands, as before the door look counted pictures
                    engine.apply_audit(entry.id, "kept", f"Confirmed in hand in store; title not visible at checkout ({verdict['why']})")
                else:
                    engine.apply_audit(entry.id, "unclear", verdict["why"])
                continue
            if result and verdict["verdict"] == "unclear" and not verdict.get("absent") and entry.id in floor and entry.verified == "confirmed" and catalog.family_of(entry.item.seen_as):
                # the same rule as the one-look check: a title read twice in their hands on the floor, not found gone
                # at the door, and not seen out of their hands by this check, is charged
                verdict = {"verdict": "kept", "why": f"Confirmed in hand in store; title not visible at checkout ({verdict['why']})"}
            engine.apply_audit(entry.id, verdict["verdict"], verdict["why"])
        for back in (result or {}).get("reinstate", []):
            # a title read twice in their hands, whose outline the floor saw go back: the book they left with looks
            # like it in both of the last views, and no other title on the bill can be that book
            if engine.add_seen(pid, back["product"], f"Read in hand earlier; {back['why']}"):
                entry = engine.entries[-1]
                engine.apply_audit(entry.id, "kept", back["why"])
                entries.append(entry)
                everything.append(entry)
        kept = sum(1 for e in everything if e.audit == "kept")
        self._update(first)
        return {"status": "checked" if result else "failed", "kept": kept, "held": sum(1 for e in everything if e.audit == "unclear"), "pictures": len((result or {}).get("pictures", [])),
                "steps": len((result or {}).get("steps", [])), "session": self.id, "person": pid}

    async def _pay(self, pid: int) -> None:
        # do not bill a line the second model is still looking at
        # (and the frames still queued for a look at their hands, which each finished look sets off)
        until = time.monotonic() + VERIFY_WAIT_S
        while (pending := list(self.checks.get(pid, ()))) and time.monotonic() < until:
            await asyncio.wait(pending, timeout=until - time.monotonic())
        engine, pay, book = self.engine, self.services.payments, self.services.faces
        left_at = time.time()
        # the lines whose titles were read in their hands on the floor, before the door look added its own readings
        floor = {e.id for e in engine.entries if e.person == pid and e.verified == "confirmed"}
        titles = {e.item.settled for e in engine.entries if e.id in floor and e.item.settled}
        if pid in self.who:
            await self._last_looks(pid)
            await self._last_goods(pid)
        # by title: a line the door look re-added for a title already confirmed in their hands on the floor (its
        # floor line lost to a track that went astray) was not first read at the door
        floor |= {e.id for e in engine.entries if e.person == pid and e.verified == "confirmed" and e.item.settled in titles}
        first = len(engine.events)
        if self.hand_asks.get(pid, (0, 0.0))[0] >= 3:
            # Their hands were read several times. A pick that never got a confirmed title from those readings is a
            # stray outline, unless the last reading said there was something it could not read.
            engine.void_unnamed(pid, engine.t, keep=self.unreadable.get(pid, 0))
        shopper = self.who.get(pid)
        # the gate's picture and time are proof only for someone the gate let in, not a visit the floor opened
        admitted = book.open_visit(shopper) if shopper and pid not in self.walk_ins else None
        gate = book.gate_shot(shopper) if admitted else None
        # someone who is not registered is never charged, so there is nothing to check for them
        check = await self._audit(pid, floor) if shopper else {"status": "skipped", "kept": 0, "held": 0}
        entries = engine.unpaid(pid)
        lines = [engine.catalog.by_id[e.item.cls] for e in entries]
        amount = round(sum(p.price for p in lines), 2)
        billed = [(p.id, p.name, p.price, e.audit_why or e.why) for p, e in zip(lines, entries)]
        detail = [(p.name, p.price, " · ".join(x for x in (p.category, p.section) if x)) for p in lines]
        base = {"amount": amount, "currency": self.catalog.currency.lower(), "items": len(lines), "id": None, "receipt_url": None, "message": "", "check": check}
        if amount <= 0:
            payment = {**base, "status": "nothing_due"}
        elif shopper is None:
            payment = {**base, "status": "not_registered", "message": "Shopper not identified, so nothing was charged."}
        elif self.dry:
            payment = {**base, "status": "dry", "message": "Rehearsal: nothing was charged and no email was sent."}
        elif not book.can_pay(shopper):
            payment = {**base, "status": "no_card", "message": "Left with unpaid items and no payment method on file."}
        elif pay is None or not pay.enabled:
            payment = {**base, "status": "off", "message": pay.problem if pay else "Payments are not set up."}
        elif not book.claim_visit(shopper, self.id):
            # two runs of the floor watched the same shopper (the page and a replay, say): only one of them charges
            payment = {**base, "status": "settled", "message": "Another run of the shop floor had already settled this visit, so nothing was charged again."}
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
        door_s = round(time.time() - left_at, 1)

        # The ledger. A walk-out with something on the bag is written down, and ends the visit. Someone who only
        # passed through the picture with empty hands has not finished shopping: their visit stays open, so they
        # are still known when they come back into view. Payments not set up, or a visit another run settled:
        # nothing happened here worth keeping either.
        held = [(line["product"], line["name"], line["price"] or 0.0, self._why_held(line)) for line in engine._cart(pid)["lines"] if self._held_back(line)]
        ledger, row = (Ledger(book) if book else None), None
        if shopper and payment["status"] not in ("off", "settled") and (billed or held):
            row = book.record_visit(shopper, self.id, payment, billed, held, dry=self.dry, shop=self.catalog.name)
            ledger.add_proofs(row, self._proofs(pid, gate))
            ledger.finish(row, payment, self._detail(pid, payment, admitted, door_s))
            if not self.dry:
                book.close_visit(shopper, self.id)  # they have left: the next visit starts at the gate
        try:
            if payment["status"] == "succeeded" and callable(getattr(pay, "invoice", None)):
                first = len(engine.events)
                made = await pay.invoice(shopper.name, shopper.email or "", detail, self.catalog.currency, payment["id"], self.catalog.name, f"{self.id}-{pid}-{len(self.payments[pid])}")
                payment["invoice"] = made
                payment["receipt_url"] = made.get("url") or payment["receipt_url"]
                engine.note("invoice", person=pid, status=made["status"], number=made.get("number"), url=made.get("url"), message=made.get("message", ""))
                self._update(first)
            mailer = self.services.mailer
            if payment["status"] == "succeeded" and shopper and shopper.email and mailer:
                first = len(engine.events)
                sent = await mailer.receipt(shopper.email, shopper.name, detail, payment, f"receipt-{payment['id']}")
                payment["email"] = sent
                engine.note("email", person=pid, status=sent["status"], to=sent["to"], message=sent["message"])
                self._update(first)
        finally:
            if row is not None:  # now with the invoice and the email on it
                ledger.finish(row, payment, self._detail(pid, payment, admitted, door_s))
            # Settled: their pictures are in the ledger and the page has the ones the check was shown. A shopper who
            # comes back starts a fresh set; on a camera left running all day, old sets would pile up.
            self.evidence.pop(pid, None)

    # the ledger: what is kept about a walk-out

    @staticmethod
    def _held_back(line: dict) -> bool:
        """Was this something the shopper still had that was kept off the bill?"""
        return line["state"] == "in_cart" and not line["charge"] and (line["product"] is None or line["billable"] or line["unread"])

    @staticmethod
    def _why_held(line: dict) -> str:
        if line["audit"] == "put_back":
            return f"Returned to the shelf: {line['audit_why']}"
        if line["audit"] == "unclear":
            return f"Not confirmed at checkout: {line['audit_why']}"
        if line["verified"] == "disputed":
            return f"No item seen in hand: {line['why']}"
        if line["unread"]:
            return "Item not identified"
        if line["repeat"]:
            return "Duplicate of another line"
        if line["product"] is None:
            return "Not in the catalogue"
        if line.get("contested"):
            return f"Seen in hand as another product: {line['contested']}"
        if line["verified"] is None and line["billable"]:
            return "Identified on the shelf only, not confirmed in hand"
        return f"Low confidence ({round(line['confidence'] * 100)}%)"

    def _proofs(self, pid: int, gate: tuple | None = None) -> list[dict]:
        """The pictures behind this shopper's bill, in the order things happened: the gate camera when it let them
        in, the floor camera when it recognised them, each item leaving the shelf, then what the check before
        charging was shown."""
        out = []
        if gate and gate[0]:
            match = f", {round(gate[1] * 100)}% face match" if gate[1] else ", at registration"
            out.append({"kind": "gate", "caption": f"Entry gate{match}", "t": None, "jpeg": gate[0]})
        seen = self.face_shots.get(pid)
        if seen:
            match = f", {round(self.scores[pid] * 100)}% face match" if self.scores.get(pid) else ""
            out.append({"kind": "face", "caption": f"Store camera, shopper recognised{match}", "t": seen[0], "jpeg": seen[1]})
        for line in self.engine._cart(pid)["lines"]:
            shot = self.pick_shots.get(line["id"])
            if shot:
                how = {"hand": "first seen in hand", "carried_in": "in hand on arrival", "handover": "handed over"}.get(line["origin"], "taken from the shelf")
                what = "Unidentified item" if line["unread"] else line["name"]
                out.append({"kind": "pick", "caption": f"{what}, {how}", "t": shot[0], "jpeg": shot[1]})
        for jpeg, (t, caption) in zip(self.shown.get(pid, []), self.shown_notes.get(pid, [])):
            out.append({"kind": "check", "caption": caption, "t": t, "jpeg": jpeg})
        return out

    def _detail(self, pid: int, payment: dict, admitted: float | None, door_s: float) -> dict:
        """The steps and the numbers behind one walk-out, as the ledger keeps them."""
        engine = self.engine
        person = engine.people.people.get(pid)
        events = [e for e in engine.events if pid in (e.get("person"), e.get("from_person"))][-400:]
        seen = next((e for e in events if e["type"] == "checkin"), {})
        left = next((e for e in reversed(events) if e["type"] == "exit"), {})
        return {
            "person": pid,
            "live": self.live,
            "t0": self.wall0,  # the time of day at second 0 of the footage: an event's time is t0 + its t
            "admitted": admitted,
            "entered": "floor" if pid in self.walk_ins else "gate",
            "lines": engine._cart(pid)["lines"],
            "events": events,
            "payment": {k: payment.get(k) for k in ("status", "amount", "currency", "id", "order", "method", "message", "check", "invoice", "email")},
            "numbers": {
                "frames": self.frames_seen,
                "model_ms": round(self.model_ms / self.frames_seen) if self.frames_seen else None,
                "in_view_s": round(left.get("t", engine.t) - person.first_t, 1) if person else None,
                "face_match": seen.get("score"),
                "hand_looks": self.hand_asks.get(pid, (0, 0.0))[0],
                "item_looks": sum(len(e.looks) for e in engine.entries if e.person == pid),
                "leave_after_s": engine.cfg.exit_after_s,
                "left_how": left.get("how"),
                "door_s": door_s,
            },
        }

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

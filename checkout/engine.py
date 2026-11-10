"""
The checkout engine: turns per-frame SAM 3.1 masks into "who took what".

Each analysed frame goes through `Engine.step`:

 1. People     person masks -> shopper tracks with stable ids (people.py).
 2. Hands      each hand mask is given to the shopper whose mask it touches most.
 3. Items      catalogue detections and the generic "item held in hand" detections are fused into item
               observations; each one knows which hand masks touch it, so which shopper is holding it.
 4. Tracks     observations are matched to item tracks. A track is a candidate, at rest on a shelf, or held.
 5. Decisions  rest -> held   the item left its resting place while a shopper's hand was on it        (pick)
               new in a hand  an item first seen in a hand, held across several frames               (pick)
               rest vanished  a steady shelf item is gone, the spot is in view, a hand was just on it (inferred pick)
               held -> rest   the item sits still with no hand on it, on a shelf or away from the shopper (return)
               held, unseen   last seen being put on a shelf, and the shopper's hands are now empty   (inferred return)
               held by other  only another shopper's hands touch it for a while                       (hand-over)

Attribution never rests on one frame: every item keeps a decaying tally of hand contact per shopper, and a pick goes
to the shopper with the largest share. That share, the class votes and the amount of evidence make up the confidence
printed with every event. Distances are fractions of the frame diagonal and durations are seconds, so the logic
does not depend on resolution or on how many frames a second the GPU manages.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from . import people as ppl
from .catalog import HAND, HELD, PERSON, Catalog
from .geometry import Det, cover, dilate, drop_groups, grow, inter, iou, make_det, nms

CAND, REST, HELD_STATE, GONE = "candidate", "rest", "held", "gone"
_NO_MASK = np.zeros((1, 1), bool)  # what a gone track behind a cart line keeps instead of its mask


@dataclass
class Config:
    mask_w: int = 480
    # people
    person_min_score: float = 0.45
    person_birth_score: float = 0.75  # real shoppers score 0.8+; reflections in glass doors stay under 0.7
    person_min_area: float = 0.012  # share of the frame
    person_confirm_hits: int = 2
    person_confirm_s: float = 0.25
    person_drop_s: float = 0.8
    exit_after_s: float = 3.0
    reid_window_s: float = 45.0
    reid_min_sim: float = 0.72
    # hands
    hand_min_score: float = 0.35
    hand_reach: float = 0.022  # how far a hand mask may sit from its owner's mask
    hand_min_cover: float = 0.08
    hand_sure: float = 0.68  # below this share of the overlap, ownership is taken from the last frame
    hand_step: float = 0.22  # how far a hand may travel between two analysed frames
    # items
    item_min_score: float = 0.35
    item_weak_score: float = 0.25
    held_min_score: float = 0.35
    item_min_area: float = 0.0005
    item_max_area: float = 0.2
    contact_reach: float = 0.016
    contact_min: float = 0.08
    contact_memory_s: float = 1.5
    rest_after_s: float = 1.0
    rest_radius: float = 0.025
    stable_after_s: float = 2.0
    pick_move: float = 0.04
    pick_far: float = 0.10  # moved this far, an item counts as taken even if it still overlaps the shelf area
    pick_confirm_s: float = 0.3
    release_s: float = 1.0
    vanish_s: float = 1.5
    vanish_untouched_s: float = 5.0
    transfer_s: float = 1.2
    candidate_ttl_s: float = 0.7
    carried_in_s: float = 1.0
    # shelves
    zone_min_score: float = 0.35
    zone_reach: float = 0.035
    zone_max_area: float = 0.7
    # billing
    charge_min_confidence: float = 0.5


@dataclass
class Obs:
    """One item seen in one frame."""

    det: Det
    votes: dict  # product id -> score
    held_hint: float  # score of the generic "item held in hand" mask on it, 0 if none
    contact: dict = field(default_factory=dict)  # person id -> 0..1
    holder: int | None = None
    at_shelf: bool = False


@dataclass
class Item:
    id: int
    mask: np.ndarray
    box: tuple
    area: int
    c: np.ndarray
    polygons: list
    score: float
    born_t: float
    last_t: float
    born_c: np.ndarray
    born_at_shelf: bool
    state: str = CAND
    votes: dict = field(default_factory=dict)
    hits: int = 1
    visible: bool = True
    held_hint: float = 0.0
    holder: int | None = None
    contact: dict = field(default_factory=dict)  # decaying tally of hand contact per shopper
    path: deque = field(default_factory=lambda: deque(maxlen=24))
    # candidate / rest
    hold_run: tuple | None = None  # (since, frames) of unbroken holding
    anchor: np.ndarray | None = None
    rest_t: float | None = None
    move_run: int = 0
    clear_since: float | None = None  # unseen with its spot in view since
    # held
    entry: "Entry | None" = None
    free_since: float | None = None
    free_frames: int = 0
    other_run: tuple | None = None
    was_away: bool = False
    last_at_shelf: bool = False
    empty_hands: int = 0
    settled: str | None = None  # product id that reading or the second model's looks settled on
    read_how: str = ""  # the evidence for `settled`, in words
    read_in_hand: bool = False  # `settled` comes from reading the item in the hand, not from its shelf spot
    read_sure: bool = False  # `settled` rests on text read clearly, not on the second model's word alone
    likeness: dict | None = None  # how alike it looks to the catalogue's pictures: {product, score}

    @property
    def seen_as(self) -> str | None:
        """The product SAM's catalogue prompts favour."""
        return max(self.votes, key=self.votes.get) if self.votes else None

    @property
    def cls(self) -> str | None:
        """The product this item counts as: the second model's settled answer if there is one, else SAM's."""
        return self.settled or self.seen_as

    def take(self, o: Obs, t: float) -> None:
        d = o.det
        self.mask, self.box, self.area, self.c, self.polygons, self.score = d.mask, d.box, d.area, d.c, d.polygons, d.score
        self.last_t, self.hits, self.visible = t, self.hits + 1, True
        self.held_hint = max(self.held_hint, o.held_hint)
        self.path.append((t, d.c))
        self.clear_since, self.empty_hands = None, 0
        for cls, score in o.votes.items():
            self.votes[cls] = self.votes.get(cls, 0.0) + score
        for pid, value in o.contact.items():
            self.contact[pid] = self.contact.get(pid, 0.0) + value


@dataclass
class Entry:
    """One line of a shopper's cart."""

    id: int
    item: Item
    person: int
    picked_t: float
    origin: str  # shelf | hand | carried_in | inferred | handover
    share: float  # of the hand contact, at the moment of the pick
    confidence: float = 0.0
    state: str = "in_cart"  # in_cart | returned | given (handed to another shopper) | merged (a duplicate line, folded away)
    returned_t: float | None = None
    verified: str | None = None  # the second model's view: confirmed | disputed | unknown, None while unsettled
    why: str = ""
    looks: list = field(default_factory=list)  # (holding, product id or None) per look
    paid: str | None = None  # id of the payment that covered this line
    audit: str | None = None  # the check before charging: kept | put_back | unclear, None if it has not run
    audit_why: str = ""
    likeness: dict | None = None  # the best likeness to a catalogue picture: {product, name, score, why}


class Engine:
    def __init__(self, catalog: Catalog, cfg: Config | None = None):
        self.catalog = catalog
        self.cfg = cfg or Config()
        self.people = ppl.PeopleTracker(self.cfg)
        self.items: dict[int, Item] = {}
        self.entries: list[Entry] = []
        self.events: list[dict] = []
        self.next_item = 1
        self.next_event = 1
        self.size: tuple[int, int] | None = None
        self.diag = 1.0
        self.zone: np.ndarray | None = None  # 0..1, where shelves are
        self.zone_near: np.ndarray | None = None
        self.zone_dirty = False
        self.t = 0.0
        self.last_t: float | None = None
        self.hands: list[Det] = []
        self.frames = 0
        self.shelf_memory: list[dict] = []  # what was read at each shelf spot: {c, area, product, how}
        self.prev_gray: np.ndarray | None = None
        self.camera_moves = 0  # frames on which the background was found to have moved
        self.camera_moved_t = -1.0  # when it last moved by a pixel or more

    # ------------------------------------------------------------------ frame

    def step(self, t: float, result: dict, frame: np.ndarray | None = None) -> dict:
        """`result` is the endpoint's answer for this frame, `frame` the BGR picture (for clothing colours)."""
        cfg = self.cfg
        if self.size is None:
            iw, ih = result["image"]["w"], result["image"]["h"]
            w = cfg.mask_w
            self.size = (w, max(8, round(w * ih / iw)))
            self.diag = math.hypot(*self.size)
            self.zone = np.zeros(self.size[::-1], np.float32)
            self.zone_near = np.zeros(self.size[::-1], np.uint8)
        w, h = self.size
        dt = 0.0 if self.last_t is None else max(0.0, t - self.last_t)
        self.t, self.last_t, self.frames = t, t, self.frames + 1
        first_event = len(self.events)

        by_label: dict[str, list[Det]] = {}
        for raw in result.get("detections", []):
            d = make_det(raw, w, h)
            if d is not None:
                by_label.setdefault(d.label, []).append(d)

        small = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA) if frame is not None else None
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV) if small is not None else None
        if small is not None:
            self._follow_camera(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))
        for e in self.people.update(t, by_label.get(PERSON, []), hsv, self.diag, w * h):
            self._person_event(e)
        present = self.people.visible()

        self._update_zone(by_label)
        self.hands = self._own_hands(by_label.get(HAND, []), present)
        observations = self._observe(by_label, present)
        self._track(t, dt, observations, present)
        self._prune()

        return self.snapshot(self.events[first_event:])

    def _prune(self) -> None:
        """Forget tracks that are gone for good. Every outline SAM ever found starts one, so on a camera left running
        all day they would pile up, each with its mask. A gone track behind a cart line is kept, without its mask."""
        lines = {e.item.id for e in self.entries}
        for item_id in [i for i, item in self.items.items() if item.state == GONE]:
            item = self.items[item_id]
            if item_id not in lines:
                del self.items[item_id]
            elif item.mask.size > 1:
                item.mask, item.polygons = _NO_MASK, []

    # ------------------------------------------------------------------ a camera that moves

    def _follow_camera(self, gray: np.ndarray) -> None:
        """Work out how the background moved since the last frame and carry everything remembered along with it.

        Picks are decided by an item leaving its resting place, which only means something if resting places stay put.
        A hand-held or nudged camera moves every one of them at once. Corner points on the background (people are
        masked out, they move on their own) are followed from the last frame to this one, and the shift, turn and zoom
        that explains most of them is applied to every remembered position, mask and shelf area."""
        prev, self.prev_gray = self.prev_gray, gray
        if prev is None:
            return
        w, h = self.size
        background = np.full(prev.shape, 255, np.uint8)
        for person in self.people.people.values():
            if person.state in (ppl.ACTIVE, ppl.TENTATIVE, ppl.LOST):
                grown, where = grow(person.mask, person.box, 0.03 * self.diag)
                background[where][grown > 0] = 0
        points = cv2.goodFeaturesToTrack(prev, 300, 0.01, 7, mask=background)
        if points is None or len(points) < 20:
            return
        moved_to, found, _ = cv2.calcOpticalFlowPyrLK(prev, gray, points, None, winSize=(21, 21), maxLevel=3)
        good = found.ravel() == 1
        if good.sum() < 20:
            return
        m, inliers = cv2.estimateAffinePartial2D(points[good], moved_to[good], method=cv2.RANSAC, ransacReprojThreshold=1.5)
        if m is None or inliers.sum() < max(15, 0.4 * good.sum()):
            return
        corners = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        shift = float(np.abs(corners @ m[:, :2].T + m[:, 2] - corners).max())
        if shift < 0.5:
            return  # a camera on a mount: nothing to do
        self.camera_moves += 1
        if shift >= 1.0:
            self.camera_moved_t = self.t

        def point(c):
            return m[:, :2] @ c + m[:, 2]

        zoom = abs(float(np.linalg.det(m[:, :2])))

        def carry(thing) -> None:
            thing.mask = cv2.warpAffine(thing.mask, m, (w, h), flags=cv2.INTER_NEAREST)
            thing.c = point(thing.c)
            x0, y0, x1, y1 = thing.box
            moved = np.float32([[x0, y0], [x1, y0], [x0, y1], [x1, y1]]) @ m[:, :2].T + m[:, 2]
            x0, y0 = max(0, int(np.floor(moved[:, 0].min()))), max(0, int(np.floor(moved[:, 1].min())))
            x1, y1 = min(w, int(np.ceil(moved[:, 0].max()))), min(h, int(np.ceil(moved[:, 1].max())))
            if x1 > x0 and y1 > y0:
                thing.box, thing.area = (x0, y0, x1, y1), max(1, round(thing.area * zoom))
            else:  # carried out of the picture
                thing.box, thing.area = (0, 0, 1, 1), 1

        for item in self.items.values():
            if item.state == GONE:
                continue
            carry(item)
            item.born_c = point(item.born_c)
            if item.anchor is not None:
                item.anchor = point(item.anchor)
            item.path = deque(((ts, point(c)) for ts, c in item.path), maxlen=item.path.maxlen)
        for person in self.people.people.values():
            if person.state != ppl.EXITED:
                carry(person)
        for hand in self.hands:
            hand.c = point(hand.c)
        for memo in self.shelf_memory:
            memo["c"] = point(memo["c"])
        self.zone = cv2.warpAffine(self.zone, m, (w, h))
        self.zone_dirty = True

    # ------------------------------------------------------------------ shelves

    def _update_zone(self, by_label: dict) -> None:
        cfg, (w, h) = self.cfg, self.size
        seen = np.zeros((h, w), np.uint8)
        for label in self.catalog.zones:
            for d in by_label.get(label, []):
                if d.score >= cfg.zone_min_score and d.area <= cfg.zone_max_area * w * h:
                    seen |= d.mask
        if seen.any():
            self.zone = np.maximum(self.zone * 0.9, seen.astype(np.float32))
            self.zone_dirty = True
        if self.zone_dirty:
            # grown at a quarter of the size and blown back up: this map only says "roughly near a shelf"
            quarter = cv2.resize((self.zone > 0.4).astype(np.uint8), None, fx=0.25, fy=0.25, interpolation=cv2.INTER_NEAREST)
            self.zone_near = cv2.resize(dilate(quarter, cfg.zone_reach * self.diag / 4), self.size, interpolation=cv2.INTER_NEAREST)
            self.zone_dirty = False

    def _mark_shelf(self, item: Item) -> None:
        """A steady resting item is itself proof of a shelf."""
        np.maximum(self.zone, item.mask.astype(np.float32), out=self.zone)
        self.zone_dirty = True

    def _at_shelf(self, d, holder: ppl.Person | None) -> bool:
        if self.zone_near is not None and self.zone_near.any():
            x0, y0, x1, y1 = d.box
            return float(self.zone_near[y0:y1, x0:x1][d.mask[y0:y1, x0:x1] > 0].mean()) >= 0.3
        # no shelf found in this scene: an arm stretched well away from the body counts as reaching for one
        if holder is None:
            return False
        half = (holder.box[2] - holder.box[0]) / 2
        return abs(d.c[0] - holder.c[0]) > 0.9 * half

    # ------------------------------------------------------------------ hands

    def _own_hands(self, dets: list[Det], present: list[ppl.Person]) -> list[Det]:
        """Give each hand mask to a shopper.

        A hand belongs to the person mask it lies in. Where one shopper reaches across another the hand can lie in
        both; a hand does not change owner between two frames, so the tie goes to whoever it belonged to a moment ago.
        """
        cfg = self.cfg
        hands = nms([d for d in dets if d.score >= cfg.hand_min_score], 0.5, 0.8)
        before = [hand for hand in self.hands if hand.extra.get("person") is not None]
        was: dict[int, Det] = {}
        if before and hands:
            dist = np.array([[float(np.linalg.norm(b.c - hand.c)) for b in before] for hand in hands]) / self.diag
            for i, k in zip(*linear_sum_assignment(dist)):
                if dist[i, k] <= cfg.hand_step:
                    was[i] = before[k]
        for i, hand in enumerate(hands):
            grown, where = grow(hand.mask, hand.box, cfg.hand_reach * self.diag)
            total = max(1, int(cv2.countNonZero(grown)))
            shares = {}
            for p in present:
                share = 0.7 * cover(hand, p) + 0.3 * int(cv2.countNonZero(grown & p.mask[where])) / total
                if share >= cfg.hand_min_cover:
                    shares[p.id] = share
            if not shares:
                continue
            ranked = sorted(shares, key=shares.get, reverse=True)
            best = shares[ranked[0]]
            owner, sure = ranked[0], best / (best + (shares[ranked[1]] if len(ranked) > 1 else 0.0))
            prev = was.get(i)
            if sure < cfg.hand_sure and prev is not None and prev.extra["person"] in shares:
                owner = prev.extra["person"]
                sure = max(shares[owner] / (best + shares[owner]), 0.5 + 0.8 * (prev.extra["sure"] - 0.5))
            hand.extra["person"], hand.extra["sure"] = owner, sure
        return hands

    # ------------------------------------------------------------------ items in this frame

    def _observe(self, by_label: dict, present: list[ppl.Person]) -> list[Obs]:
        cfg, (w, h) = self.cfg, self.size
        lo, hi = cfg.item_min_area * w * h, cfg.item_max_area * w * h

        def usable(d: Det, floor: float) -> bool:
            if d.score < floor or not lo <= d.area <= hi:
                return False
            return not any(iou(d, p) > 0.5 for p in present) and not any(iou(d, hand) > 0.5 for hand in self.hands)

        # Catalogue detections; where two products claim one object, both votes are kept on the better mask.
        # A phrase that outlines its own product clearly often outlines the item beside it too, faintly: in one
        # shopper's two hands a power bank's phrase scored 0.84 on the power bank and 0.36 on the tissue pack, whose
        # own phrase scored 0.30. So a vote counts for less the weaker it is than that phrase's best in the frame.
        # Two of the same product score alike and keep their votes; an item no other phrase claims keeps its name.
        named: list[Det] = []
        for prompt, product in self.catalog.by_prompt.items():
            found = nms(drop_groups([d for d in by_label.get(prompt, []) if usable(d, cfg.item_weak_score)]))
            top = max((d.score for d in found), default=1.0)
            for d in found:
                d.extra["votes"] = {product.id: d.score * min(1.0, d.score / top)}
                named.append(d)
        merged: list[Det] = []
        for d in sorted(named, key=lambda d: -d.score):
            for k in merged:
                i = inter(d, k)
                if i and (i / (d.area + k.area - i) > 0.5 or i / min(d.area, k.area) > 0.8):
                    for cls, score in d.extra["votes"].items():
                        k.extra["votes"][cls] = max(k.extra["votes"].get(cls, 0.0), score)
                    break
            else:
                merged.append(d)

        observations = [Obs(d, d.extra["votes"], 0.0) for d in merged]
        for g in nms([d for d in by_label.get(HELD, []) if usable(d, cfg.held_min_score)]):
            best, best_iou = None, 0.35
            for o in observations:
                i = inter(g, o.det)
                share = max(i / (g.area + o.det.area - i), 0.8 * i / min(g.area, o.det.area)) if i else 0.0
                if share > best_iou:
                    best, best_iou = o, share
            if best:
                best.held_hint = max(best.held_hint, g.score)
            else:
                observations.append(Obs(g, {}, g.score))  # something in a hand that the catalogue does not name
        # a faint catalogue match stands only where the held-item mask agrees that something is there
        observations = [o for o in observations if o.held_hint or o.det.score >= cfg.item_min_score]

        by_id = {p.id: p for p in present}
        for o in observations:
            grown, where = grow(o.det.mask, o.det.box, cfg.contact_reach * self.diag)
            for hand in self.hands:
                pid = hand.extra.get("person")
                if pid is None:
                    continue
                touch = int(cv2.countNonZero(grown & hand.mask[where])) / hand.area
                if touch >= cfg.contact_min:
                    o.contact[pid] = max(o.contact.get(pid, 0.0), min(1.0, touch * 2) * hand.extra["sure"])
            if not o.contact and o.held_hint >= 0.5:
                # SAM says it is held but no hand mask came back: fall back to the body it overlaps
                grown_area = max(1, int(cv2.countNonZero(grown)))
                for p in present:
                    if int(cv2.countNonZero(grown & p.mask[where])) / grown_area >= 0.15:
                        o.contact[p.id] = 0.4
            if o.contact:
                o.holder = max(o.contact, key=o.contact.get)
            o.at_shelf = self._at_shelf(o.det, by_id.get(o.holder))
        return observations

    # ------------------------------------------------------------------ item tracks

    def _similar(self, item: Item, o: Obs, t: float) -> float:
        dist = float(np.linalg.norm(item.c - o.det.c)) / self.diag
        overlap = iou(item, o.det)
        carried = item.state == HELD_STATE or (item.state == CAND and item.hold_run)
        owner = item.holder if item.state == HELD_STATE else self._owner(item, 0.3)
        if owner is not None and owner in o.contact:
            reach = 0.35  # in the hand that holds it, or that was just on it: it goes wherever that hand goes
            sim = 0.4 + 0.6 * max(overlap, 1 - dist / reach)
        elif carried:
            # A carried item whose holder's hand is not on this observation: only its own outline continuing counts.
            # Nearness alone would let the track hop onto whatever sits close by the moment a hand mask is missed.
            return overlap if overlap >= 0.25 else 0.0
        elif o.holder is not None:
            reach = 0.15  # a resting item that a hand has just moved
            sim = max(overlap, 1 - dist / reach)
        else:
            reach = 0.06
            sim = max(overlap, 1 - dist / reach)
        if dist > reach and overlap < 0.2:
            return 0.0
        theirs = max(o.votes, key=o.votes.get) if o.votes else None
        if not self.catalog.same_kind(item.cls, theirs):
            sim *= 0.6
        return sim

    def _track(self, t: float, dt: float, observations: list[Obs], present: list[ppl.Person]) -> None:
        cfg = self.cfg
        fade = math.exp(-dt / cfg.contact_memory_s) if dt else 1.0
        tracks = [i for i in self.items.values() if i.state != GONE]
        for item in tracks:
            if not item.visible:
                continue  # the tally is what was known when the item was last seen
            for pid in list(item.contact):
                item.contact[pid] *= fade
                if item.contact[pid] < 0.02:
                    del item.contact[pid]

        # tracks seen a moment ago get first claim on what is in view; only what they leave over can be an item
        # coming back out of a bag or from behind someone
        matched: dict[int, Obs] = {}
        used: set[int] = set()
        for group in ([i for i in tracks if i.visible], [i for i in tracks if not i.visible]):
            free = [j for j in range(len(observations)) if j not in used]
            if not group or not free:
                continue
            sim = np.array([[self._similar(item, observations[j], t) for j in free] for item in group])
            for i, k in zip(*linear_sum_assignment(1 - sim)):
                if sim[i, k] > 0.15:
                    matched[group[i].id] = observations[free[k]]
                    used.add(free[k])

        by_id = {p.id: p for p in present}
        hands_of = {pid: 0 for pid in by_id}
        for hand in self.hands:
            if hand.extra.get("person") in hands_of:
                hands_of[hand.extra["person"]] += 1
        # shoppers with a hand on anything at all this frame: their hands are not empty
        busy = {pid for o in observations for pid in o.contact}

        for item in tracks:
            o = matched.get(item.id)
            if o is not None:
                item.take(o, t)
                if item.state == CAND:
                    self._candidate(item, o, t, by_id)
                elif item.state == REST:
                    self._resting(item, o, t)
                elif item.state == HELD_STATE:
                    self._held(item, o, t, by_id)
            else:
                item.visible = False
                self._unseen(item, t, present, hands_of, busy)

        for j, o in enumerate(observations):
            if j in used:
                continue
            d = o.det
            item = Item(self.next_item, d.mask, d.box, d.area, d.c, d.polygons, d.score, t, t, d.c, o.at_shelf)
            item.votes, item.held_hint, item.contact = dict(o.votes), o.held_hint, dict(o.contact)
            item.path.append((t, d.c))
            if o.holder is not None:
                item.hold_run = (t, 1)
            self.items[item.id] = item
            self.next_item += 1

    def _owner(self, item: Item, floor: float = 0.0) -> int | None:
        """The shopper with the most hand contact on this item lately, not just in this frame."""
        if not item.contact:
            return None
        pid = max(item.contact, key=item.contact.get)
        return pid if item.contact[pid] >= floor else None

    def _still(self, item: Item, since: float) -> bool:
        pts = [c for (ts, c) in item.path if ts >= since]
        if len(pts) < 2:
            return False
        pts = np.array(pts)
        return float(np.linalg.norm(pts - pts.mean(0), axis=1).max()) <= self.cfg.rest_radius * self.diag

    def _candidate(self, item: Item, o: Obs, t: float, by_id: dict) -> None:
        cfg = self.cfg
        if o.holder is not None:
            run = item.hold_run
            item.hold_run = (run[0], run[1] + 1) if run else (t, 1)
            since, frames = item.hold_run
            moved = float(np.linalg.norm(item.c - item.born_c)) / self.diag
            # on the shelf with a hand on it is browsing; it has to come away (or be carried a good distance) to count
            away = not o.at_shelf or moved >= cfg.pick_far
            if frames >= 2 and t - since >= cfg.pick_confirm_s and away and (item.held_hint >= 0.4 or moved >= cfg.pick_move):
                pid = self._owner(item)
                person = by_id.get(pid)
                early = person is not None and item.born_t - person.first_t <= cfg.carried_in_s
                self._pick(item, pid, "carried_in" if early else "hand", o.at_shelf)
            return
        item.hold_run = None
        if t - item.born_t >= cfg.rest_after_s and item.hits >= 3 and self._still(item, t - cfg.rest_after_s):
            item.state, item.anchor, item.rest_t = REST, item.c.copy(), t
            self._recall(item)

    def _resting(self, item: Item, o: Obs, t: float) -> None:
        cfg = self.cfg
        moved = float(np.linalg.norm(item.c - item.anchor)) / self.diag
        if moved < cfg.rest_radius:
            item.anchor = 0.9 * item.anchor + 0.1 * item.c
            item.move_run = 0
            if t - item.rest_t >= cfg.stable_after_s and item.hits % 4 == 0:
                self._mark_shelf(item)
            return
        if moved < max(cfg.pick_move, 0.5 * math.sqrt(item.area) / self.diag):
            return
        pid = self._owner(item, 0.0 if o.contact else 0.3)
        if pid is None:
            # it moved with nobody near: jitter, or the camera was nudged. Settle at the new place.
            item.move_run += 1
            if item.move_run >= 4:
                item.anchor, item.move_run = item.c.copy(), 0
            return
        if o.at_shelf and moved < cfg.pick_far:
            # handled, tilted or nudged along the row, but still on the shelf: not taken yet. On a tight shelf the
            # outlines of neighbours also trade places a little whenever a hand is among them.
            return
        item.move_run += 1
        if item.move_run >= 2 or o.held_hint >= 0.4:
            self._pick(item, pid, "shelf", o.at_shelf)

    def _held(self, item: Item, o: Obs, t: float, by_id: dict) -> None:
        cfg = self.cfg
        item.last_at_shelf = o.at_shelf
        if not o.at_shelf:
            item.was_away = True
        if o.holder == item.holder:
            item.free_since, item.free_frames, item.other_run = None, 0, None
            return
        if o.holder is None:
            item.other_run = None
            owner = by_id.get(item.holder)
            hands_in_view = any(hand.extra.get("person") == item.holder for hand in self.hands)
            against = owner is not None and cover(item, _Grown(owner, dilate(owner.mask, cfg.hand_reach * self.diag, owner.box))) >= 0.2
            # Letting go has to be seen, not assumed from a hand mask that failed to come back for a frame: the
            # holder's hands are in view and none is on the item, SAM does not call it held, and it is not up
            # against the holder's arm or body (a book held in front of a shelf overlaps the shelf in the picture).
            if o.held_hint >= 0.4 or against or (owner is not None and not hands_in_view):
                item.free_since, item.free_frames = None, 0
                return
            item.free_since = t if item.free_since is None else item.free_since
            item.free_frames += 1
            if item.free_frames >= 2 and t - item.free_since >= cfg.release_s and self._still(item, item.free_since):
                self._return(item, "put back on the shelf" if o.at_shelf else "put down and left", 0.9)
            return
        # in someone else's hand, and not in the holder's
        if item.holder in o.contact:
            item.other_run = None
            return
        run = item.other_run
        item.other_run = (o.holder, run[1], run[2] + 1) if run and run[0] == o.holder else (o.holder, t, 1)
        pid, since, frames = item.other_run
        if frames >= 3 and t - since >= cfg.transfer_s:
            self._transfer(item, pid)

    def _unseen(self, item: Item, t: float, present: list[ppl.Person], hands_of: dict, busy: set) -> None:
        cfg = self.cfg
        if item.state == CAND:
            if t - item.last_t > cfg.candidate_ttl_s:
                item.state = GONE
        elif item.state == REST:
            covering = [hand for hand in self.hands if inter(item, hand)]
            for hand in covering:
                pid = hand.extra.get("person")
                if pid is not None:  # a hand closed over it: the strongest contact there is
                    item.contact[pid] = item.contact.get(pid, 0.0) + 0.6 * hand.extra["sure"]
            if covering or any(cover(item, p) >= 0.35 for p in present):
                item.clear_since = None
                return
            item.clear_since = t if item.clear_since is None else item.clear_since
            # with no hand anywhere near it lately, a missing detection is far more likely a flicker than a theft
            if t - item.clear_since < (cfg.vanish_s if item.contact else cfg.vanish_untouched_s):
                return
            steady = item.rest_t is not None and item.last_t - item.rest_t >= cfg.stable_after_s
            pid = self._owner(item, 0.25)
            # "it was there, now it is not" only holds for a camera that has not moved in between
            if steady and pid is not None and self.camera_moved_t < item.last_t:
                self._pick(item, pid, "inferred", True)
            else:
                item.state = GONE
        elif item.state == HELD_STATE:
            if item.holder in hands_of and hands_of[item.holder] and item.holder not in busy:
                item.empty_hands += 1
            if item.last_at_shelf and item.was_away and item.empty_hands >= 3 and t - item.last_t >= cfg.vanish_s:
                self._return(item, "hand came back empty from the shelf", 0.6)

    # ------------------------------------------------------------------ decisions

    def _share(self, item: Item, pid: int) -> float:
        """How much of the recent hand contact on this item was this shopper's."""
        total = sum(item.contact.values())
        return item.contact.get(pid, 0.0) / total if total else 0.5

    def _confidence(self, entry: Entry) -> float:
        """Whose it is (share of hand contact) x how often it was seen x how clearly the catalogue names it.
        Recomputed as evidence comes in, so a pick that starts unsure firms up over the next frames."""
        item = entry.item
        votes = sum(item.votes.values())
        named = (max(item.votes.values()) / votes) * min(1.0, max(item.votes.values()) / 0.8) if votes else 0.0
        seen = min(1.0, item.hits / 4)
        conf = entry.share * (0.55 + 0.45 * seen) * (0.6 + 0.4 * named)
        return round(conf * (0.7 if entry.origin == "inferred" else 1.0), 2)

    def _pick(self, item: Item, pid: int, origin: str, at_shelf: bool) -> None:
        t = self.t
        if origin in ("hand", "carried_in"):
            # the same shopper already has such an item that dropped out of sight: usually this is that item again
            for other in sorted(self.items.values(), key=lambda i: -i.last_t):
                if other is item or other.state != HELD_STATE or other.holder != pid or other.visible:
                    continue
                if not self.catalog.same_kind(item.cls, other.cls):
                    continue
                fresh_pick = item.born_at_shelf and other.was_away and item.born_t - other.last_t >= 1.5 and other.entry.origin != "inferred"
                if not fresh_pick:
                    self._absorb(other, item)
                    return
        if origin == "inferred":
            # the shopper may already be holding it under a track that started in their hand
            for other in self.items.values():
                same = self.catalog.same_kind(item.cls, other.cls)
                if other.state == HELD_STATE and other.holder == pid and same and other.entry.origin in ("hand", "carried_in") and other.entry.picked_t >= item.last_t - 1.0:
                    self._inherit(other, item)  # what was read off it on the shelf still holds in the hand
                    item.state = GONE
                    return
        if item.state == REST:
            self._forget_spot(item)
        item.state, item.holder = HELD_STATE, pid
        item.free_since, item.free_frames, item.other_run = None, 0, None
        item.was_away, item.last_at_shelf, item.empty_hands = not at_shelf, at_shelf, 0
        entry = Entry(len(self.entries) + 1, item, pid, t, origin, self._share(item, pid))
        entry.confidence = self._confidence(entry)
        item.entry = entry
        self.entries.append(entry)
        if item.settled:
            mine = any(e is not entry and e.person == pid and e.state == "in_cart" and e.item.settled == item.settled for e in self.entries)
            if mine:
                # the shopper already holds that very product: the name this track brought from the shelf is wrong
                item.settled, item.read_how, item.read_sure = None, "", False
            else:
                # The name was read while the item stood on the shelf. On a tight shelf the outline that follows a
                # hand out is not always the one that was read, so the name is shown but waits to be read again in
                # the hand before it is billed.
                entry.why = item.read_how or "Identified on the shelf before it was taken"
        self._emit("pick", person=pid, entry=entry.id, item=item.id, origin=origin, confidence=entry.confidence)

    def _absorb(self, kept: Item, seen: Item) -> None:
        """`seen` is `kept` coming back into view: carry on with the old track and its cart line."""
        kept.mask, kept.box, kept.area, kept.c, kept.polygons, kept.score = seen.mask, seen.box, seen.area, seen.c, seen.polygons, seen.score
        kept.last_t, kept.visible, kept.hits = seen.last_t, True, kept.hits + seen.hits
        kept.path.extend(seen.path)
        for cls, score in seen.votes.items():
            kept.votes[cls] = kept.votes.get(cls, 0.0) + score
        for pid, value in seen.contact.items():
            kept.contact[pid] = kept.contact.get(pid, 0.0) + value
        kept.free_since, kept.free_frames, kept.empty_hands = None, 0, 0
        self._inherit(kept, seen)
        seen.state = GONE

    def _inherit(self, kept: Item, other: Item) -> None:
        """Two tracks turned out to be one item: keep what was already worked out about which product it is."""
        if other.settled and not kept.settled:
            kept.settled, kept.read_how, kept.read_sure = other.settled, other.read_how, other.read_sure
            if kept.entry and kept.entry.verified is None:
                kept.entry.why = other.read_how or "Identified on the shelf before it was taken"

    def _return(self, item: Item, how: str, confidence: float) -> None:
        entry = item.entry
        if not entry.paid:  # a line settled at an earlier walk-out stays as it was charged: undoing that is a refund
            entry.state, entry.returned_t = "returned", self.t
            self._emit("return", person=entry.person, entry=entry.id, item=item.id, how=how, confidence=confidence)
        item.entry, item.holder = None, None
        if item.visible:
            item.state, item.anchor, item.rest_t, item.move_run = REST, item.c.copy(), self.t, 0
            item.contact.clear()
        else:
            item.state = GONE

    def _transfer(self, item: Item, pid: int) -> None:
        old = item.entry
        old.state, old.returned_t = "given", self.t
        entry = Entry(len(self.entries) + 1, item, pid, self.t, "handover", self._share(item, pid))
        entry.confidence = self._confidence(entry)
        self.entries.append(entry)
        self._emit("handover", person=pid, from_person=old.person, entry=entry.id, item=item.id, confidence=entry.confidence)
        item.entry, item.holder, item.other_run = entry, pid, None

    def _person_event(self, e: dict) -> None:
        if e["type"] == "exit":
            for item in self.items.values():
                if item.state == HELD_STATE and item.holder == e["person"]:
                    item.state = GONE  # it left with the shopper; the cart line stays
            # how: "out_of_view" (unseen for exit_after_s) or "ended" (the footage stopped with them still inside)
            self._emit("exit", person=e["person"], total=self._cart(e["person"])["total"], how=e.get("how", "out_of_view"), after_s=self.cfg.exit_after_s)
        else:
            self._emit(e["type"], person=e["person"])

    def same_person(self, new: int, old: int) -> bool:
        """Their face says the person tracked as `new` is the shopper tracked before as `old`, whose outline was
        lost for a while (they turned, were hidden, or stepped out long enough to count as having left). `new` is
        folded into `old`: one shopper, one bag.

        Nothing they carried through the gap may be billed twice. So a line of `new` for something first seen
        already in the hand (not seen leaving a shelf) joins the line `old` has for that kind of item, paid or
        not, instead of standing beside it. What was seen leaving a shelf after the gap is a new line."""
        people = self.people.people
        a, b = people.get(old), people.get(new)
        if a is None or b is None or a is b or (a.seen and a.state == ppl.ACTIVE):
            return False
        a.mask, a.box, a.area, a.c, a.polygons, a.score = b.mask, b.box, b.area, b.c, b.polygons, b.score
        a.hist = b.hist if b.hist is not None else a.hist
        a.last_t, a.seen, a.hits, a.state, a.exit_t = b.last_t, b.seen, a.hits + b.hits, ppl.ACTIVE, None
        del people[new]
        for item in self.items.values():
            if item.holder == new:
                item.holder = old
            if new in item.contact:
                item.contact[old] = item.contact.get(old, 0.0) + item.contact.pop(new)
            if item.other_run and item.other_run[0] == new:
                item.other_run = (old, *item.other_run[1:])
        had = [e for e in self.entries if e.person == old and e.state == "in_cart"]
        for entry in self.entries:
            if entry.person != new:
                continue
            entry.person = old
            if entry.state != "in_cart" or entry.origin not in ("hand", "carried_in", "seen"):
                continue
            item = entry.item
            twin = next((e for e in had if e.item is not item and not (e.item.visible and e.item.state == HELD_STATE) and self.catalog.same_kind(e.item.cls, item.cls)), None)
            if twin is None:
                continue
            had.remove(twin)
            entry.state = "merged"
            self._emit("void", person=old, entry=entry.id, into=twin.id)
            if twin.item.settled and not self.catalog.same_kind(item.settled, twin.item.settled):
                item.settled = None
            if twin.item.settled and not item.settled:
                item.settled, item.read_how, item.read_in_hand, item.read_sure = twin.item.settled, twin.item.read_how, twin.item.read_in_hand, twin.item.read_sure
            twin.item.state, twin.item.entry = GONE, None
            twin.item, item.entry = item, twin  # the outline in view carries the line they already had
        for hand in self.hands:
            if hand.extra.get("person") == new:
                hand.extra["person"] = old
        for event in self.events:  # what happened to `new` happened to this shopper
            if event.get("person") == new:
                event["person"] = old
                if event["type"] == "enter":
                    event["type"] = "reenter"
            if event.get("from_person") == new:
                event["from_person"] = old
        self._emit("same", person=old, was=new)
        return True

    def note(self, kind: str, **data) -> None:
        """Add an event from outside the frame loop (check-in, second look, payment) to the same timeline."""
        self._emit(kind, **data)

    def _emit(self, kind: str, **data) -> None:
        self.events.append({"id": self.next_event, "t": round(self.t, 2), "type": kind, **data})
        self.next_event += 1

    # ------------------------------------------------------------------ output

    def _chargeable(self, entry: Entry) -> bool:
        """Billed only when the evidence holds up; anything doubtful is kept back for a person to look at."""
        product = self.catalog.by_id.get(entry.item.cls)
        if entry.state != "in_cart" or product is None or not product.billable or entry.verified == "disputed":
            return False
        if entry.paid:
            return True
        if entry.audit in ("put_back", "unclear"):
            return False  # the last check before charging did not see them leave with it
        if product.family:
            # which member of a family it is rests on reading it: no confirmed reading, no bill. And one bill per
            # title per shopper: a second line with the same title is the same item seen under another track.
            return entry.verified == "confirmed" and not self._repeat(entry)
        if entry.verified == "confirmed":
            return True
        # not confirmed: SAM's phrase alone may name it, unless a look in the hand named another product
        return self._confidence(entry) >= self.cfg.charge_min_confidence and not self._contested(entry)

    def _repeat(self, entry: Entry) -> bool:
        """Is there an earlier line in this shopper's cart for the very same family product?"""
        title = entry.item.settled
        return title is not None and any(
            o.id < entry.id and o.person == entry.person and o.state == "in_cart" and o.verified == "confirmed" and o.item.settled == title for o in self.entries
        )

    def _line(self, entry: Entry) -> dict:
        product = self.catalog.by_id.get(entry.item.cls)
        if entry.state == "in_cart" and not entry.paid:
            entry.confidence = self._confidence(entry)
        unread = bool(product and product.head)  # "a book", title not worked out yet
        return {
            "unread": unread,
            "repeat": bool(product and product.family and entry.state == "in_cart" and self._repeat(entry)),
            "charge": self._chargeable(entry),
            "verified": entry.verified,
            "why": entry.why,
            "paid": entry.paid,
            "audit": entry.audit,
            "audit_why": entry.audit_why,
            "id": entry.id,
            "item": entry.item.id,
            "product": product.id if product else None,
            "name": product.name if product else "Unrecognised item",
            "price": product.price if product else None,
            "billable": bool(product and product.billable),
            "category": product.category if product else "",
            "section": product.section if product else "",
            "state": entry.state,
            "origin": entry.origin,
            "confidence": entry.confidence,
            "picked_t": round(entry.picked_t, 2),
            "returned_t": None if entry.returned_t is None else round(entry.returned_t, 2),
            "in_view": entry.item.visible and entry.item.state == HELD_STATE and entry.state == "in_cart",
            "likeness": entry.likeness,
            # held back because a look in the hand named this other product (the name of it), else ""
            "contested": self.catalog.by_id[other].name if entry.verified != "confirmed" and (other := self._contested(entry)) in self.catalog.by_id else "",
        }

    def _cart(self, pid: int) -> dict:
        lines = [self._line(e) for e in self.entries if e.person == pid and e.state != "merged"]
        kept = [line for line in lines if line["state"] == "in_cart"]
        person = self.people.people.get(pid)
        return {
            "person": pid,
            "state": "left" if person is None or person.state == ppl.EXITED else "shopping",
            "since": round(person.first_t, 2) if person else None,
            "lines": lines,
            "count": sum(1 for line in kept if line["charge"]),
            "total": round(sum(line["price"] for line in kept if line["charge"]), 2),
            "due": round(sum(line["price"] for line in kept if line["charge"] and not line["paid"]), 2),
            # in the cart but not billed: not in the catalogue, weak evidence, or the second model disagreed
            "review": sum(1 for line in kept if not line["charge"] and (line["product"] is None or line["billable"] or line["unread"])),
            "held_back": round(sum(line["price"] or 0 for line in kept if not line["charge"] and line["billable"]), 2),
        }

    # ------------------------------------------------------------------ outside checks and payment

    def _spot(self, item: Item) -> dict | None:
        """The remembered reading for the shelf spot this item sits at, if there is one."""
        at = item.anchor if item.anchor is not None else item.c
        for memo in self.shelf_memory:
            near = float(np.linalg.norm(memo["c"] - at)) <= self.cfg.rest_radius * self.diag
            if near and 0.6 <= item.area / memo["area"] <= 1.6:
                return memo
        return None

    def _recall(self, item: Item) -> None:
        """A track that starts at a spot whose item was already read is that same item seen again (its outline
        flickered out for a while): the reading still holds."""
        memo = self._spot(item)
        if memo and not item.settled and self.catalog.same_kind(item.seen_as, memo["product"]):
            item.settled, item.read_how, item.read_sure = memo["product"], memo["how"], memo["sure"]

    def _forget_spot(self, item: Item) -> None:
        """The item left its spot: whatever stands there next has to be read afresh."""
        memo = self._spot(item)
        if memo:
            self.shelf_memory = [m for m in self.shelf_memory if m is not memo]

    def settle(self, item_id: int, product_id: str, how: str, sure: bool = True) -> bool:
        """Something outside the frame loop worked out which product an item is, such as the title read off a book
        still on its shelf. Only a member of the family SAM saw the item as is accepted. `sure` is False when the
        name is only the second model's reading, with no clear text behind it."""
        item = self.items.get(item_id)
        product = self.catalog.by_id.get(product_id)
        if item is None or item.state == GONE or product is None or product.head or item.settled:
            return False
        if not self.catalog.same_kind(item.seen_as, product_id):
            return False
        item.settled, item.read_how, item.read_sure = product_id, how[:160], sure
        if item.state == REST:
            self._forget_spot(item)
            self.shelf_memory.append({"c": item.anchor.copy(), "area": item.area, "product": product_id, "how": item.read_how, "sure": sure})
        if item.entry and item.entry.verified is None:
            item.entry.why = item.read_how  # shown on the line; billing still waits for a reading in the hand
        return True

    def _renamable(self, seen_as: str | None, product_id: str) -> bool:
        """May the looks give an item another product than SAM's phrase said? For products outside a family (and
        shoppers' own things), yes: each has its own phrase, and a phrase can outline the item next to its product.
        Not into or out of a family (books): which title it is rests on reading it."""
        product = self.catalog.by_id.get(product_id)
        return product is not None and not product.head and product.family is None and self.catalog.family_of(seen_as) is None

    def _contested(self, entry: Entry) -> str | None:
        """The other product a look at it in the hand named, if one named another product than the one this line
        would be billed as."""
        cls = entry.item.cls
        return next((p for h, p, _, only in entry.looks if h and p and p != cls and not only), None)

    def name_held(self, item_id: int, product_id: str, why: str, overrule: bool = False) -> None:
        """A held item was clearly identified in the hand (its text read, or the second model's looks agreeing).

        Tracking a small item in a moving hand is the least reliable part of all this: a track can jump from one book
        to the next in a stack. What is read in the hand is far more reliable, so the cart follows the readings:

        same product as before        nothing changes
        the track had no name yet     it gets this one
        its name came from the shelf  the reading in the hand wins (the track followed the wrong spine out)
        the shopper already has a line for this product on another track
                                      this track is that item again; an unnamed duplicate line is folded into it
                                      (so two copies of one title in the same hands are billed once)
        it had another name, also read in the hand
                                      the track has moved on to a different item. The old line stays in the cart
                                      on a hidden track, and this one starts a new line.
        """
        item = self.items.get(item_id)
        product = self.catalog.by_id.get(product_id)
        if item is None or item.state != HELD_STATE or item.entry is None or product is None or product.head:
            return
        if not overrule and not self.catalog.same_kind(item.seen_as, product_id):
            return  # `overrule`: the looks agree it is another product than SAM's phrase said (apply_look)
        entry, pid = item.entry, item.entry.person
        if entry.paid:
            return

        def confirm(e: Entry) -> None:
            changed = (e.verified, e.item.settled) != ("confirmed", product_id)
            e.item.settled, e.item.read_how, e.item.read_in_hand, e.item.read_sure = product_id, why[:160], True, True
            e.verified, e.why = "confirmed", why[:160]
            if changed:
                self._emit("verify", person=pid, entry=e.id, item=e.item.id, verdict="confirmed", why=e.why, looks=len(e.looks))

        if item.settled == product_id:
            confirm(entry)
            return
        twin = next((e for e in self.entries if e is not entry and e.person == pid and e.state == "in_cart" and e.item.settled == product_id), None)
        if twin is not None and twin.item.visible and twin.item is not item:
            # Both tracks are in view and read the same: two outlines of one book, or a stack whose top book's
            # title shows inside the outline of the one beneath. Either way this reading says nothing about this
            # track, so it keeps whatever name it had.
            return
        if twin is not None:
            if item.settled is None or not item.read_in_hand:
                # the same item under a second track: one line, carried on by the track that is in view
                twin.item.state, twin.item.entry = GONE, None
                entry.state = "merged"
                self._emit("void", person=pid, entry=entry.id, into=twin.id)
                twin.item, item.entry = item, twin
                item.was_away = True
                confirm(twin)
            return
        if item.settled is None or not item.read_in_hand:
            confirm(entry)
            return
        # another product, and the old name was also read in the hand: two different items
        ghost = Item(self.next_item, item.mask, item.box, item.area, item.c, item.polygons, item.score, item.born_t, item.last_t, item.born_c, item.born_at_shelf)
        self.next_item += 1
        ghost.state, ghost.holder, ghost.visible, ghost.was_away = HELD_STATE, pid, False, True
        ghost.votes, ghost.settled, ghost.read_how, ghost.read_in_hand, ghost.hits = dict(item.votes), item.settled, item.read_how, True, item.hits
        ghost.entry, entry.item = entry, ghost
        self.items[ghost.id] = ghost
        fresh = Entry(len(self.entries) + 1, item, pid, self.t, "hand", self._share(item, pid))
        item.entry, item.settled = fresh, None
        self.entries.append(fresh)
        fresh.confidence = self._confidence(fresh)
        self._emit("pick", person=pid, entry=fresh.id, item=item.id, origin="hand", confidence=fresh.confidence)
        confirm(fresh)

    def name_line(self, pid: int, product_id: str, why: str) -> bool:
        """The second model, shown the shopper's hands, read a title on more than one look. A line of theirs that
        carries that title only provisionally (a name brought from the shelf) is confirmed by it; otherwise a pick
        that has no confirmed name yet takes it. It never adds a line."""
        product = self.catalog.by_id.get(product_id)
        if product is None or product.head or not product.family:
            return False
        mine = [e for e in self.entries if e.person == pid and e.state == "in_cart" and not e.paid]

        def confirm(entry: Entry) -> bool:
            entry.item.settled, entry.item.read_how, entry.item.read_in_hand, entry.item.read_sure = product_id, why[:160], True, True
            entry.verified, entry.why = "confirmed", why[:160]
            self._emit("verify", person=pid, entry=entry.id, item=entry.item.id, verdict="confirmed", why=entry.why, looks=len(entry.looks))
            return True

        same = next((e for e in mine if e.item.settled == product_id), None)
        if same is not None:
            return confirm(same) if same.verified != "confirmed" else False
        open_picks = [e for e in mine if e.verified is None and self.catalog.family_of(e.item.seen_as) == product.family]
        open_picks.sort(key=lambda e: (e.item.settled is not None, e.id))  # picks with no name at all first
        return confirm(open_picks[0]) if open_picks else False

    def add_seen(self, pid: int, product_id: str, why: str) -> bool:
        """The second model read this title in the shopper's hands, more than once, and no pick of theirs is free to
        take the name: the track-level view missed that pick (a book slid straight into a stack, say). The line is
        added on the reading alone, on a track that is never in view."""
        product, person = self.catalog.by_id.get(product_id), self.people.people.get(pid)
        if product is None or product.head or not product.family or person is None:
            return False
        if any(e.person == pid and e.state == "in_cart" and e.item.settled == product_id for e in self.entries):
            return False
        ghost = Item(self.next_item, person.mask, person.box, person.area, person.c.copy(), [], 1.0, self.t, self.t, person.c.copy(), False)
        self.next_item += 1
        ghost.state, ghost.holder, ghost.visible, ghost.was_away, ghost.hits = HELD_STATE, pid, False, True, 4
        ghost.votes = {product.family: 1.0}
        ghost.settled, ghost.read_how, ghost.read_in_hand, ghost.read_sure = product_id, why[:160], True, True
        self.items[ghost.id] = ghost
        entry = Entry(len(self.entries) + 1, ghost, pid, self.t, "seen", 1.0)
        entry.verified, entry.why = "confirmed", why[:160]
        entry.confidence = self._confidence(entry)
        ghost.entry = entry
        self.entries.append(entry)
        self._emit("pick", person=pid, entry=entry.id, item=ghost.id, origin="seen", confidence=entry.confidence)
        return True

    def has_title(self, pid: int, product_id: str) -> bool:
        """Is this title on the shopper's cart right now?"""
        return any(e.person == pid and e.state == "in_cart" and e.item.settled == product_id for e in self.entries)

    def awaiting(self, pid: int, product_id: str) -> bool:
        """Does this shopper have a line that already carries this title, from the shelf or from an earlier
        reading, and only waits for it to be read in their hands?"""
        return any(e.person == pid and e.state == "in_cart" and not e.paid and e.verified is None and e.item.settled == product_id for e in self.entries)

    def drop_title(self, pid: int, product_id: str, how: str) -> bool:
        """The shopper is no longer carrying this title (it has stopped showing up in their hands): off the cart."""
        for entry in self.entries:
            if entry.person == pid and entry.state == "in_cart" and not entry.paid and entry.item.settled == product_id:
                if entry.item.visible and entry.item.state == HELD_STATE:
                    return False  # its outline is in their hand right now: turned away or covered, not put down
                self._return(entry.item, how, 0.7)
                return True
        return False

    def void_unnamed(self, pid: int, older_than: float, keep: int = 0) -> int:
        """The second model sees nothing in the shopper's hands that it cannot name, so picks still without a
        confirmed title are second outlines of books already on the cart. They are folded away."""
        n = 0
        stray = [e for e in self.entries if e.person == pid and e.state == "in_cart" and not e.paid and e.verified is None and e.picked_t <= older_than and self.catalog.family_of(e.item.seen_as)]
        # `keep` of them stay: the model said that many things in the hands could not be read (the newest picks)
        for entry in stray[: max(0, len(stray) - keep)]:
            if True:
                entry.state = "merged"
                entry.item.state, entry.item.entry = GONE, None
                self._emit("void", person=pid, entry=entry.id, into=None)
                n += 1
        return n

    def holding(self, pid: int) -> bool:
        """Does the frame loop currently see something in this shopper's hand?"""
        return any(i.state == HELD_STATE and i.holder == pid and i.visible for i in self.items.values())

    def unnamed(self, pid: int) -> int:
        """How many of a shopper's picks have no confirmed title yet."""
        return sum(1 for e in self.entries if e.person == pid and e.state == "in_cart" and not e.paid and e.verified is None and self.catalog.family_of(e.item.seen_as))

    def apply_look(self, entry_id: int, holding: bool, product_id: str | None, why: str = "", weight: int = 1, names_only: bool = False) -> None:
        """A second model looked at one frame of a held item. One look can be wrong (an item seen edge-on, a hand
        in the way), so the verdict comes from the looks together:

        disputed    most looks saw no item in the hand
        confirmed   the looks give one product two votes, or one look names the very product SAM favours
        unknown     it is held, but no look could name a catalogue product
        (unsettled) the looks and SAM disagree: SAM's label and the confidence rule stand until more looks come in

        A look counts double (`weight` 2) when the text read off the item by the local OCR points to the same product.
        A look that only says which product the item looks like (`names_only`: the likeness to the catalogue's pictures)
        helps name it, but is not counted for or against it being held: whether the item left the shelf is for the
        looks at the shopper's hands to say.
        """
        entry = next((e for e in self.entries if e.id == entry_id), None)
        if entry is None or entry.state != "in_cart" or entry.paid:
            return
        item = entry.item
        named = self.catalog.by_id.get(product_id)
        entry.looks.append((holding, product_id if named and not named.head else None, weight, names_only))
        held = [(p, w) for h, p, w, _ in entry.looks if h]
        held_looks = sum(1 for h, _, _, only in entry.looks if h and not only)
        not_held = sum(1 for h, _, _, only in entry.looks if not h and not only)
        tally: dict[str, int] = {}
        for p, w in held:
            if p:
                tally[p] = tally.get(p, 0) + w
        best = max(tally.values(), default=0)
        tops = [p for p, n in tally.items() if n == best]
        winner = None
        if len(tops) == 1 and (best >= 2 or tops[0] == item.seen_as):
            winner = tops[0]
        elif tops and item.seen_as in tops:
            winner = item.seen_as
        if winner and not self.catalog.same_kind(item.seen_as, winner):
            # Another product than SAM's phrase says. SAM's phrases can outline the item beside the right one, so
            # the looks may overrule them, but only on the second model's own word plus one more (a second look, or
            # the likeness to the pictures): the pictures are compared with billable products only, so on their own
            # they would turn a shopper's phone into the power bank it resembles. Books keep their own rules.
            looked = any(h and p == winner and not only for h, p, _, only in entry.looks)
            if not looked or not self._renamable(item.seen_as, winner):
                winner = None
        if not_held > held_looks:
            verdict = "disputed"
        elif winner:
            verdict = "confirmed"
        elif held and not tally:
            verdict = "unknown"
        else:
            verdict = None
        if verdict == "confirmed":
            if item.state == HELD_STATE and item.entry is entry:
                self.name_held(item.id, winner, why, overrule=self._renamable(item.seen_as, winner))
            elif (entry.verified, item.settled) != ("confirmed", winner):
                # the item is out of sight by now (its shopper has left, say), so only the cart line can change
                twin = next((e for e in self.entries if e is not entry and e.person == entry.person and e.state == "in_cart" and e.item.settled == winner), None)
                if twin is not None and not item.read_in_hand:
                    entry.state = "merged"  # the shopper already has a line for this very product: same item, second track
                    self._emit("void", person=entry.person, entry=entry.id, into=twin.id)
                    return
                item.settled, item.read_how, item.read_sure, entry.verified, entry.why = winner, why[:160], True, "confirmed", why[:160]
                self._emit("verify", person=entry.person, entry=entry.id, item=item.id, verdict="confirmed", why=entry.why, looks=len(entry.looks))
            return
        if entry.verified == "confirmed" and item.settled:
            return  # already identified (on the shelf or in the hand); looks that could not agree do not undo that
        if verdict != entry.verified:
            entry.verified, entry.why = verdict, why[:160]
            self._emit("verify", person=entry.person, entry=entry.id, item=item.id, verdict=verdict or "unsettled", why=entry.why, looks=len(entry.looks))

    def note_likeness(self, entry_id: int, product_id: str, score: float, why: str) -> None:
        """How alike a picked item looks to its nearest catalogue picture (likeness.py). Kept on the line and the
        item for the page and the ledger; the vote it may carry goes through apply_look like any other look."""
        entry = next((e for e in self.entries if e.id == entry_id), None)
        product = self.catalog.by_id.get(product_id)
        if entry is None or product is None:
            return
        best = entry.likeness
        if best is None or score > best["score"]:
            entry.likeness = {"product": product_id, "name": product.name, "score": round(score, 3), "why": why}
            entry.item.likeness = {"product": product_id, "name": product.name, "score": round(score, 3)}

    def apply_audit(self, entry_id: int, verdict: str, why: str = "") -> None:
        """The check before charging has looked at a line: kept (charge it), put_back or unclear (do not)."""
        entry = next((e for e in self.entries if e.id == entry_id), None)
        if entry is None or entry.paid:
            return
        entry.audit, entry.audit_why = verdict, why[:160]
        self._emit("audit", person=entry.person, entry=entry.id, item=entry.item.id, verdict=verdict, why=entry.audit_why)

    def unpaid(self, pid: int) -> list[Entry]:
        return [e for e in self.entries if e.person == pid and not e.paid and self._chargeable(e)]

    def mark_paid(self, entries: list[Entry], payment_id: str) -> None:
        for entry in entries:
            entry.paid = payment_id

    def close(self) -> list[int]:
        """The footage ended: everyone still inside is treated as walking out now."""
        first = len(self.events)
        for person in self.people.people.values():
            if person.state in (ppl.ACTIVE, ppl.LOST):
                person.state, person.exit_t = ppl.EXITED, self.t
                self._person_event({"type": "exit", "person": person.id, "how": "ended"})
        return [e["person"] for e in self.events[first:] if e["type"] == "exit"]

    def carts(self) -> list[dict]:
        known = [p.id for p in self.people.people.values() if p.state != ppl.TENTATIVE]
        return [self._cart(pid) for pid in sorted(set(known) | {e.person for e in self.entries if e.state != "merged"})]

    def snapshot(self, new_events: list[dict] | None = None) -> dict:
        people = [
            {"id": p.id, "state": p.state, "box": _nbox(p.box, self.size), "polygons": p.polygons, "score": round(p.score, 2)}
            for p in self.people.people.values()
            if p.state == ppl.ACTIVE and p.seen
        ]
        hands = [{"box": d.nbox, "polygons": d.polygons, "person": d.extra.get("person")} for d in self.hands]
        items = []
        for item in self.items.values():
            if not item.visible or item.state == GONE or (item.state == CAND and not item.hold_run):
                continue
            product = self.catalog.by_id.get(item.cls)
            items.append(
                {
                    "id": item.id,
                    "state": item.state,
                    "holder": item.holder if item.state == HELD_STATE else None,
                    "product": product.id if product else None,
                    "name": product.name if product else "Unrecognised item",
                    "billable": bool(product and product.billable),
                    "unread": bool(product and product.head),
                    "read": bool(item.settled),
                    "box": _nbox(item.box, self.size),
                    "polygons": item.polygons,
                    "score": round(item.score, 2),
                    "likeness": item.likeness,
                }
            )
        return {"t": round(self.t, 3), "people": people, "hands": hands, "items": items, "events": new_events or [], "carts": self.carts()}

    def zone_outline(self) -> list:
        """The shelf area as normalised polygons, for the overlay."""
        if self.zone is None:
            return []
        w, h = self.size
        contours, _ = cv2.findContours((self.zone > 0.4).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out = []
        for c in contours:
            if cv2.contourArea(c) >= 0.002 * w * h:
                pts = cv2.approxPolyDP(c, 0.004 * cv2.arcLength(c, True), True).reshape(-1, 2)
                out.append([[round(float(x) / w, 4), round(float(y) / h, 4)] for x, y in pts])
        return out


@dataclass
class _Grown:
    """A person with a dilated mask, shaped like a Det for `cover`."""

    person: ppl.Person
    mask: np.ndarray

    @property
    def box(self):
        h, w = self.mask.shape
        return (0, 0, w, h)


def _nbox(box: tuple, size: tuple) -> list[float]:
    w, h = size
    return [round(box[0] / w, 4), round(box[1] / h, 4), round(box[2] / w, 4), round(box[3] / h, 4)]

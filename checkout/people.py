"""
Shopper tracking: one stable id per person from per-frame SAM person masks.

A detection is matched to a track on three cues: how much the two masks overlap, how alike the clothing colours are
(a hue/saturation histogram of the upper and lower half of the mask), and how far the centre moved. A track that
goes unseen is kept as `lost` and can be picked up again; after `exit_after_s` the shopper has left and their cart
is closed. A shopper who walks back in within `reid_window_s` is recognised by clothing and gets the same id back.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from .geometry import Det, iou, nms

TENTATIVE, ACTIVE, LOST, EXITED = "tentative", "active", "lost", "exited"


@dataclass
class Person:
    id: int
    mask: np.ndarray
    box: tuple
    area: int
    c: np.ndarray
    polygons: list
    score: float
    hist: np.ndarray | None
    first_t: float
    last_t: float
    hits: int = 1
    state: str = TENTATIVE
    exit_t: float | None = None
    seen: bool = True  # matched in the current frame

    @property
    def present(self) -> bool:
        return self.state in (TENTATIVE, ACTIVE) and self.seen

    def take(self, d: Det, t: float, hist) -> None:
        self.mask, self.box, self.area, self.c, self.polygons, self.score = d.mask, d.box, d.area, d.c, d.polygons, d.score
        if hist is not None:
            self.hist = hist if self.hist is None else 0.85 * self.hist + 0.15 * hist
        self.last_t, self.hits, self.seen = t, self.hits + 1, True


def appearance(hsv: np.ndarray | None, d: Det) -> np.ndarray | None:
    """Clothing signature: hue/saturation histograms of the top and bottom half of the mask, each summing to 0.5."""
    if hsv is None:
        return None
    x0, y0, x1, y1 = d.box
    mid = (y0 + y1) // 2
    parts = []
    for ya, yb in ((y0, mid), (mid, y1)):
        if yb <= ya:
            parts.append(np.zeros(96, np.float32))
            continue
        hist = cv2.calcHist([hsv[ya:yb, x0:x1]], [0, 1], d.mask[ya:yb, x0:x1], [16, 6], [0, 180, 0, 256]).flatten()
        total = hist.sum()
        parts.append(hist / total * 0.5 if total else hist)
    return np.concatenate(parts).astype(np.float32)


def alike(a: np.ndarray | None, b: np.ndarray | None) -> float:
    """Bhattacharyya coefficient, 0..1. 0.5 when either side has no picture to compare."""
    if a is None or b is None:
        return 0.5
    return float(np.sqrt(a * b).sum())


class PeopleTracker:
    def __init__(self, cfg):
        self.cfg = cfg
        self.people: dict[int, Person] = {}
        self.next_id = 1

    def visible(self) -> list[Person]:
        return [p for p in self.people.values() if p.present]

    def update(self, t: float, dets: list[Det], hsv: np.ndarray | None, diag: float, frame_area: int) -> list[dict]:
        cfg, events = self.cfg, []
        dets = nms([d for d in dets if d.score >= cfg.person_min_score and d.area >= cfg.person_min_area * frame_area], 0.5, 0.8)
        hists = [appearance(hsv, d) for d in dets]
        tracks = [p for p in self.people.values() if p.state != EXITED]
        for p in tracks:
            p.seen = False

        if tracks and dets:
            sim = np.zeros((len(tracks), len(dets)))
            for i, p in enumerate(tracks):
                gone = t - p.last_t
                for j, d in enumerate(dets):
                    overlap = iou(p, d)
                    look = alike(p.hist, hists[j])
                    dist = float(np.linalg.norm(p.c - d.c)) / diag
                    reach = min(0.6, 0.2 + 0.15 * gone)  # a lost shopper may have walked on
                    if overlap < 0.05 and not (dist < reach and look > 0.55):
                        continue
                    sim[i, j] = 0.55 * overlap + 0.3 * look + 0.15 * max(0.0, 1 - dist / reach)
            rows, cols = linear_sum_assignment(1 - sim)
            taken = set()
            for i, j in zip(rows, cols):
                if sim[i, j] > 0.15:
                    tracks[i].take(dets[j], t, hists[j])
                    taken.add(j)
            rest = [(d, h) for j, (d, h) in enumerate(zip(dets, hists)) if j not in taken]
        else:
            rest = list(zip(dets, hists))

        for d, hist in rest:
            back = self._returning(t, hist) if d.score >= cfg.person_birth_score else None
            if back:
                back.state, back.exit_t = ACTIVE, None
                back.take(d, t, hist)
                events.append({"type": "reenter", "person": back.id})
            elif d.score >= cfg.person_birth_score:
                self.people[self.next_id] = Person(self.next_id, d.mask, d.box, d.area, d.c, d.polygons, d.score, hist, t, t)
                self.next_id += 1

        for p in list(self.people.values()):
            if p.state == TENTATIVE:
                if p.seen and p.hits >= cfg.person_confirm_hits and t - p.first_t >= cfg.person_confirm_s:
                    p.state = ACTIVE
                    events.append({"type": "enter", "person": p.id})
                elif t - p.last_t > cfg.person_drop_s:
                    del self.people[p.id]
            elif p.state == ACTIVE and not p.seen:
                p.state = LOST
            elif p.state == LOST and p.seen:
                p.state = ACTIVE
            if p.state == LOST and t - p.last_t >= cfg.exit_after_s:
                p.state, p.exit_t = EXITED, t
                events.append({"type": "exit", "person": p.id})
        return events

    def _returning(self, t: float, hist) -> Person | None:
        """A shopper who left a moment ago and is dressed the same."""
        if hist is None:
            return None
        best, best_sim = None, self.cfg.reid_min_sim
        for p in self.people.values():
            if p.state == EXITED and p.exit_t is not None and t - p.exit_t <= self.cfg.reid_window_s:
                s = alike(p.hist, hist)
                if s > best_sim:
                    best, best_sim = p, s
        return best

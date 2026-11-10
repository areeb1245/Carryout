"""
Reading what is printed on an item, on this machine, and matching it to the catalogue.

SAM says "this is a book" and gives its outline. To know which book, the outline is cut out of the frame (so the
neighbours' spines are not read too), turned so vertical spine text lies flat, and read with RapidOCR (PP-OCR models
through onnxruntime, about 40 ms a reading on a laptop). The text is then compared with every title and author in the
item's family.

OCR on a spine drops spaces and swaps the odd letter ("THERICHESTMANINBABYION"), so the comparison ignores spaces and
case and scores the best-matching stretch of the text against each title. A title is accepted from the text alone only
when it scores high and clearly beats the runner-up; anything less is passed, with the text, to the second model.
"""
from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from difflib import SequenceMatcher

import cv2
import numpy as np

from .catalog import Product

SURE = 0.8  # score at which the text alone settles the title
LEAD = 0.12  # and by how much it must beat the next title
MIN_SIDE = 64  # crops are enlarged so the short side is at least this many pixels


def squash(text: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def part_score(needle: str, hay: str) -> float:
    """How well `needle` matches its best-fitting stretch of `hay`, 0..1."""
    if not needle or not hay:
        return 0.0
    if needle in hay:
        return 1.0
    if len(hay) <= len(needle):
        return SequenceMatcher(None, needle, hay).ratio()
    matcher = SequenceMatcher(None, needle, hay)
    best = 0.0
    # candidate windows start where a matching block starts, as in the usual partial-ratio trick
    for block in matcher.get_matching_blocks():
        start = max(0, block.b - block.a)
        window = hay[start : start + len(needle)]
        best = max(best, SequenceMatcher(None, needle, window).ratio())
    return best


@dataclass
class Reading:
    text: str  # what was read, pieces joined with " | "
    product: Product | None  # best match in the family, if any text matched at all
    score: float
    sure: bool  # good enough to settle the title without asking the second model

    def quote(self) -> str:
        return self.text[:80]


class Reader:
    def __init__(self):
        self._ocr = None
        self.lock = threading.Lock()
        try:
            import rapidocr_onnxruntime  # noqa: F401

            self.ready = True
        except ImportError:
            self.ready = False

    def _engine(self):
        if self._ocr is None:
            from rapidocr_onnxruntime import RapidOCR

            # "max" keeps small crops small: the default enlarges every crop to 736 px and takes over a second
            self._ocr = RapidOCR(det_limit_side_len=480, det_limit_type="max")
        return self._ocr

    def text_on(self, frame: np.ndarray, polygons: list, box: list[float]) -> str:
        """The text inside an item's outline. `polygons` and `box` are normalised to the frame."""
        if not self.ready:
            return ""
        h, w = frame.shape[:2]
        x0, y0, x1, y1 = max(0, int(box[0] * w) - 2), max(0, int(box[1] * h) - 2), min(w, int(box[2] * w) + 2), min(h, int(box[3] * h) + 2)
        if x1 - x0 < 6 or y1 - y0 < 6:
            return ""
        crop = frame[y0:y1, x0:x1].copy()
        mask = np.zeros(crop.shape[:2], np.uint8)
        for poly in polygons:
            pts = (np.asarray(poly, np.float32) * [w, h] - [x0, y0]).astype(np.int32)
            cv2.fillPoly(mask, [pts], 255)
        mask = cv2.dilate(mask, np.ones((5, 5), np.uint8))
        crop[mask == 0] = (127, 127, 127)
        short = min(crop.shape[:2])
        if short < MIN_SIDE:
            scale = MIN_SIDE / short
            crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        tall = crop.shape[0] > 1.3 * crop.shape[1]
        # a spine on a shelf is tall with the text running up or down it; a cover held up is read as it is
        views = [cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE), cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)] if tall else [crop, cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE), cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)]
        best, best_weight = [], 0.0
        with self.lock:
            ocr = self._engine()
            for view in views:
                found, _ = ocr(view, use_cls=False)
                pieces = [(text, score) for _, text, score in (found or []) if score >= 0.6 and len(squash(text)) >= 2]
                weight = sum(score * len(squash(text)) for text, score in pieces)
                if weight > best_weight:
                    best, best_weight = pieces, weight
        return " | ".join(text for text, _ in best)

    def match(self, text: str, products: list[Product]) -> Reading:
        hay = squash(text)
        if len(hay) < 3 or not products:
            return Reading(text, None, 0.0, False)
        scored = []
        for p in products:
            title = part_score(squash(p.name), hay)
            # the author adds weight but cannot carry a match alone: two books by one author sit on the same shelf
            author = max((part_score(squash(a), hay) for a in re.split(r"\band\b|,|&", p.author) if len(squash(a)) >= 4), default=0.0)
            scored.append((0.8 * title + 0.2 * author if author > 0.75 else 0.9 * title, p))
        scored.sort(key=lambda pair: -pair[0])
        top, product = scored[0]
        lead = top - scored[1][0] if len(scored) > 1 else top
        # short titles ("Ikigai") match by accident too easily to be settled by text alone
        sure = top >= SURE and lead >= LEAD and len(squash(product.name)) >= 6
        return Reading(text, product if top >= 0.45 else None, round(top, 2), sure)

    def support(self, text: str, product: Product) -> float:
        """How much the text speaks for one given product: its title or one of its authors, whichever shows best."""
        hay = squash(text)
        authors = [part_score(squash(a), hay) for a in re.split(r"\band\b|,|&", product.author) if len(squash(a)) >= 4]
        return max([part_score(squash(product.name), hay), *authors]) if hay else 0.0

    def read(self, frame: np.ndarray, polygons: list, box: list[float], products: list[Product]) -> Reading:
        return self.match(self.text_on(frame, polygons, box), products)

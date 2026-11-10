"""
Does what a shopper picked up look like the catalogue's pictures of it?

A product's example pictures (catalog.py, `examples`) are turned into vectors once, by the `/v1/embed` call of the
SAM endpoint (DINOv2), and kept on disk next to nothing else: `data/likeness/<shop>/<file>.json`. A picked item is
cut out of the frame by its box and turned into a vector the same way. The cosine between the two says how alike
they look: about 0.6 and up for the same product from another angle, under 0.35 for something else.

It is one more look at a pick, not the decision. A clear likeness to one product (`SURE`, and `MARGIN` ahead of the
next) counts as one vote for it in the engine's tally of looks (engine.apply_look); a pick that resembles none of
the pictures gets no vote and is left to the other looks. Only products with pictures take part.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import cv2
import numpy as np

from .catalog import Catalog
from .paths import DATA, SHOPS
from .sam_client import SamClient

log = logging.getLogger("checkout")
CACHE = DATA / "likeness"
SURE = 0.55  # this alike, and it is a vote for that product
MARGIN = 0.05  # ahead of the next product by this much
CROP_SIDE = 320


def examples_dir(shop: str) -> Path:
    return SHOPS / shop / "examples"


def crop(frame: np.ndarray, box: list[float], grow: float = 0.08) -> bytes | None:
    """The item cut out of the frame by its box (0..1), a little loose, as a JPEG."""
    h, w = frame.shape[:2]
    bw, bh = box[2] - box[0], box[3] - box[1]
    x0, y0 = int(max(0.0, box[0] - grow * bw) * w), int(max(0.0, box[1] - grow * bh) * h)
    x1, y1 = int(min(1.0, box[2] + grow * bw) * w), int(min(1.0, box[3] + grow * bh) * h)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    piece = frame[y0:y1, x0:x1]
    k = CROP_SIDE / max(piece.shape[:2])
    if k < 1:
        piece = cv2.resize(piece, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    return cv2.imencode(".jpg", piece, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes()


class Likeness:
    def __init__(self, sam: SamClient, catalog: Catalog):
        self.sam = sam
        self.catalog = catalog
        self.vectors: dict[str, np.ndarray] = {}  # product id -> its example vectors, one per row
        self.ready = False
        self.failed = ""

    @property
    def wanted(self) -> bool:
        return bool(self.catalog.shop) and any(p.examples for p in self.catalog.products if p.billable)

    async def prepare(self) -> None:
        """Vectors for every example picture: from the cache, or one embed call for the ones not seen before."""
        if not self.wanted:
            return
        folder, cache = examples_dir(self.catalog.shop), CACHE / self.catalog.shop
        cache.mkdir(parents=True, exist_ok=True)
        todo: list[tuple[str, str, bytes]] = []
        found: dict[str, list] = {}
        for p in self.catalog.products:
            for name in p.examples:
                kept = cache / f"{name}.json"
                if kept.exists():
                    found.setdefault(p.id, []).append(json.loads(kept.read_text()))
                elif (folder / name).exists():
                    todo.append((p.id, name, (folder / name).read_bytes()))
        try:
            if todo:
                for (pid, name, _), vec in zip(todo, await self.sam.embed([b for *_, b in todo])):
                    (cache / f"{name}.json").write_text(json.dumps(vec))
                    found.setdefault(pid, []).append(vec)
        except Exception as e:  # the endpoint has no /v1/embed, or is down: the other looks carry on without this one
            self.failed = str(e) or type(e).__name__
            log.warning("example pictures could not be embedded: %s", self.failed)
        self.vectors = {pid: np.asarray(rows, np.float32) for pid, rows in found.items() if rows}
        self.ready = bool(self.vectors)

    async def compare(self, picture: bytes, among: list[str] | None = None) -> list[tuple[str, float]]:
        """(product id, likeness) for the products with pictures, most alike first. Likeness is the best cosine
        over that product's pictures."""
        if not self.ready:
            return []
        vec = np.asarray((await self.sam.embed([picture]))[0], np.float32)
        scores = [(pid, float((rows @ vec).max())) for pid, rows in self.vectors.items() if among is None or pid in among]
        return sorted(scores, key=lambda s: -s[1])

    @staticmethod
    def verdict(scores: list[tuple[str, float]]) -> tuple[str | None, str]:
        """The product the likeness votes for, if it is clear, and a few words on it."""
        if not scores:
            return None, ""
        best, score = scores[0]
        second = scores[1][1] if len(scores) > 1 else 0.0
        if score >= SURE and score - second >= MARGIN:
            return best, f"Looks like the catalogue picture ({round(score * 100)} %)"
        return None, f"Unlike the catalogue pictures (best {round(score * 100)} %)"

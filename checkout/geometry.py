"""Mask helpers. Every mask is a uint8 0/1 array at the engine's working size (480 px wide by default)."""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


def raster(polygons: list, w: int, h: int) -> np.ndarray:
    """Normalised polygons ([[x, y], ...] in 0..1) to a filled 0/1 mask."""
    mask = np.zeros((h, w), np.uint8)
    for poly in polygons or []:
        if len(poly) >= 3:
            pts = np.round(np.asarray(poly, np.float32) * [w, h]).astype(np.int32)
            cv2.fillPoly(mask, [pts], 1)
    return mask


def _kernel(radius: float) -> tuple[np.ndarray, int]:
    r = max(1, int(round(radius)))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)), r


def dilate(mask: np.ndarray, radius: float, box: tuple | None = None) -> np.ndarray:
    """Grow a mask by `radius` pixels. With the mask's `box` given, only that part of the frame is worked on."""
    kernel, r = _kernel(radius)
    if box is None:
        return cv2.dilate(mask, kernel)
    grown, (ys, xs) = grow(mask, box, radius)
    out = np.zeros_like(mask)
    out[ys, xs] = grown
    return out


def grow(mask: np.ndarray, box: tuple, radius: float) -> tuple[np.ndarray, tuple[slice, slice]]:
    """A mask grown by `radius` pixels, as the small patch of the frame it lives in plus where that patch sits.
    Most masks are a hand or a product: growing the patch instead of the whole frame is some fifty times cheaper."""
    kernel, r = _kernel(radius)
    h, w = mask.shape
    ys, xs = slice(max(0, box[1] - r - 1), min(h, box[3] + r + 1)), slice(max(0, box[0] - r - 1), min(w, box[2] + r + 1))
    return cv2.dilate(mask[ys, xs], kernel), (ys, xs)


@dataclass
class Det:
    """One SAM detection, rasterised."""

    label: str
    score: float
    polygons: list
    mask: np.ndarray
    box: tuple[int, int, int, int]  # x0, y0, x1, y1 in mask pixels, x1/y1 exclusive
    area: int
    c: np.ndarray  # centroid, mask pixels
    extra: dict = field(default_factory=dict)

    @property
    def nbox(self) -> list[float]:
        h, w = self.mask.shape
        x0, y0, x1, y1 = self.box
        return [round(x0 / w, 4), round(y0 / h, 4), round(x1 / w, 4), round(y1 / h, 4)]


def make_det(raw: dict, w: int, h: int) -> Det | None:
    polygons = raw.get("polygons") or []
    if not polygons:
        # a detection without a mask (masks switched off): fall back to its box
        x0, y0, x1, y1 = raw["box"]
        polygons = [[[x0, y0], [x1, y0], [x1, y1], [x0, y1]]]
    mask = raster(polygons, w, h)
    # box, area and centre from the outline's own extent: scanning the whole frame for each mask is slow
    pts = np.concatenate([np.asarray(p, np.float32) for p in polygons if len(p) >= 3] or [np.zeros((0, 2), np.float32)]) * [w, h]
    if not len(pts):
        return None
    x0, y0 = max(0, int(np.floor(pts[:, 0].min()))), max(0, int(np.floor(pts[:, 1].min())))
    x1, y1 = min(w, int(np.ceil(pts[:, 0].max())) + 1), min(h, int(np.ceil(pts[:, 1].max())) + 1)
    if x1 <= x0 or y1 <= y0:
        return None
    m = cv2.moments(mask[y0:y1, x0:x1], binaryImage=True)
    if m["m00"] < 1:
        return None
    centre = np.array([m["m10"] / m["m00"] + x0, m["m01"] / m["m00"] + y0])
    return Det(raw["label"], float(raw["score"]), polygons, mask, (x0, y0, x1, y1), int(m["m00"]), centre)


def boxes_touch(a: tuple, b: tuple, pad: int = 0) -> bool:
    return a[0] - pad < b[2] and b[0] - pad < a[2] and a[1] - pad < b[3] and b[1] - pad < a[3]


def inter(a, b) -> int:
    """Overlap in pixels between two objects that carry `.mask` and `.box`."""
    if not boxes_touch(a.box, b.box):
        return 0
    x0, y0 = max(a.box[0], b.box[0]), max(a.box[1], b.box[1])
    x1, y1 = min(a.box[2], b.box[2]), min(a.box[3], b.box[3])
    return int(np.count_nonzero(a.mask[y0:y1, x0:x1] & b.mask[y0:y1, x0:x1]))


def iou(a, b) -> float:
    i = inter(a, b)
    return i / (a.area + b.area - i) if i else 0.0


def cover(a, b) -> float:
    """Share of `a` that lies inside `b`."""
    return inter(a, b) / a.area if a.area else 0.0


def nms(dets: list[Det], iou_max: float = 0.5, cover_max: float = 0.8) -> list[Det]:
    """Keep the best-scoring mask of each overlapping group. SAM often returns an object twice, or an object and a
    merged pair; the lower score of any two that share most of the smaller one is dropped."""
    kept: list[Det] = []
    for d in sorted(dets, key=lambda d: -d.score):
        for k in kept:
            i = inter(d, k)
            if i and (i / (d.area + k.area - i) > iou_max or i / min(d.area, k.area) > cover_max):
                break
        else:
            kept.append(d)
    return kept


def drop_groups(dets: list[Det]) -> list[Det]:
    """Remove masks that are really several objects lumped together.

    Asked for "book" on a full shelf, SAM returns each spine and also a few blobs covering a whole row. A mask with two
    or more clearly smaller masks of the same kind inside it is such a blob; the things inside it are the objects.
    """
    out = []
    for d in dets:
        inside = sum(1 for o in dets if o is not d and o.area < 0.6 * d.area and cover(o, d) > 0.8)
        if inside < 2:
            out.append(d)
    return out

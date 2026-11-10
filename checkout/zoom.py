"""
A closer look at the shopper's hands, for items held further from the camera.

SAM finds an object less well the smaller it is in the frame. A tissue pack held at arm's length, with the shopper a
few steps back, was not outlined at all in the whole frame; the same frame cut down to the hands and blown up to
2-3 times the size gave it 0.45-0.59, and the power bank in the other hand scored higher too. So while the hands are
small in the frame, every frame is also sent to SAM cut down to them (around where they were a moment ago, so both
calls go out together), for the catalogue's phrases only. What it finds is put back into frame coordinates and
joins the frame's own detections; the engine already merges two outlines of one object.
"""
from __future__ import annotations

import cv2
import numpy as np

MIN_HAND_SCORE = 0.4
WIDEST = 0.62  # hands spread over more of the frame than this: they are close enough, no closer look
NARROWEST = 0.3  # at most about 3x: past that the frame's own pixels run out
SIDE = 960  # the cut-out is sent at this width, as a frame is


def region(hands: list[list[float]]) -> list[float] | None:
    """The part of the frame round these hands to look at closer, as x0 y0 x1 y1 in 0..1, or None when they are big
    enough in the frame already. It takes in what they hold (items stick out above and beside a hand) and has the
    frame's shape (as wide as high, in these units), so the item is blown up the same both ways."""
    if not hands:
        return None
    x0, y0 = min(b[0] for b in hands), min(b[1] for b in hands)
    x1, y1 = max(b[2] for b in hands), max(b[3] for b in hands)
    # what a hand holds reaches about half a hand past it to the side and more than half a hand above it
    w = max(b[2] - b[0] for b in hands)
    h = max(b[3] - b[1] for b in hands)
    x0, x1 = x0 - max(0.08, 0.5 * w), x1 + max(0.08, 0.5 * w)
    y0, y1 = y0 - max(0.12, 0.6 * h), y1 + 0.05
    side = max(x1 - x0, y1 - y0, NARROWEST)
    if side > WIDEST:
        return None
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    x0, y0 = min(max(0.0, cx - side / 2), 1.0 - side), min(max(0.0, cy - side / 2), 1.0 - side)
    return [x0, y0, x0 + side, y0 + side]


def regions(hands: list[list[float]]) -> list[list[float]]:
    """Where to look closer: round all the hands at once if they are near each other, else round each of the two
    biggest on its own (arms held out to the sides). Empty when the hands are close to the camera already."""
    together = region(hands)
    if together is not None or len(hands) < 2:
        return [together] if together else []
    biggest = sorted(hands, key=lambda b: -(b[2] - b[0]) * (b[3] - b[1]))[:2]
    found = [r for r in (region([b]) for b in biggest) if r]
    if len(found) == 2 and _shared(*found) > 0.3:
        return found[:1]  # two hands close together: one cut-out shows both well enough, at half the cost
    return found


def _shared(a: list[float], b: list[float]) -> float:
    """How much of the smaller box the two share."""
    ix, iy = max(0.0, min(a[2], b[2]) - max(a[0], b[0])), max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return ix * iy / min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))


def hands_in(result: dict, label: str = "hand") -> list[list[float]]:
    return [d["box"] for d in result.get("detections", []) if d["label"] == label and d["score"] >= MIN_HAND_SCORE]


def cut(frame: np.ndarray, box: list[float]) -> bytes:
    """The region as a JPEG, blown up to `SIDE` wide."""
    h, w = frame.shape[:2]
    part = frame[int(box[1] * h): int(box[3] * h), int(box[0] * w): int(box[2] * w)]
    scale = SIDE / max(1, part.shape[1])
    part = cv2.resize(part, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return cv2.imencode(".jpg", part, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes()


def back(detections: list[dict], box: list[float]) -> list[dict]:
    """Detections in the cut-out, moved back into the whole frame's coordinates."""
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    out = []
    for d in detections:
        b = d["box"]
        out.append({
            **d,
            "box": [x0 + b[0] * w, y0 + b[1] * h, x0 + b[2] * w, y0 + b[3] * h],
            "polygons": [[[x0 + x * w, y0 + y * h] for x, y in poly] for poly in d.get("polygons", [])],
            "zoom": True,
        })
    return out

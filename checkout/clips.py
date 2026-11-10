"""
Clips for the catalogue screen: footage of the products, cut into frames, looked at with SAM, so a product can be
picked out by hand and added with a phrase SAM is known to find it by.

    import       a video (uploaded, or one of samples/) becomes up to MAX_FRAMES frames, evenly spread, longer side
                 FRAME_SIDE, in data/clips/<id>/. Nothing leaves this machine but the frames SAM is asked about.
    look         one frame through SAM with the generic phrases, the catalogue's own and any typed in. Answers are
                 cached on disk by request, so scrolling back and forth costs nothing.
    more like    SAM 3's visual prompt: the box round the chosen object is the example, on that frame and the
                 frames near it.
    camera       held up to a camera instead: the page sends a frame about every 0.7 s to a clip started for it,
                 SAM finds the item in hand on each, the page keeps the frames it was clearly seen on, and the
                 server picks the views that differ most (by picture vectors) as the product's pictures.
    phrases      each candidate phrase is asked on the chosen frame and a few around it. A phrase wins when its
                 outline covers the chosen object (overlap with its box) on most of those frames, at a good score.
                 This is how "tissue roll" against "tissue roll pack" is settled: by what SAM actually outlines.
"""
from __future__ import annotations

import json
import re
import uuid
from pathlib import Path

import cv2
import numpy as np

from .paths import DATA
from .sam_client import ROOT

CLIPS = DATA / "clips"
MAX_FRAMES = 120
FRAME_SIDE = 960
GENERIC = ["item held in hand", "product", "package", "bottle", "box"]
FIT = 0.6  # overlap with the chosen box for an outline to be that object
FIT_NEAR = 0.35  # on the frames either side, where it may have moved a little
CLIP_ID = re.compile(r"^[a-f0-9]{12}$")


def clip_dir(clip: str) -> Path | None:
    return CLIPS / clip if CLIP_ID.match(clip or "") and (CLIPS / clip).is_dir() else None


def frame_path(clip: str, n: int) -> Path | None:
    folder = clip_dir(clip)
    path = folder / f"f{n:04d}.jpg" if folder else None
    return path if path and path.exists() else None


def info(clip: str) -> dict | None:
    folder = clip_dir(clip)
    return json.loads((folder / "clip.json").read_text()) if folder and (folder / "clip.json").exists() else None


def cut(video: Path, title: str, every_s: float = 0.5) -> dict:
    """Frames from `video`, one every `every_s` seconds (fewer for a long clip), into a new clip folder."""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError("This file could not be opened as a video.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    length = total / fps if total else 0.0
    step_s = max(every_s, length / MAX_FRAMES) if length else every_s
    clip = uuid.uuid4().hex[:12]
    folder = CLIPS / clip
    folder.mkdir(parents=True, exist_ok=True)
    times, n, next_t, index = [], 0, 0.0, 0
    size = None
    while n < MAX_FRAMES:
        ok, frame = cap.read()
        if not ok:
            break
        t = index / fps
        index += 1
        if t + 1e-6 < next_t:
            continue
        next_t = t + step_s
        h, w = frame.shape[:2]
        k = min(1.0, FRAME_SIDE / max(h, w))
        if k < 1:
            frame = cv2.resize(frame, (round(w * k), round(h * k)), interpolation=cv2.INTER_AREA)
        size = frame.shape[1], frame.shape[0]
        cv2.imwrite(str(folder / f"f{n:04d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
        times.append(round(t, 2))
        n += 1
    cap.release()
    if not n:
        folder.rmdir()
        raise ValueError("No frames could be read from this video.")
    meta = {"clip": clip, "title": title[:80], "frames": n, "times": times, "width": size[0], "height": size[1], "length": round(length, 1)}
    (folder / "clip.json").write_text(json.dumps(meta))
    return meta


def start_live(title: str) -> dict:
    """An empty clip that frames are added to one at a time, from a camera."""
    clip = uuid.uuid4().hex[:12]
    folder = CLIPS / clip
    folder.mkdir(parents=True, exist_ok=True)
    meta = {"clip": clip, "title": title[:80], "frames": 0, "times": [], "width": 0, "height": 0, "length": 0.0, "live": True}
    (folder / "clip.json").write_text(json.dumps(meta))
    return meta


def add_frame(clip: str, jpeg: bytes, t: float) -> int | None:
    """Add one camera frame (a JPEG) to a live clip. Returns its number, or None when the clip is full or not live."""
    meta = info(clip)
    if meta is None or not meta.get("live") or meta["frames"] >= MAX_FRAMES:
        return None
    image = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return None
    h, w = image.shape[:2]
    k = min(1.0, FRAME_SIDE / max(h, w))
    if k < 1:
        image = cv2.resize(image, (round(w * k), round(h * k)), interpolation=cv2.INTER_AREA)
    n = meta["frames"]
    cv2.imwrite(str(CLIPS / clip / f"f{n:04d}.jpg"), image, [cv2.IMWRITE_JPEG_QUALITY, 88])
    meta.update(frames=n + 1, times=meta["times"] + [round(t, 2)], width=image.shape[1], height=image.shape[0], length=round(t, 1))
    (CLIPS / clip / "clip.json").write_text(json.dumps(meta))
    return n


def diverse(vectors: list[list[float]], k: int, first: int = 0) -> list[int]:
    """`k` of the pictures that differ most from each other (farthest first), starting from `first`."""
    if not vectors:
        return []
    v = np.asarray(vectors, np.float32)
    chosen = [first]
    closest = v @ v[first]
    while len(chosen) < min(k, len(v)):
        nxt = int(np.argmin(closest))
        if nxt in chosen:
            break
        chosen.append(nxt)
        closest = np.maximum(closest, v @ v[nxt])
    return chosen


def spread(count: int, k: int) -> list[int]:
    """`k` evenly spaced of `count`, for when there are no picture vectors."""
    if count <= k:
        return list(range(count))
    return sorted({round(i * (count - 1) / (k - 1)) for i in range(k)})


def listed() -> list[dict]:
    if not CLIPS.exists():
        return []
    out = [info(p.name) for p in CLIPS.iterdir() if p.is_dir()]
    return sorted((m for m in out if m), key=lambda m: m["clip"])


def overlap(a: list[float], b: list[float]) -> float:
    """Intersection over union of two boxes x0 y0 x1 y1."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


# What a camera capture asks SAM for. "item held in hand" alone often finds nothing (a power bank held up scores 0
# while "phone" and "device" outline it at 0.9), so concrete nouns are asked too, and the hand decides which is held.
HELD = ["hand", "item held in hand", "phone", "device", "box", "bottle", "package", "can", "jar", "tube"]
HAND_REACH = 0.04  # how far past a hand's box an item may start and still count as in that hand
TOO_BIG = 0.6  # an outline covering more of the frame than this is the background, a shelf or the person


def held(detections: list[dict]) -> dict | None:
    """The item in a hand: the surest outline that is not a hand and touches one. None when no hand holds anything."""
    hands = [d["box"] for d in detections if d["label"] == "hand"]
    best = None
    for d in detections:
        x0, y0, x1, y1 = d["box"]
        if d["label"] == "hand" or (x1 - x0) * (y1 - y0) > TOO_BIG:
            continue
        touching = any(x0 <= h[2] + HAND_REACH and h[0] - HAND_REACH <= x1 and y0 <= h[3] + HAND_REACH and h[1] - HAND_REACH <= y1 for h in hands)
        if (touching or d["label"] == "item held in hand") and (best is None or d["score"] > best["score"]):
            best = d
    return best


def nearby(n: int, frames: int, spread: int = 2) -> list[int]:
    """The chosen frame and up to `spread` on either side."""
    return [i for i in range(n - spread, n + spread + 1) if 0 <= i < frames]


def rank_phrases(n: int, box: list[float], answers: dict[int, dict], phrases: list[str], boxes: dict[int, list[float]] | None = None) -> list[dict]:
    """How well each phrase finds the chosen object. `answers` are SAM's answers per frame for all the phrases.
    On the chosen frame the object is where the box is; on the others it may have moved a little, so a looser
    overlap with the box counts there. `boxes` gives the object's own box on other frames, where it is known
    (pictures taken from a camera): those count as strictly as the chosen frame."""
    boxes = boxes or {}
    rows = []
    for phrase in phrases:
        on_frame, frames_hit, scores = None, 0, []
        for i, answer in answers.items():
            target = box if i == n else boxes.get(i, box)
            found = [d for d in answer.get("detections", []) if d["label"] == phrase]
            best = max(found, key=lambda d: overlap(d["box"], target), default=None)
            fit = overlap(best["box"], target) if best else 0.0
            if i == n:
                on_frame = {"fit": round(fit, 2), "score": round(best["score"], 2) if best else 0.0, "count": len(found)}
            if fit >= (FIT if i == n or i in boxes else FIT_NEAR):
                frames_hit += 1
                scores.append(best["score"])
        hit = bool(on_frame and on_frame["fit"] >= FIT)
        rows.append({
            "phrase": phrase,
            "fits": hit,
            "fit": on_frame["fit"] if on_frame else 0.0,
            "score": on_frame["score"] if on_frame else 0.0,
            "others": max(0, (on_frame["count"] if on_frame else 0) - (1 if hit else 0)),  # other outlines on that frame
            "frames": frames_hit,
            "of": len(answers),
            "mean": round(sum(scores) / len(scores), 2) if scores else 0.0,
        })
    # a phrase that covers the object, on the most frames, at the best score, outlining the fewest other things
    return sorted(rows, key=lambda r: (-int(r["fits"]), -r["frames"], -r["mean"], -r["fit"], r["others"]))

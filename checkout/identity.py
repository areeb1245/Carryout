"""
Customer profiles: the face a shopper checked in with, the card that goes with it, and what they bought before.

Faces are found with YuNet and turned into a 128-number signature with SFace, both run on this machine by OpenCV
(about 15 ms a face on a laptop CPU, so there is nothing to host). Two signatures of the same person score a cosine of
roughly 0.4 to 0.95; two different people stay under 0.2. A profile keeps a few signatures taken at different angles,
and a face belongs to the profile whose best score clears `MATCH`.

Everything is in one SQLite file, `data/checkout.db`:

    profiles    id, name, email, Stripe customer and saved card, a small face picture
    faces       the signatures of each profile
    visits      one row per walk-out: when, how much, how the payment went
    purchases   the lines of each visit

No picture or signature leaves this machine. `forget_all` empties the file.
"""
from __future__ import annotations

import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .sam_client import ROOT

MODELS = ROOT / "models"
DETECTOR = MODELS / "face_detection_yunet_2023mar.onnx"
RECOGNISER = MODELS / "face_recognition_sface_2021dec.onnx"
MATCH = 0.36  # OpenCV's recommended cosine threshold for SFace is 0.363
MARGIN = 0.08  # the best profile must beat the second best by this much
MIN_FACE_PX = 36
MAX_TEMPLATES = 6

SCHEMA = """
CREATE TABLE IF NOT EXISTS profiles (
    id TEXT PRIMARY KEY, name TEXT NOT NULL, email TEXT, customer TEXT, payment_method TEXT, brand TEXT, last4 TEXT,
    created REAL NOT NULL, thumb BLOB
);
CREATE TABLE IF NOT EXISTS faces (
    id INTEGER PRIMARY KEY, profile TEXT NOT NULL REFERENCES profiles(id) ON DELETE CASCADE, signature BLOB NOT NULL
);
CREATE TABLE IF NOT EXISTS visits (
    id INTEGER PRIMARY KEY, profile TEXT NOT NULL REFERENCES profiles(id) ON DELETE CASCADE, session TEXT,
    ended REAL NOT NULL, amount REAL NOT NULL, currency TEXT, status TEXT NOT NULL, payment TEXT, receipt_url TEXT
);
CREATE TABLE IF NOT EXISTS purchases (
    id INTEGER PRIMARY KEY, visit INTEGER NOT NULL REFERENCES visits(id) ON DELETE CASCADE, product TEXT, name TEXT, price REAL
);
"""


@dataclass
class Face:
    signature: np.ndarray  # 1 x 128 float32
    quality: float  # detector score x width in pixels
    thumb: bytes  # small JPEG


@dataclass
class Profile:
    id: str
    name: str
    email: str | None = None
    customer: str | None = None
    payment_method: str | None = None
    brand: str | None = None
    last4: str | None = None
    created: float = 0.0
    has_thumb: bool = False
    templates: list = field(default_factory=list)  # 1 x 128 float32 arrays


class Profiles:
    def __init__(self, path: Path = ROOT / "data" / "checkout.db"):
        path.parent.mkdir(parents=True, exist_ok=True)
        # used from the event loop only; the flag just lets the loop's thread differ from the one that built this
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.execute("PRAGMA foreign_keys = ON")
        self.db.executescript(SCHEMA)
        self.profiles: dict[str, Profile] = {}
        self.ready = DETECTOR.exists() and RECOGNISER.exists()
        if self.ready:
            self.detector = cv2.FaceDetectorYN.create(str(DETECTOR), "", (320, 320), 0.7, 0.3, 50)
            self.recogniser = cv2.FaceRecognizerSF.create(str(RECOGNISER), "")
        for row in self.db.execute("SELECT id, name, email, customer, payment_method, brand, last4, created, thumb IS NOT NULL FROM profiles"):
            self.profiles[row[0]] = Profile(*row[:8], has_thumb=bool(row[8]))
        for pid, blob in self.db.execute("SELECT profile, signature FROM faces ORDER BY id"):
            if pid in self.profiles:
                self.profiles[pid].templates.append(np.frombuffer(blob, np.float32).reshape(1, -1))

    # ---------------------------------------------------------------- profiles

    def add(self, name: str, face: Face | None, email: str = "") -> Profile:
        profile = Profile(uuid.uuid4().hex[:10], name.strip()[:40] or f"Guest {len(self.profiles) + 1}", email.strip()[:120] or None, created=time.time())
        self.profiles[profile.id] = profile
        self.db.execute("INSERT INTO profiles (id, name, email, created) VALUES (?, ?, ?, ?)", (profile.id, profile.name, profile.email, profile.created))
        self.db.commit()
        if face:
            self.learn(profile, face)
        return profile

    def set_card(self, profile: Profile, customer: str, payment_method: str, brand: str, last4: str) -> None:
        profile.customer, profile.payment_method, profile.brand, profile.last4 = customer, payment_method, brand, last4
        self.db.execute("UPDATE profiles SET customer = ?, payment_method = ?, brand = ?, last4 = ? WHERE id = ?", (customer, payment_method, brand, last4, profile.id))
        self.db.commit()

    def learn(self, profile: Profile, face: Face) -> None:
        """Keep another view of this person's face, so they are recognised from more angles."""
        if len(profile.templates) >= MAX_TEMPLATES:
            return
        signature = np.ascontiguousarray(face.signature, np.float32).reshape(1, -1)
        profile.templates.append(signature)
        self.db.execute("INSERT INTO faces (profile, signature) VALUES (?, ?)", (profile.id, signature.tobytes()))
        if not profile.has_thumb:
            profile.has_thumb = True
            self.db.execute("UPDATE profiles SET thumb = ? WHERE id = ?", (face.thumb, profile.id))
        self.db.commit()

    def thumb(self, profile_id: str) -> bytes | None:
        row = self.db.execute("SELECT thumb FROM profiles WHERE id = ?", (profile_id,)).fetchone()
        return row[0] if row else None

    def record_visit(self, profile: Profile, session: str, payment: dict, lines: list[tuple[str, str, float]]) -> None:
        """One walk-out. `lines` are (product id, name, price) of what was billed or left unpaid."""
        cur = self.db.execute(
            "INSERT INTO visits (profile, session, ended, amount, currency, status, payment, receipt_url) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (profile.id, session, time.time(), payment["amount"], payment["currency"], payment["status"], payment.get("id"), payment.get("receipt_url")),
        )
        self.db.executemany("INSERT INTO purchases (visit, product, name, price) VALUES (?, ?, ?, ?)", [(cur.lastrowid, *line) for line in lines])
        self.db.commit()

    def summary(self, profile: Profile) -> dict:
        """What the page shows when this person is recognised."""
        visits, spent, last = self.db.execute("SELECT COUNT(*), COALESCE(SUM(CASE WHEN status = 'succeeded' THEN amount END), 0), MAX(ended) FROM visits WHERE profile = ?", (profile.id,)).fetchone()
        recent = [r[0] for r in self.db.execute("SELECT p.name FROM purchases p JOIN visits v ON v.id = p.visit WHERE v.profile = ? AND v.status = 'succeeded' ORDER BY p.id DESC LIMIT 3", (profile.id,))]
        owed = self.db.execute("SELECT COALESCE(SUM(amount), 0) FROM visits WHERE profile = ? AND status IN ('declined', 'failed', 'no_card', 'off')", (profile.id,)).fetchone()[0]
        return {
            "id": profile.id,
            "name": profile.name,
            "email": profile.email,
            "brand": profile.brand,
            "last4": profile.last4,
            "has_face": bool(profile.templates),
            "has_card": bool(profile.payment_method),
            "visits": visits,
            "spent": round(spent, 2),
            "unpaid": round(owed, 2),
            "last_visit": last,
            "recent": recent,
        }

    def all(self) -> list[dict]:
        return [self.summary(p) for p in sorted(self.profiles.values(), key=lambda p: p.created)]

    def forget_all(self) -> int:
        n = len(self.profiles)
        self.profiles.clear()
        self.db.executescript("DELETE FROM purchases; DELETE FROM visits; DELETE FROM faces; DELETE FROM profiles;")
        self.db.commit()
        self.db.execute("VACUUM")  # so the signatures are gone from the file, not just unlinked
        return n

    # ---------------------------------------------------------------- faces

    def faces_in(self, frame: np.ndarray) -> list[tuple[np.ndarray, float, float]]:
        """Every face in the frame as (detector row, centre x, centre y), centres normalised to 0..1."""
        if not self.ready:
            return []
        h, w = frame.shape[:2]
        self.detector.setInputSize((w, h))
        _, rows = self.detector.detect(frame)
        if rows is None:
            return []
        return [(row, float(row[0] + row[2] / 2) / w, float(row[1] + row[3] / 2) / h) for row in rows if row[2] >= MIN_FACE_PX]

    def read(self, frame: np.ndarray, row: np.ndarray) -> Face:
        signature = self.recogniser.feature(self.recogniser.alignCrop(frame, row)).copy()
        h, w = frame.shape[:2]
        x, y, fw, fh = (int(v) for v in row[:4])
        pad = int(0.25 * max(fw, fh))
        crop = frame[max(0, y - pad) : min(h, y + fh + pad), max(0, x - pad) : min(w, x + fw + pad)]
        thumb = cv2.imencode(".jpg", cv2.resize(crop, (96, 96), interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()
        return Face(signature, float(row[14]) * float(row[2]), thumb)

    def score(self, profile: Profile, face: Face) -> float:
        if not profile.templates:
            return 0.0
        return max(float(self.recogniser.match(t, face.signature, cv2.FaceRecognizerSF_FR_COSINE)) for t in profile.templates)

    def who(self, face: Face, skip: set[str] = frozenset()) -> tuple[Profile, float] | None:
        """The profile this face belongs to, if one clearly does."""
        ranked = sorted(((self.score(p, face), p) for p in self.profiles.values() if p.id not in skip), key=lambda pair: -pair[0])
        if not ranked or ranked[0][0] < MATCH:
            return None
        if len(ranked) > 1 and ranked[0][0] - ranked[1][0] < MARGIN:
            return None
        return ranked[0][1], ranked[0][0]

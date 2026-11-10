"""
The gate: nobody shops before they are registered.

A person stands in front of the gate camera. If their face matches a profile that has a payment method on file they
are let in, which opens a shopping visit. If no profile matches they register on the spot: name, email, a payment
method saved with the payment provider, and the face the camera has of them. Then they are let in.

The shop floor (pipeline.py) only bills people with an open visit, so this is the one way onto a bill.

    look      a camera frame -> who this is: nobody, someone known, or a new face (with a ticket to register on)
    register  a ticket (new face) or a pass (known face without a usable method) plus a payment method
    approved  the shopper came back from approving a PayPal account
    admit     a pass -> an open visit

Tickets, passes and approvals in flight live in memory for a few minutes; nothing here outlives the process.
"""
from __future__ import annotations

import asyncio
import base64
import time
import uuid

import cv2
import numpy as np

from .identity import SURE, Face, Profile, Profiles
from .mailer import valid_email

TTL_S = 600
MIN_QUALITY = 60.0  # detector score x face width in pixels: nearer than this is too small to register on
SAME_FACE = 0.5  # two unknown faces scoring this alike are one person still standing there

FRIENDLY = {
    "NOT_ENABLED_TO_VAULT_PAYMENT_SOURCE": "This PayPal sandbox app is not set up to save cards. Use a PayPal account instead, or switch on Vault for the app in the PayPal developer dashboard.",
    "AUTH": "PayPal did not accept this app's sandbox keys.",
}


class Gate:
    def __init__(self, faces: Profiles, payments=None, shop: str = "Walk-out checkout"):
        self.faces, self.payments, self.shop = faces, payments, shop
        self.tickets: dict[str, dict] = {}  # unknown faces waiting to register: {face, until}
        self.passes: dict[str, dict] = {}  # recognised people: {profile, until}
        self.approvals: dict[str, dict] = {}  # PayPal account approvals in flight

    # ------------------------------------------------------------------ looking

    async def look(self, frame: np.ndarray | None) -> dict:
        if frame is None or not self.faces.ready:
            return {"status": "no_face"}
        faces = self.faces.faces_in(frame)
        if not faces:
            return {"status": "no_face"}
        row = max(faces, key=lambda f: f[0][2])[0]  # the nearest person is the one at the gate
        h, w = frame.shape[:2]
        k = min(1.0, 720 / max(h, w))  # what the camera saw, kept small: it goes on the visit if they are let in
        shot = cv2.imencode(".jpg", cv2.resize(frame, (round(w * k), round(h * k)), interpolation=cv2.INTER_AREA) if k < 1 else frame, [cv2.IMWRITE_JPEG_QUALITY, 82])[1].tobytes()
        answer = await self.see(self.faces.read(frame, row), shot)
        # where the face is, so the screen can show the shopper that it is them the gate is looking at
        answer["box"] = [round(float(v), 4) for v in (row[0] / w, row[1] / h, (row[0] + row[2]) / w, (row[1] + row[3]) / h)]
        return answer

    async def see(self, face: Face | None, shot: bytes | None = None) -> dict:
        """What the gate says to the face in front of it. `shot` is the camera picture the face was found in."""
        self._sweep()
        if face is None:
            return {"status": "no_face"}
        match = await asyncio.to_thread(self.faces.find, face)  # the vector search waits on the database
        if match:
            profile, score = match
            token = self._pass(profile)
            self.passes[token].update(shot=shot, score=round(float(score), 2))
            return {"status": "known", "pass": token, "ready": self.faces.can_pay(profile), "profile": self.faces.summary(profile), "score": round(float(score), 2)}
        if face.quality < MIN_QUALITY:
            return {"status": "closer"}
        ticket = self._ticket(face)
        if shot and self.tickets[ticket]["face"] is face:
            self.tickets[ticket]["shot"] = shot  # the picture their best view so far came from
        thumb = "data:image/jpeg;base64," + base64.b64encode(self.tickets[ticket]["face"].thumb).decode()
        return {"status": "new", "ticket": ticket, "face": thumb}

    def _sweep(self) -> None:
        now = time.time()
        for store in (self.tickets, self.passes, self.approvals):
            for key in [k for k, v in store.items() if v["until"] < now]:
                del store[key]

    def _pass(self, profile: Profile) -> str:
        for token, held in self.passes.items():
            if held["profile"] == profile.id:
                held["until"] = time.time() + TTL_S
                return token
        token = uuid.uuid4().hex
        self.passes[token] = {"profile": profile.id, "until": time.time() + TTL_S}
        return token

    def _ticket(self, face: Face) -> str:
        """One ticket per person standing there, however many frames they are seen in; it keeps their best picture."""
        for ticket, held in self.tickets.items():
            if self._alike(held["face"], face) >= SAME_FACE:
                if face.quality > held["face"].quality:
                    held["face"] = face
                held["until"] = time.time() + TTL_S
                return ticket
        ticket = uuid.uuid4().hex
        self.tickets[ticket] = {"face": face, "until": time.time() + TTL_S}
        return ticket

    @staticmethod
    def _alike(a: Face, b: Face) -> float:
        x, y = a.signature.ravel(), b.signature.ravel()
        return float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-9))

    def _who(self, token: str | None) -> Profile | None:
        held = self.passes.get(token or "")
        return self.faces.profiles.get(held["profile"]) if held else None

    # ------------------------------------------------------------------ registering and letting in

    def admit(self, token: str | None) -> dict:
        profile = self._who(token)
        if profile is None:
            return {"status": "error", "message": "Please look at the camera again."}
        if not self.faces.can_pay(profile):
            return {"status": "needs_method", "profile": self.faces.summary(profile)}
        held = self.passes.get(token or "", {})
        self.faces.admit(profile, held.get("shot"), held.get("score"))
        return {"status": "admitted", "profile": self.faces.summary(profile), "score": held.get("score")}

    async def register(self, ticket: str | None = None, token: str | None = None, name: str = "", email: str = "", method: str = "paypal", test_card: str = "", card: dict | None = None, return_base: str = "") -> dict:
        """A new face (`ticket`) becomes a profile, or a known one (`token`) gets a payment method, and is let in.
        A card is saved at once. A PayPal account answers {status: approve, reg, approve_url}: the shopper approves
        there, and `approved(reg)` finishes the job."""
        known = self._who(token)
        held = self.tickets.get(ticket or "")
        if known is None and held is None:
            return {"status": "error", "message": "Please look at the camera again."}
        if known is None:
            # registered a moment ago, or recognised now from a better picture. Only a sure match: someone who
            # merely looks like a registered shopper must get a profile of their own, not that shopper's.
            again = self.faces.who(held["face"], floor=SURE)
            if again:
                known = again[0]
        name, email = name.strip(), email.strip()
        if known is None and not name:
            return {"status": "error", "field": "name", "message": "Enter your name."}
        if known is None or not known.email or email:
            if not valid_email(email):
                return {"status": "error", "field": "email", "message": "Enter an email address the invoice can go to."}
        pay = self.payments
        if pay is None or not pay.enabled:
            return {"status": "error", "message": pay.problem if pay else "Payments are not set up."}
        ref = known.id if known else (ticket or uuid.uuid4().hex)[:10]
        seen = held if held else self.passes.get(token or "", {})
        details = {"profile": known.id if known else None, "face": None if known else held["face"], "ticket": ticket, "name": name, "email": email, "shot": seen.get("shot"), "score": seen.get("score")}
        try:
            if method == "card":
                saved = await pay.save_card(name or known.name, test_card, f"{ref}-{uuid.uuid4().hex[:6]}", email or None, details=card or None)
                return self._let_in(details, saved)
            reg = uuid.uuid4().hex
            back = f"{return_base.rstrip('/')}/gate/return?reg={reg}"
            begun = await pay.begin_wallet(f"{ref}-{reg[:6]}", back, back + "&cancelled=1", self.shop)
            self.approvals[reg] = {**details, "setup": begun["id"], "state": "waiting", "result": None, "until": time.time() + TTL_S}
            return {"status": "approve", "reg": reg, "approve_url": begun["approve_url"]}
        except ValueError as e:
            return {"status": "error", "field": "card", "message": str(e)[:1].upper() + str(e)[1:] + "."}
        except Exception as e:
            return {"status": "error", "message": self._explain(e)}

    async def approved(self, reg: str, cancelled: bool = False) -> dict:
        """PayPal sent the shopper back. Turn the approval into a saved method, then let them in."""
        held = self.approvals.get(reg)
        if held is None:
            return {"status": "error", "message": "That registration has expired. Please start again at the gate."}
        if held["state"] != "waiting":
            return held["result"]
        if cancelled:
            held["state"], held["result"] = "cancelled", {"status": "cancelled", "message": "PayPal approval was cancelled. Nothing was saved."}
            return held["result"]
        held["state"] = "saving"
        try:
            saved = await self.payments.finish_wallet(held["setup"])
            held["result"] = self._let_in(held, saved)
        except Exception as e:
            held["result"] = {"status": "error", "message": self._explain(e)}
        held["state"] = held["result"]["status"]
        return held["result"]

    def progress(self, reg: str) -> dict:
        held = self.approvals.get(reg)
        if held is None:
            return {"status": "error", "message": "That registration has expired. Please start again at the gate."}
        return held["result"] if held["result"] else {"status": "waiting"}

    def _let_in(self, details: dict, saved) -> dict:
        book = self.faces
        profile = book.profiles.get(details.get("profile") or "")
        if profile is None:
            profile = book.add(details["name"], details["face"], details["email"])
            self.tickets.pop(details.get("ticket") or "", None)
        elif details.get("email") and details["email"] != profile.email:
            profile.email = details["email"]
            book.db.execute("UPDATE profiles SET email = ? WHERE id = ?", (profile.email, profile.id))
        book.set_card(profile, saved.customer, saved.payment_method, saved.brand, saved.last4, getattr(saved, "payer", ""))
        book.admit(profile, details.get("shot"), details.get("score"))
        return {"status": "admitted", "new": details.get("profile") is None, "profile": book.summary(profile)}

    @staticmethod
    def _explain(error: Exception) -> str:
        issue = getattr(error, "issue", "")
        if issue in FRIENDLY:
            return FRIENDLY[issue]
        text = str(error) or type(error).__name__
        return f"The payment method was not saved: {text[:160]}"


def decode(jpeg: bytes) -> np.ndarray | None:
    """A camera frame as the browser sent it."""
    if not jpeg:
        return None
    return cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)

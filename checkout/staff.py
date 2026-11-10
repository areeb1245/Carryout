"""
Staff sign-in: one staff code, traded for a token.

The shop's staff screens (floor, catalogue, dashboard) and everything they call are private. The gate, where
shoppers register and check in, stays public. A member of staff types the staff code (`STAFF_CODE`) once; the server
answers with a signed token (a JWT, HS256, with `JWT_SECRET`) that it also sets as an HttpOnly cookie, so the
browser sends it with every page, call and socket. The token can be sent as `Authorization: Bearer <token>` too.

A signed token alone would stay good until it expires, whatever happens. So each one is also written down: its id,
a random salt, and the SHA-256 of salt + token. A token is accepted only if its signature and expiry hold, its id is
on file, not revoked and not expired, and the hash of what was sent matches. The token itself is never stored.
Signing out revokes it.

Wrong codes are throttled per address: `MAX_TRIES` in `TRY_WINDOW_S`.
"""
from __future__ import annotations

import hashlib
import hmac
import secrets
import sqlite3
import threading
import time
import uuid
from pathlib import Path

import jwt

ISSUER = "paypal-hackathon"
LIFETIME_S = 12 * 3600  # one shift
MAX_TRIES = 5
TRY_WINDOW_S = 600
MIN_CODE = 8
COOKIE = "staff"


class StaffError(RuntimeError):
    pass


def _digest(text: str) -> bytes:
    return hashlib.sha256(text.encode()).digest()


class Staff:
    def __init__(self, code: str, secret: str, path: Path, required: bool = False):
        """`code` and `secret` from the environment. `required`: a hosted copy, which must not start without
        both. Without a code (on this computer only) nobody can sign in, and the server leaves its own screens
        open to this computer alone (server.py)."""
        code, secret = (code or "").strip(), (secret or "").strip()
        if required and (len(code) < MIN_CODE or len(secret) < 32):
            raise StaffError(f"STAFF_CODE (at least {MIN_CODE} characters) and JWT_SECRET (at least 32) must be set")
        self.enabled = bool(code)
        self._code = _digest(code) if code else None
        # without a secret of its own, tokens are signed with a random one: they last until the server restarts
        self._secret = secret or secrets.token_urlsafe(48)
        self.tries: dict[str, list[float]] = {}
        self.lock = threading.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS staff_tokens (jti TEXT PRIMARY KEY, salt BLOB NOT NULL, hash BLOB NOT NULL, "
            "issued REAL NOT NULL, expires REAL NOT NULL, revoked REAL, address TEXT)"
        )
        self.db.commit()

    # ---------------------------------------------------------------- signing in

    def throttled(self, address: str) -> bool:
        now = time.time()
        with self.lock:
            recent = [t for t in self.tries.get(address, []) if now - t < TRY_WINDOW_S]
            self.tries[address] = recent
            return len(recent) >= MAX_TRIES

    def sign_in(self, code: str, address: str = "") -> tuple[str, float] | None:
        """The staff code -> (token, expiry time), or None when it is wrong (or sign-in is not set up)."""
        if not self.enabled or not isinstance(code, str):
            return None
        if not hmac.compare_digest(_digest(code.strip()), self._code):
            with self.lock:
                self.tries.setdefault(address, []).append(time.time())
            return None
        with self.lock:
            self.tries.pop(address, None)
        now = time.time()
        jti = uuid.uuid4().hex
        token = jwt.encode({"iss": ISSUER, "sub": "staff", "jti": jti, "iat": int(now), "exp": int(now + LIFETIME_S)}, self._secret, algorithm="HS256")
        salt = secrets.token_bytes(16)
        with self.lock:
            self.db.execute(
                "INSERT INTO staff_tokens (jti, salt, hash, issued, expires, address) VALUES (?, ?, ?, ?, ?, ?)",
                (jti, salt, hashlib.sha256(salt + token.encode()).digest(), now, now + LIFETIME_S, address[:64]),
            )
            self.db.execute("DELETE FROM staff_tokens WHERE expires < ?", (now - 7 * 86400,))  # keep a week of history
            self.db.commit()
        return token, now + LIFETIME_S

    # ---------------------------------------------------------------- checking a token

    def check(self, token: str | None) -> str | None:
        """The token's id if it is good, else None."""
        if not token:
            return None
        try:
            claims = jwt.decode(token, self._secret, algorithms=["HS256"], issuer=ISSUER, options={"require": ["exp", "iat", "jti", "sub"]})
        except jwt.PyJWTError:
            return None
        if claims.get("sub") != "staff":
            return None
        with self.lock:
            row = self.db.execute("SELECT salt, hash, expires, revoked FROM staff_tokens WHERE jti = ?", (str(claims["jti"]),)).fetchone()
        if row is None:
            return None
        salt, stored, expires, revoked = row
        if revoked is not None or expires < time.time():
            return None
        if not hmac.compare_digest(hashlib.sha256(salt + token.encode()).digest(), stored):
            return None
        return str(claims["jti"])

    def sign_out(self, token: str | None) -> bool:
        jti = self.check(token)
        if jti is None:
            return False
        with self.lock:
            self.db.execute("UPDATE staff_tokens SET revoked = ? WHERE jti = ?", (time.time(), jti))
            self.db.commit()
        return True

    def close(self) -> None:
        self.db.close()


def token_from(headers, cookies) -> str | None:
    """The token a request carries: a Bearer header, else the cookie."""
    auth = headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip() or None
    return cookies.get(COOKIE) or None

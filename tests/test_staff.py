"""Staff sign-in: the code for a token, the token checked against its salted hash on file, and the guard in front of
every route: the gate public, everything else for signed-in staff only."""
import asyncio
import hashlib
import re
import sqlite3

import httpx
import jwt
import pytest
from fastapi.routing import APIRoute, APIWebSocketRoute
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from checkout import server
from checkout.staff import COOKIE, MAX_TRIES, MAX_TRIES_ALL, Staff, StaffError

CODE = "open-sesame-42"
SECRET = "s" * 40


@pytest.fixture
def staff(tmp_path):
    keeper = Staff(CODE, SECRET, tmp_path / "staff.db")
    yield keeper
    keeper.close()


def test_the_code_buys_a_token_that_is_stored_only_as_a_salted_hash(staff, tmp_path):
    assert staff.sign_in("wrong code") is None
    token, expires = staff.sign_in(CODE, "1.2.3.4")
    assert staff.check(token)
    row = sqlite3.connect(tmp_path / "staff.db").execute("SELECT salt, hash, address FROM staff_tokens").fetchone()
    salt, stored, where = row
    assert len(salt) == 16 and stored == hashlib.sha256(salt + token.encode()).digest() and where == "1.2.3.4"
    assert token.encode() not in b"".join(x if isinstance(x, bytes) else str(x).encode() for x in row)  # never the token itself


def test_a_token_is_refused_once_signed_out_tampered_with_or_unknown(staff):
    token, _ = staff.sign_in(CODE)
    head, body, sig = token.split(".")
    assert staff.check(f"{head}.{body}.{sig[:-2]}AA") is None  # the signature does not hold
    claims = jwt.decode(token, SECRET, algorithms=["HS256"], issuer="paypal-hackathon")
    forged = jwt.encode({**claims, "jti": "f" * 32}, SECRET, algorithm="HS256")
    assert staff.check(forged) is None  # signed right, but never issued
    assert staff.check(jwt.encode(claims, "another secret" * 4, algorithm="HS256")) is None
    assert staff.check(jwt.encode(claims, SECRET, algorithm="HS512")) is None  # only HS256 is taken
    assert staff.sign_out(token) and staff.check(token) is None
    assert staff.check(None) is None and staff.check("not a token") is None


def test_an_expired_token_is_refused(staff, tmp_path):
    token, _ = staff.sign_in(CODE)
    staff.db.execute("UPDATE staff_tokens SET expires = 0")
    assert staff.check(token) is None


def test_wrong_codes_are_throttled(staff):
    for _ in range(MAX_TRIES):
        assert not staff.throttled("9.9.9.9")
        staff.sign_in("guess", "9.9.9.9")
    assert staff.throttled("9.9.9.9") and not staff.throttled("8.8.8.8")


def test_guessing_from_many_made_up_addresses_is_capped(staff):
    for n in range(MAX_TRIES_ALL):
        staff.sign_in("guess", f"10.0.{n // 250}.{n % 250}")
    assert staff.throttled("172.16.0.1")  # a fresh address too


def test_a_hosted_copy_will_not_start_without_a_code_and_a_secret(tmp_path):
    with pytest.raises(StaffError):
        Staff("", "", tmp_path / "a.db", required=True)
    with pytest.raises(StaffError):
        Staff("short", SECRET, tmp_path / "b.db", required=True)
    Staff(CODE, SECRET, tmp_path / "c.db", required=True).close()


# ------------------------------------------------------------------------------------------------ the guard


def ask(method, path, client=("203.0.113.5", 4000), **kw):
    async def go():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app, client=client), base_url="http://shop.test") as c:
            return await c.request(method, path, **kw)

    return asyncio.run(go())


@pytest.fixture
def guarded(staff, monkeypatch):
    monkeypatch.setattr(server.app.state, "staff", staff, raising=False)
    return staff


def filled(path: str) -> str:
    return re.sub(r"\{[^}]+\}", "x1", path)


def test_every_route_but_the_gate_needs_staff(guarded):
    """Walks every route of the real app: none outside the public list answers without a token."""
    checked = 0
    for route in server.app.routes:
        if not isinstance(route, APIRoute):
            continue
        path = filled(route.path)
        if route.path in server.PUBLIC or route.path.startswith(server.PUBLIC_UNDER):
            continue
        for method in route.methods - {"HEAD"}:
            r = ask(method, path, headers={"accept": "text/html"})
            assert r.status_code in (401, 303), (method, route.path, r.status_code)
            r = ask(method, path)
            assert r.status_code == 401, (method, route.path, r.status_code)
            checked += 1
    assert checked >= 29  # every staff route there is today
    for mounted in ("/samples/bookshop-demo.mp4",):
        assert ask("GET", mounted).status_code == 401
    assert ask("GET", "/dashboard", headers={"accept": "text/html"}).headers["location"] == "/signin?next=%2Fdashboard"


def test_the_gate_and_sign_in_stay_public(guarded):
    assert ask("GET", "/healthz").json() == {"ok": True}
    assert ask("GET", "/signin").status_code == 200
    assert ask("GET", "/gate").status_code == 200
    assert ask("GET", "/static/gate.js").status_code == 200
    assert ask("GET", "/", headers={"host": "gate.shop.test"}).status_code == 200  # the gate's own address


def test_signing_in_opens_the_staff_screens_and_signing_out_closes_them(guarded):
    async def go():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app, client=("203.0.113.5", 4000)), base_url="https://shop.test") as c:
            assert (await c.post("/api/staff/signin", json={"code": "nope"})).status_code == 401
            r = await c.post("/api/staff/signin", json={"code": CODE}, headers={"x-forwarded-proto": "https"})
            assert r.status_code == 200 and r.json()["token"]
            cookie = r.headers["set-cookie"].lower()
            assert "httponly" in cookie and "samesite=lax" in cookie and "secure" in cookie
            assert (await c.get("/api/staff/me")).json()["signed_in"]
            assert (await c.get("/api/clips")).status_code == 200  # a staff call, with the cookie
            token = r.json()["token"]
            c.cookies.clear()
            assert (await c.get("/api/clips", headers={"authorization": f"Bearer {token}"})).status_code == 200
            await c.post("/api/staff/signout", headers={"authorization": f"Bearer {token}"})
            assert (await c.get("/api/clips", headers={"authorization": f"Bearer {token}"})).status_code == 401

    asyncio.run(go())


def test_too_many_wrong_codes_are_turned_away(guarded):
    for _ in range(MAX_TRIES):
        assert ask("POST", "/api/staff/signin", json={"code": "guess"}).status_code == 401
    assert ask("POST", "/api/staff/signin", json={"code": CODE}).status_code == 429


def test_without_a_staff_code_only_this_computer_gets_in(tmp_path, monkeypatch):
    keeper = Staff("", "", tmp_path / "open.db")
    monkeypatch.setattr(server.app.state, "staff", keeper, raising=False)
    assert ask("GET", "/api/clips", client=("127.0.0.1", 5000)).status_code == 200
    assert ask("GET", "/api/clips", client=("203.0.113.5", 5000)).status_code == 503  # through a tunnel: not set up
    # a forwarded header (any tunnel or proxy adds one) is never taken for this computer
    assert ask("GET", "/api/clips", client=("127.0.0.1", 5000), headers={"x-forwarded-for": "127.0.0.1"}).status_code == 503
    monkeypatch.setattr(server, "HOSTED", True)
    assert ask("GET", "/api/clips", client=("127.0.0.1", 5000)).status_code == 503  # hosted: never open
    keeper.close()


def test_the_floor_socket_needs_staff_and_this_servers_page(guarded):
    client = TestClient(server.app)  # not entered: the shop's services are not started
    route = next(r for r in server.app.routes if isinstance(r, APIWebSocketRoute))
    assert route.path == "/ws"
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/ws"):
            pass
    token, _ = guarded.sign_in(CODE)
    with pytest.raises(WebSocketDisconnect):  # signed in, but opened from another site's page
        with client.websocket_connect("/ws", headers={"authorization": f"Bearer {token}", "origin": "https://evil.example"}):
            pass
    assert COOKIE == "staff"

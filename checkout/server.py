"""
The live app: two screens and the calls behind them.

    uv run uvicorn checkout.server:app --port 8765

The gate (/gate, or gate.localhost:8765)   where a shopper is recognised or registers, and is let in
    POST /api/gate/look        one camera frame (a JPEG) -> who this is: no_face | closer | known | new
    POST /api/gate/register    a new face plus name, email and a payment method -> admitted, or a PayPal address to approve at
    GET  /gate/return          where PayPal sends the shopper back after approving
    GET  /api/gate/progress/x  has that approval come through yet?
    POST /api/gate/admit       a recognised face with a method on file -> an open visit

The dashboard (/dashboard)   for the shop, not for shoppers: sales, every transaction with its proof, the customers
    GET  /api/dashboard                    the totals, sales by day, top products, the transactions, the customers
    GET  /api/dashboard/transactions/7     one transaction: its lines, the steps that led to the bill, its pictures
    GET  /api/dashboard/proof/7/0          one of those pictures
    POST /api/dashboard/transactions/7/refund   {"line": 31} gives one line back, {} what is left of the bill
    POST /api/dashboard/transactions/7/charge   {"line": 32, "product": "deep-work"} charges a held-back line by hand
    POST /api/dashboard/transactions/7/clear    {"line": 32} marks a held-back line as not taken

The catalogue (/catalogue)   for the shop: the products, their pictures, and footage to pick new ones out of
    PUT  /api/catalog/bookshop             the whole catalogue
    POST /api/examples/bookshop            a JPEG: a picture of a product -> {file}, then listed under its `examples`
    GET  /api/examples/bookshop/ex-x.jpg   one of those pictures
    POST /api/clips                        a video (the body), or {"sample": "IMG_6401.MOV"} -> its frames
    GET  /api/clips/x/frames/3             one frame
    POST /api/clips/x/frames/3/look        {prompts, example?: box} -> SAM's outlines on that frame
    POST /api/clips/x/describe             {frame, box} -> the second model's name, phrases, category for the object
    POST /api/clips/x/phrases              {frame, box, phrases} -> which phrase SAM finds that object by
    POST /api/clips/live                   a clip for a camera, frames sent one by one:
    POST /api/clips/x/frames?t=1.4         a JPEG -> its number and the item in hand SAM sees on it
    POST /api/clips/x/auto                 {views: [{frame, box}]} -> the pictures to keep, what it is, its best phrase

The shop floor (/)   frames from the browser over a WebSocket, checkout states back
    Browser -> server   {"type": "start", "shop": "bookshop", "catalog": {...}, "dry": bool, "live": bool}   new visit; `catalog` overrides the shop file
                        {"type": "frame", "seq": n, "t": seconds}   then one binary message: the JPEG
                        {"type": "end"}                             the footage is over: everyone walks out and pays
    Server -> browser   {"type": "ready", session}
                        {"type": "state", ...}       one per frame, in frame order: people, hands, items, events, carts, meta
                        {"type": "update", ...}      between frames: a face, a second look, the last check or a payment came through
                        {"type": "finished", ...}    after "end", once every payment has been attempted
                        {"type": "lost", seq, error} the model call for that frame failed

Who may use what (`guard`)
    Public          the gate and its calls, the PayPal return page, the staff sign-in page, /healthz, the static files
    Staff only      everything else: the floor, the catalogue, the dashboard, their calls and the floor's socket.
                    The staff code is traded for a token (staff.py), kept as an HttpOnly cookie or sent as a Bearer
                    header. Pages answer an unsigned-in visitor with the sign-in page, calls with 401.
    POST /api/staff/signin   {"code": "..."} -> {token, expires}, and the cookie
    POST /api/staff/signout  revokes the token, clears the cookie
    GET  /api/staff/me       200 while signed in

Without a STAFF_CODE (on this computer, for development) nobody can sign in, and the staff screens are open to this
computer only: a request through a tunnel is refused. A hosted copy (Render) does not start without STAFF_CODE and
JWT_SECRET. Every key (SAM, the verifier, PayPal, the database) stays in this process.
"""
from __future__ import annotations

import asyncio
import dataclasses
import types
import html
import json
import logging
import os
import time
import uuid
from urllib.parse import quote
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from . import clips
from .catalog import Catalog
from .gate import Gate, decode
from .identity import Profiles
from .ledger import Ledger
from .mailer import Mailer
from .payments import TEST_CARDS, Payments
from .pipeline import Services, Session
from . import paths
from .paths import DATA, SHOPS
from .reader import Reader
from .staff import COOKIE, Staff, StaffError, token_from
from .sam_client import MAX_FPS, MAX_PROMPTS, ROOT, SamClient, load_env
from .vectors import FaceIndex
from .verify import Verifier

log = logging.getLogger("checkout")
WEB = ROOT / "web"
SAMPLES = ROOT / "samples"
DEFAULT_SHOP = "bookshop"
MAX_FRAME_BYTES = 2 * 1024 * 1024
MAX_IN_FLIGHT = 6
HOSTED = bool(os.environ.get("RENDER") or os.environ.get("REQUIRE_STAFF"))  # Render sets RENDER=true


@asynccontextmanager
async def lifespan(app: FastAPI):
    paths.prepare()
    env = load_env()
    try:
        app.state.staff = Staff(env.get("STAFF_CODE", ""), env.get("JWT_SECRET", ""), paths.DATABASE, required=HOSTED)
    except StaffError as e:
        raise RuntimeError(f"Not starting: {e}") from None
    index = FaceIndex()
    payments = Payments()
    services = Services(SamClient(), Verifier(), Profiles(provider=payments.name, index=index), payments, Mailer(), Reader())
    app.state.services = services
    app.state.ledger = Ledger(services.faces)
    app.state.refunding = set()  # transactions with a refund, and lines with a charge, on their way to PayPal
    app.state.sessions = {}  # the last few visits, so the page can fetch the pictures a bag was checked against
    app.state.gate = Gate(services.faces, payments, Catalog.load(shop_file(DEFAULT_SHOP)).name)
    app.state.clip_sam = SamClient(cache_dir=DATA / "clip-cache")  # the catalogue screen's looks, cached
    yield
    app.state.staff.close()
    await app.state.clip_sam.close()
    await services.sam.close()
    await services.verifier.close()
    await services.mailer.close()
    await payments.close()
    index.close()


app = FastAPI(title="SAM 3.1 checkout", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)


def tunnelled(request: Request) -> bool:
    """Did this request come in through a tunnel, from somewhere other than this computer? The tunnel's agent
    connects from this machine and says who it is forwarding for."""
    host = request.client.host if request.client else ""
    return bool(request.headers.get("x-forwarded-for")) or host not in ("127.0.0.1", "::1", "localhost")


PUBLIC = {"/gate", "/gate/return", "/signin", "/api/staff/signin", "/api/staff/signout", "/healthz", "/favicon.ico"}
PUBLIC_UNDER = ("/static/", "/api/gate/")


def public(request: Request) -> bool:
    path = request.url.path
    if path in PUBLIC or path.startswith(PUBLIC_UNDER):
        return True
    return path == "/" and request.headers.get("host", "").lower().startswith("gate.")  # the gate's own address


def address(request: Request) -> str:
    """Who is asking, for throttling sign-in: the first forwarded address behind a proxy, else the connection's."""
    forwarded = request.headers.get("x-forwarded-for", "")
    return forwarded.split(",")[0].strip() if forwarded else (request.client.host if request.client else "")


def staff_ok(request) -> bool:
    """Signed-in staff, or (with no staff code set up, on this computer only) this computer itself."""
    staff: Staff | None = getattr(app.state, "staff", None)
    if staff is None:
        return False
    if staff.check(token_from(request.headers, request.cookies)):
        return True
    return not staff.enabled and not HOSTED and not tunnelled(request)


@app.middleware("http")
async def guard(request: Request, call_next):
    if public(request) or staff_ok(request):
        return await call_next(request)
    staff: Staff | None = getattr(app.state, "staff", None)
    if staff is not None and not staff.enabled:
        return JSONResponse({"error": "Staff sign-in is not set up: set STAFF_CODE and JWT_SECRET."}, status_code=503)
    path = request.url.path
    if request.method == "GET" and not path.startswith("/api/") and "text/html" in request.headers.get("accept", ""):
        target = path + (f"?{request.url.query}" if request.url.query else "")
        return Response(status_code=303, headers={"Location": f"/signin?next={quote(target, safe='')}", "Cache-Control": "no-store"})
    return JSONResponse({"error": "Sign in as staff first."}, status_code=401, headers={"WWW-Authenticate": "Bearer"})


@app.get("/healthz")
def healthz():
    """For the host's health checks: touches nothing, so it never wakes the GPU (that is /api/health)."""
    return {"ok": True}


@app.get("/signin")
def signin_page():
    return FileResponse(WEB / "signin.html", headers={"Cache-Control": "no-store"})


@app.post("/api/staff/signin")
async def staff_signin(request: Request, body: dict):
    staff: Staff = app.state.staff
    who = address(request)
    if not staff.enabled:
        return JSONResponse({"error": "Staff sign-in is not set up: set STAFF_CODE and JWT_SECRET."}, status_code=503)
    if staff.throttled(who):
        return JSONResponse({"error": "Too many wrong codes. Try again in a few minutes."}, status_code=429)
    issued = staff.sign_in(str(body.get("code", ""))[:200], who)
    if issued is None:
        return JSONResponse({"error": "That staff code is not right."}, status_code=401)
    token, expires = issued
    answer = JSONResponse({"token": token, "expires": int(expires)}, headers={"Cache-Control": "no-store"})
    secure = request.headers.get("x-forwarded-proto", request.url.scheme) == "https"
    answer.set_cookie(COOKIE, token, max_age=int(expires - time.time()), httponly=True, samesite="lax", secure=secure, path="/")
    return answer


@app.post("/api/staff/signout")
async def staff_signout(request: Request):
    app.state.staff.sign_out(token_from(request.headers, request.cookies))
    answer = JSONResponse({"signed_out": True})
    answer.delete_cookie(COOKIE, path="/")
    return answer


@app.get("/api/staff/me")
async def staff_me(request: Request):
    staff: Staff = app.state.staff
    return {"staff": True, "signed_in": bool(staff.check(token_from(request.headers, request.cookies))), "sign_in": staff.enabled}


def shop_file(shop: str):
    """The catalogue file of a shop, or None for a name that is not one of the files in shops/."""
    known = {p.stem: p for p in SHOPS.glob("*.json")}
    return known.get(shop)


def sample_list() -> list[dict]:
    manifest = SAMPLES / "samples.json"
    listed = json.loads(manifest.read_text()) if manifest.exists() else {}
    return [{"file": name, "title": info["title"], "shop": info.get("shop", DEFAULT_SHOP)} for name, info in listed.items() if (SAMPLES / name).exists()]


@app.get("/")
def index(request: Request):
    # gate.localhost:8765 is the gate's own address; everything else gets the shop floor
    page = "gate.html" if request.headers.get("host", "").lower().startswith("gate.") else "index.html"
    return FileResponse(WEB / page, headers={"Cache-Control": "no-store"})


@app.get("/gate")
def gate_page():
    return FileResponse(WEB / "gate.html", headers={"Cache-Control": "no-store"})


@app.post("/api/gate/look")
async def gate_look(request: Request):
    jpeg = await request.body()
    if len(jpeg) > MAX_FRAME_BYTES:
        return JSONResponse({"status": "error", "message": "That picture is too large."}, status_code=413)
    return await app.state.gate.look(decode(jpeg))


@app.post("/api/gate/register")
async def gate_register(request: Request, body: dict):
    card = body.get("card") if isinstance(body.get("card"), dict) else None
    # PayPal sends the shopper back to this server. Reached through a tunnel, that is the tunnel's own https
    # address; otherwise "localhost", because PayPal will not take a made-up host name (gate.localhost).
    host = request.headers.get("host", "")
    if tunnelled(request) and host:
        back = f"{request.headers.get('x-forwarded-proto', 'https')}://{host}"
    else:
        back = f"http://localhost:{request.url.port or 80}"
    return await app.state.gate.register(
        ticket=body.get("ticket"), token=body.get("pass"), name=str(body.get("name", "")), email=str(body.get("email", "")),
        method=str(body.get("method", "paypal")), test_card=str(body.get("test_card", "")), card=card, return_base=back,
    )


@app.get("/api/gate/progress/{reg}")
async def gate_progress(reg: str):
    return app.state.gate.progress(reg)


@app.post("/api/gate/admit")
async def gate_admit(body: dict):
    return app.state.gate.admit(body.get("pass"))


@app.get("/gate/return")
async def gate_return(reg: str = "", cancelled: str = ""):
    """PayPal sends the shopper here after the approval window. The gate screen is still open in the other window
    and picks the result up by itself; this page only says what happened."""
    result = await app.state.gate.approved(reg, cancelled=bool(cancelled))
    done = result["status"] == "admitted"
    title = "PayPal is set up" if done else "PayPal was not set up"
    text = "You can close this window and walk in." if done else html.escape(result.get("message", "Please try again at the gate."))
    page = (WEB / "return.html").read_text().replace("{{title}}", title).replace("{{text}}", text).replace("{{state}}", "ok" if done else "bad")
    return HTMLResponse(page, headers={"Cache-Control": "no-store"})


@app.get("/dashboard")
def dashboard_page():
    return FileResponse(WEB / "dashboard.html", headers={"Cache-Control": "no-store"})


@app.get("/api/dashboard")
async def dashboard(rehearsals: bool = False):
    ledger: Ledger = app.state.ledger
    shop = Catalog.load(shop_file(DEFAULT_SHOP))
    return {
        "shop": shop.name,
        "summary": ledger.summary(shop.currency),
        "transactions": ledger.transactions(rehearsals=rehearsals),
        "customers": ledger.customers(shop.currency),
        "services": app.state.services.status(),
    }


def catalogue_of(shop: str | None) -> Catalog:
    """The catalogue of the shop a transaction was made in (by its name), or of the default shop."""
    for path in SHOPS.glob("*.json"):
        catalog = Catalog.load(path)
        if shop and catalog.name == shop:
            return catalog
    return Catalog.load(shop_file(DEFAULT_SHOP))


def with_charging(found: dict) -> dict:
    """A transaction as the dashboard shows it: with whether its held-back lines can still be charged by hand, and
    the catalogue to choose the item from."""
    ok, why_not = app.state.ledger.chargeable(found["id"])
    pay = app.state.services.payments
    if ok and (pay is None or not pay.enabled):
        ok, why_not = False, pay.problem if pay else "Payments are not set up."
    catalog = catalogue_of(found["shop"])
    found["charging"] = {"ok": ok, "note": why_not, "currency": catalog.currency.lower(), "products": [{"id": p.id, "name": p.name, "price": p.price} for p in catalog.products if p.billable and p.price > 0]}
    return found


@app.get("/api/dashboard/transactions/{visit}")
async def dashboard_transaction(visit: int):
    found = app.state.ledger.transaction(visit)
    return with_charging(found) if found else JSONResponse({"error": "There is no such transaction."}, status_code=404)


@app.post("/api/dashboard/transactions/{visit}/charge")
async def dashboard_charge(visit: int, body: dict, request: Request):
    """Charge a line that was held back at the door, now that a person has looked at its pictures and says which
    item it was. It becomes a transaction of its own, with its own invoice and email."""
    ledger: Ledger = app.state.ledger
    services: Services = app.state.services
    found = ledger.transaction(visit)
    line = body.get("line")
    if found is None or not isinstance(line, int):
        return JSONResponse({"error": "There is no such line."}, status_code=404)
    if (visit, line) in app.state.refunding:
        return JSONResponse({"error": "That line is being charged right now."}, status_code=409)
    catalog = catalogue_of(found["shop"])
    app.state.refunding.add((visit, line))
    try:
        result = await ledger.charge_later(visit, line, catalog.by_id.get(str(body.get("product") or "")), catalog.currency, catalog.name, services.payments, services.mailer)
    finally:
        app.state.refunding.discard((visit, line))
    if result["status"] != "succeeded":
        said = {"declined": "The shopper's payment method was declined.", "failed": f"The payment did not go through: {result['message']}"}.get(result["status"], result["message"])
        return JSONResponse({"error": said, "transaction": with_charging(ledger.transaction(visit))}, status_code=409 if result["status"] == "refused" else 502)
    return {"charged": result["visit"], "transaction": with_charging(ledger.transaction(visit))}


@app.post("/api/dashboard/transactions/{visit}/clear")
async def dashboard_clear(visit: int, body: dict, request: Request):
    """A held-back line was looked at and was not taken: it leaves the review queue, uncharged."""
    line = body.get("line")
    ok, why_not = app.state.ledger.clear(visit, line) if isinstance(line, int) else (False, "There is no such line.")
    if not ok:
        return JSONResponse({"error": why_not}, status_code=409)
    return {"transaction": with_charging(app.state.ledger.transaction(visit))}


@app.get("/api/dashboard/proof/{visit}/{n}")
async def dashboard_proof(visit: int, n: int):
    picture = app.state.ledger.proof(visit, n)
    if not picture:
        return Response(status_code=404)
    return Response(picture, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=3600"})


@app.post("/api/dashboard/transactions/{visit}/refund")
async def dashboard_refund(visit: int, body: dict, request: Request):
    """Give money back through PayPal: the price of one line ({"line": id}), or what is left of the bill. The
    ledger decides how much can still go back, so a line is never refunded twice."""
    ledger: Ledger = app.state.ledger
    pay = app.state.services.payments
    line = body.get("line")
    line = int(line) if isinstance(line, (int, str)) and str(line).isdigit() else None
    if visit in app.state.refunding:
        return JSONResponse({"error": "A refund for this transaction is on its way to PayPal. Try again in a moment."}, status_code=409)
    amount, why_not = ledger.refundable(visit, line)
    if amount <= 0:
        return JSONResponse({"error": why_not}, status_code=409)
    if pay is None or not pay.enabled:
        return JSONResponse({"error": pay.problem if pay else "Payments are not set up."}, status_code=503)
    found = ledger.transaction(visit)
    app.state.refunding.add(visit)
    try:
        name = next((p["name"] for p in found["lines"] if p["id"] == line), "")
        note = f"Refund for {name}" if name else f"Refund from {found['shop'] or 'the shop'}"
        # the key names what is refunded, so pressing twice is one refund
        result = await pay.refund(found["payment"], amount, found["currency"], f"refund-{visit}-{line or 'rest'}-{len(found['refunds'])}", note)
        ledger.add_refund(visit, line, amount, found["currency"], result, str(body.get("reason") or "")[:200])
    finally:
        app.state.refunding.discard(visit)
    if result["status"] == "failed":
        return JSONResponse({"error": f"PayPal did not make the refund: {result['message']}", "transaction": with_charging(ledger.transaction(visit))}, status_code=502)
    return {"refund": result, "transaction": with_charging(ledger.transaction(visit))}


@app.get("/api/gate/config")
async def gate_config():
    """What the public gate page needs: the shop's name, the sandbox test cards, which services are up."""
    path = shop_file(DEFAULT_SHOP)
    services: Services = app.state.services
    up = services.status()
    return {
        "shop": path.stem,
        "shops": [{"id": path.stem, "name": Catalog.load(path).name}],
        "test_cards": TEST_CARDS,
        "services": {k: up.get(k) for k in ("faces", "payments", "provider", "payments_note", "email")} | {"demo_email": ""},
    }


@app.get("/api/config")
async def config(shop: str = DEFAULT_SHOP):
    path = shop_file(shop) or shop_file(DEFAULT_SHOP)
    catalog = Catalog.load(path)
    services: Services = app.state.services
    return {
        "shop": path.stem,
        "shops": [{"id": p.stem, "name": json.loads(p.read_text()).get("name", p.stem)} for p in sorted(SHOPS.glob("*.json"), key=lambda p: p.stem != DEFAULT_SHOP)],
        "catalog": catalog.to_dict(),
        "samples": sample_list(),
        "max_prompts": MAX_PROMPTS,
        "max_fps": MAX_FPS,
        "test_cards": TEST_CARDS,
        "services": services.status(),
        "profiles": services.faces.all(),
        "gate_url": "/gate",
    }


@app.get("/api/health")
async def health():
    """Also wakes the GPU: the Modal app scales to zero, and the first call after idle takes up to a minute."""
    try:
        return await app.state.services.sam.health()
    except Exception as e:
        return JSONResponse({"status": "down", "error": str(e) or type(e).__name__}, status_code=503)


@app.put("/api/catalog/{shop}")
def save_catalog(shop: str, body: dict):
    path = shop_file(shop)
    if path is None:
        return JSONResponse({"error": "No such shop."}, status_code=404)
    catalog = Catalog.from_dict(body)
    if not any(p.billable for p in catalog.products):
        return JSONResponse({"error": "The catalogue needs at least one product."}, status_code=400)
    path.write_text(json.dumps(catalog.to_dict(), indent=2, ensure_ascii=False) + "\n")
    return {"catalog": catalog.to_dict()}


# ---------------------------------------------------------------------- the catalogue screen

MAX_EXAMPLE_BYTES = 1024 * 1024
MAX_CLIP_BYTES = 300 * 1024 * 1024


@app.get("/catalogue")
async def catalogue_page():
    return FileResponse(WEB / "catalogue.html", headers={"Cache-Control": "no-store"})


@app.post("/api/examples/{shop}")
async def add_example(shop: str, request: Request):
    """A picture of a product (a JPEG cut out of a frame), saved beside the shop's catalogue."""
    from .likeness import examples_dir

    if shop_file(shop) is None:
        return JSONResponse({"error": "No such shop."}, status_code=404)
    body = await request.body()
    if not body or len(body) > MAX_EXAMPLE_BYTES or body[:3] != b"\xff\xd8\xff":
        return JSONResponse({"error": "Send one JPEG under 1 MB."}, status_code=400)
    folder = examples_dir(shop)
    folder.mkdir(parents=True, exist_ok=True)
    name = f"ex-{uuid.uuid4().hex[:16]}.jpg"
    (folder / name).write_bytes(body)
    return {"file": name}


@app.get("/api/examples/{shop}/{name}")
async def example(shop: str, name: str):
    from .catalog import EXAMPLE_NAME
    from .likeness import examples_dir

    path = examples_dir(shop) / name if shop_file(shop) and EXAMPLE_NAME.match(name) else None
    if path is None or not path.exists():
        return Response(status_code=404)
    return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "max-age=86400"})


@app.get("/api/clips")
async def clip_list():
    return {"clips": clips.listed(), "samples": sample_list()}


@app.post("/api/clips")
async def add_clip(request: Request):
    """A video in the body (any type OpenCV reads), or JSON {"sample": name} for one of samples/."""
    kind = request.headers.get("content-type", "")
    if kind.startswith("application/json"):
        data = await request.json()
        listed = {s["file"]: s for s in sample_list()}
        name = str(data.get("sample") or "")
        if name not in listed:
            return JSONResponse({"error": "No such sample."}, status_code=404)
        source, title, scratch = SAMPLES / name, listed[name]["title"], None
    else:
        length = request.headers.get("content-length", "")
        if not length.isdigit() or int(length) > MAX_CLIP_BYTES:
            return JSONResponse({"error": "Send a video under 300 MB."}, status_code=413)
        clips.CLIPS.mkdir(parents=True, exist_ok=True)
        scratch = clips.CLIPS / f"upload-{uuid.uuid4().hex[:8]}"
        with scratch.open("wb") as out:
            async for chunk in request.stream():
                out.write(chunk)
        source, title = scratch, (request.headers.get("x-title") or "Imported clip")[:80]
    try:
        meta = await asyncio.to_thread(clips.cut, source, title)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    finally:
        if scratch is not None:
            scratch.unlink(missing_ok=True)
    return meta


@app.post("/api/clips/live")
async def live_clip(body: dict):
    return clips.start_live(str(body.get("title") or "Camera")[:80])


@app.post("/api/clips/{clip}/frames")
async def live_frame(clip: str, request: Request, t: float = 0.0, look: str = ""):
    """A camera frame for a live clip: kept, and SAM asked where the item in hand is. `look` adds phrases (comma
    separated) for what is being held, when the usual nouns miss it."""
    body = await request.body()
    if not body or len(body) > MAX_FRAME_BYTES:
        return JSONResponse({"error": "Send one JPEG under 2 MB."}, status_code=400)
    n = await asyncio.to_thread(clips.add_frame, clip, body, t)
    if n is None:
        return JSONResponse({"error": "This capture is full or closed. Start a new one."}, status_code=409)
    extra = [p.strip().lower()[:40] for p in look.split(",") if p.strip()][:4]
    prompts = list(dict.fromkeys(clips.HELD + extra))
    try:
        answer = await app.state.services.sam.segment(clips.frame_path(clip, n).read_bytes(), prompts)
    except Exception as e:
        return JSONResponse({"frame": n, "detections": [], "held": None, "error": str(e) or "The camera model did not answer."}, status_code=200)
    return {"frame": n, "detections": answer["detections"], "held": clips.held(answer["detections"]), "ms": answer.get("ms")}


@app.post("/api/clips/{clip}/auto")
async def clip_auto(clip: str, body: dict):
    """From the views a camera capture kept: the most different ones as pictures, the vision model's description of
    the clearest, and its phrases tried on every kept view. All the catalogue form needs but the price."""
    from .likeness import crop

    meta = clips.info(clip)
    views = [(int(v.get("frame", -1)), _box(v.get("box")), float(v.get("score") or 0)) for v in body.get("views") or []]
    views = [(n, box, score) for n, box, score in views if meta and 0 <= n < meta["frames"] and box][:24]
    if not views:
        return JSONResponse({"error": "No clear view of the item was kept. Hold it closer, in good light."}, status_code=400)
    frames = {n: decode(clips.frame_path(clip, n).read_bytes()) for n, _, _ in views}
    # the clearest view: sure of the outline, and big in the picture
    best = max(range(len(views)), key=lambda i: views[i][2] * ((views[i][1][2] - views[i][1][0]) * (views[i][1][3] - views[i][1][1])) ** 0.5)
    sam = app.state.clip_sam
    try:  # the views that differ most, by picture vectors; evenly spread if the endpoint cannot make vectors
        vectors = await app.state.services.sam.embed([crop(frames[n], box) for n, box, _ in views])
        keep = clips.diverse(vectors, 4, best)
    except Exception:
        keep = sorted(set(clips.spread(len(views), 4)) | {best})[:4]
    verifier: Verifier = app.state.services.verifier
    catalog = Catalog.load(shop_file(str(body.get("shop") or "")) or shop_file(DEFAULT_SHOP))
    known = sorted({f"{p.category} > {p.section}" if p.section else p.category for p in catalog.products if p.billable and p.category})
    n0, box0, _ = views[best]
    described = await verifier.describe(frames[n0], box0, known) if verifier and verifier.enabled else None
    ranked = []
    if described and described["phrases"]:
        tried = [views[i] for i in keep]
        try:
            answers = await asyncio.gather(*(sam.segment(clips.frame_path(clip, n).read_bytes(), described["phrases"]) for n, _, _ in tried))
            ranked = clips.rank_phrases(n0, box0, dict(zip([n for n, _, _ in tried], answers)), described["phrases"], {n: box for n, box, _ in tried})
        except Exception as e:
            log.warning("phrase test failed: %s", e)
    return {
        "pictures": [{"frame": views[i][0], "box": views[i][1]} for i in keep],
        "best": {"frame": n0, "box": box0},
        "described": described,
        "ranked": ranked,
        "note": "" if described else "The vision model is not set up or gave no answer, so name and phrase are left to you.",
    }


@app.get("/api/clips/{clip}")
async def clip_info(clip: str):
    meta = clips.info(clip)
    return meta or JSONResponse({"error": "No such clip."}, status_code=404)


@app.get("/api/clips/{clip}/frames/{n}")
async def clip_frame(clip: str, n: int):
    path = clips.frame_path(clip, n)
    if path is None:
        return Response(status_code=404)
    return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "max-age=86400"})


def _box(value) -> list[float] | None:
    try:
        box = [min(1.0, max(0.0, float(v))) for v in value][:4]
    except (TypeError, ValueError):
        return None
    return box if len(box) == 4 and box[2] - box[0] > 0.005 and box[3] - box[1] > 0.005 else None


@app.post("/api/clips/{clip}/frames/{n}/look")
async def clip_look(clip: str, n: int, body: dict):
    """SAM on one frame: the phrases given (or the generic ones and the shop's), and, with `example`, everything
    that looks like the object in that box."""
    path = clips.frame_path(clip, n)
    if path is None:
        return JSONResponse({"error": "No such frame."}, status_code=404)
    prompts = [str(p).strip().lower()[:60] for p in body.get("prompts") or [] if str(p).strip()][:24]
    example = _box(body.get("example")) if body.get("example") else None
    exemplars = [{"prompt": str(body.get("hint") or "")[:60], "label": "like the example", "boxes": [example]}] if example else None
    try:
        answer = await app.state.clip_sam.segment(path.read_bytes(), list(dict.fromkeys(prompts)), float(body.get("threshold") or 0.3), exemplars)
    except Exception as e:
        missing = "422" in str(e) or "404" in str(e)
        why = "The camera model on Modal is the older version, without examples. Deploy modal_app/sam_checkout.py." if exemplars and missing else (str(e) or "The camera model did not answer.")
        return JSONResponse({"error": why}, status_code=502)
    return {"frame": n, "detections": answer["detections"], "ms": answer.get("ms")}


@app.post("/api/clips/{clip}/describe")
async def clip_describe(clip: str, body: dict):
    path = clips.frame_path(clip, int(body.get("frame", -1)))
    box = _box(body.get("box"))
    if path is None or box is None:
        return JSONResponse({"error": "Pick an object on a frame first."}, status_code=400)
    verifier: Verifier = app.state.services.verifier
    if not verifier or not verifier.enabled:
        return JSONResponse({"error": "The vision model is not set up (LLM_URL in .env)."}, status_code=503)
    catalog = Catalog.load(shop_file(str(body.get("shop") or "")) or shop_file(DEFAULT_SHOP))
    known = sorted({f"{p.category} > {p.section}" if p.section else p.category for p in catalog.products if p.billable and p.category})
    frame = decode(path.read_bytes())
    answer = await verifier.describe(frame, box, known)
    return answer or JSONResponse({"error": "The vision model gave no usable answer. Try again, or a frame where the object is clearer."}, status_code=502)


@app.post("/api/clips/{clip}/phrases")
async def clip_phrases(clip: str, body: dict):
    """Ask SAM for each phrase on the chosen frame and two either side: which one outlines the chosen object."""
    meta = clips.info(clip)
    n, box = int(body.get("frame", -1)), _box(body.get("box"))
    phrases = list(dict.fromkeys(str(p).strip().lower()[:60] for p in body.get("phrases") or [] if str(p).strip()))[:8]
    if meta is None or box is None or not 0 <= n < meta["frames"] or not phrases:
        return JSONResponse({"error": "Pick an object and give at least one phrase."}, status_code=400)
    frames = clips.nearby(n, meta["frames"])
    sam = app.state.clip_sam
    try:
        answers = await asyncio.gather(*(sam.segment(clips.frame_path(clip, i).read_bytes(), phrases) for i in frames))
    except Exception as e:
        return JSONResponse({"error": str(e) or "The camera model did not answer."}, status_code=502)
    return {"frame": n, "frames": frames, "ranked": clips.rank_phrases(n, box, dict(zip(frames, answers)), phrases)}


@app.get("/api/profiles")
async def profiles():
    return {"profiles": app.state.services.faces.all()}


@app.get("/api/face/{profile_id}")
async def face(profile_id: str):
    picture = app.state.services.faces.thumb(profile_id)
    if not picture:
        return Response(status_code=404)
    return Response(picture, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@app.get("/api/evidence/{session}/{person}/{n}")
async def evidence(session: str, person: int, n: int):
    """One of the pictures the last check before charging was shown for this shopper."""
    held = app.state.sessions.get(session)
    shots = held.shown.get(person, []) if held else []
    if not 0 <= n < len(shots):
        return Response(status_code=404)
    return Response(shots[n], media_type="image/jpeg", headers={"Cache-Control": "private, max-age=3600"})


@app.delete("/api/profiles")
async def forget(request: Request):
    """Delete every profile: faces (here and in the vector index), payment links and visit history. PayPal's sandbox keeps its vault."""
    return {"forgotten": app.state.services.faces.forget_all()}


@app.websocket("/ws")
async def ws(socket: WebSocket):
    # The floor's socket can charge cards: staff only, and only from a page of this server. A browser sends the
    # cookie with a socket to any site that opens one, so the Origin is checked too.
    origin = socket.headers.get("origin", "")
    if origin and origin.split("://", 1)[-1].lower() != socket.headers.get("host", "").lower():
        await socket.close(code=1008)
        return
    if not staff_ok(socket):
        await socket.close(code=1008)
        return
    await socket.accept()
    services: Services = app.state.services
    outbox: asyncio.Queue = asyncio.Queue()
    session: Session | None = None
    wake = asyncio.Event()
    pending: dict | None = None
    ending = False

    async def pump():
        """Engine states, in frame order, into the outbox. After "end", close the visit once the frames are done."""
        nonlocal ending
        while True:
            await wake.wait()
            current = session
            if current is None or not current.in_flight:
                if ending and current is not None:
                    ending = False
                    await current.finish()
                wake.clear()
                continue
            try:
                state = await current.next_state()
            except Exception as e:  # a bug in the engine must not silently stop the visit
                log.exception("frame failed")
                state = {"error": f"{type(e).__name__}: {e}"}
            if current is session:  # else a restart happened while this frame was in flight
                outbox.put_nowait({"type": "lost" if "error" in state else "state", "session": current.id, **state})

    async def relay():
        while True:
            await socket.send_json(await outbox.get())

    tasks = [asyncio.ensure_future(pump()), asyncio.ensure_future(relay())]
    try:
        while True:
            message = await socket.receive()
            if message["type"] == "websocket.disconnect":
                break
            if message.get("text") is not None:
                data = json.loads(message["text"])
                kind = data.get("type")
                if kind == "start":
                    if session:
                        session.cancel()
                        keep_pictures(session)
                    path = shop_file(str(data.get("shop", ""))) or shop_file(DEFAULT_SHOP)
                    catalog = Catalog.from_dict(data["catalog"]) if data.get("catalog") else Catalog.load(path)
                    catalog.shop = path.stem
                    # "dry": everything runs except the charge and the email, for trying footage out
                    used = dataclasses.replace(services, payments=None, mailer=None) if data.get("dry") else services
                    session = Session(used, catalog, outbox, dry=bool(data.get("dry")), live=bool(data.get("live")))
                    app.state.sessions[session.id] = session
                    for old in list(app.state.sessions)[:-6]:
                        del app.state.sessions[old]
                    pending, ending = None, False
                    outbox.put_nowait({"type": "ready", "session": session.id, "services": services.status()})
                elif kind == "frame":
                    pending = data
                elif kind == "end" and session:
                    ending = True
                    wake.set()
            elif message.get("bytes") is not None and session and pending:
                header, pending = pending, None
                jpeg = message["bytes"]
                if len(jpeg) > MAX_FRAME_BYTES or session.in_flight >= MAX_IN_FLIGHT:
                    why = "frame too large" if len(jpeg) > MAX_FRAME_BYTES else "too many frames in flight"
                    outbox.put_nowait({"type": "lost", "session": session.id, "seq": header.get("seq"), "error": why})
                    continue
                session.submit(int(header.get("seq", 0)), float(header.get("t", 0.0)), jpeg)
                wake.set()
    except WebSocketDisconnect:
        pass
    finally:
        for task in tasks:
            task.cancel()
        if session:
            session.cancel()
            keep_pictures(session)


def keep_pictures(session: Session) -> None:
    """A visit that is over keeps only what the page can still ask for: the pictures each bag was checked against.
    Its frames, tracks and models' answers go, so a few old visits do not hold on to a few hundred MB."""
    if session.id in app.state.sessions:
        app.state.sessions[session.id] = types.SimpleNamespace(shown=session.shown)


app.mount("/samples", StaticFiles(directory=SAMPLES), name="samples")
class Fresh(StaticFiles):
    """The pages' scripts and styles, always checked against the file: a phone that opened the page an hour ago must
    not go on running the script it kept from then."""

    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-cache"
        return response


app.mount("/static", Fresh(directory=WEB), name="static")

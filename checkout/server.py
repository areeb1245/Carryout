"""
The live app: two screens and the calls behind them.

    uv run uvicorn checkout.server:app --port 8765

The gate (/gate, or gate.localhost:8765)   where a shopper is recognised or registers, and is let in
    POST /api/gate/look        one camera frame (a JPEG) -> who this is: no_face | closer | known | new
    POST /api/gate/register    a new face plus name, email and a payment method -> admitted, or a PayPal address to approve at
    GET  /gate/return          where PayPal sends the shopper back after approving
    GET  /api/gate/progress/x  has that approval come through yet?
    POST /api/gate/admit       a recognised face with a method on file -> an open visit

The shop floor (/)   frames from the browser over a WebSocket, checkout states back
    Browser -> server   {"type": "start", "shop": "bookshop", "catalog": {...}, "dry": bool}   new visit; `catalog` overrides the shop file
                        {"type": "frame", "seq": n, "t": seconds}   then one binary message: the JPEG
                        {"type": "end"}                             the footage is over: everyone walks out and pays
    Server -> browser   {"type": "ready", session}
                        {"type": "state", ...}       one per frame, in frame order: people, hands, items, events, carts, meta
                        {"type": "update", ...}      between frames: a face, a second look, the last check or a payment came through
                        {"type": "finished", ...}    after "end", once every payment has been attempted
                        {"type": "lost", seq, error} the model call for that frame failed

Every key (SAM, the verifier, PayPal, the database) stays in this process. The server listens on localhost only.
"""
from __future__ import annotations

import asyncio
import dataclasses
import html
import json
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from .catalog import Catalog
from .gate import Gate, decode
from .identity import Profiles
from .mailer import Mailer
from .payments import TEST_CARDS, Payments
from .pipeline import Services, Session
from .reader import Reader
from .sam_client import MAX_FPS, MAX_PROMPTS, ROOT, SamClient
from .vectors import FaceIndex
from .verify import Verifier

log = logging.getLogger("checkout")
WEB = ROOT / "web"
SAMPLES = ROOT / "samples"
SHOPS = ROOT / "shops"
DEFAULT_SHOP = "bookshop"
MAX_FRAME_BYTES = 2 * 1024 * 1024
MAX_IN_FLIGHT = 6


@asynccontextmanager
async def lifespan(app: FastAPI):
    index = FaceIndex()
    payments = Payments()
    services = Services(SamClient(), Verifier(), Profiles(provider=payments.name, index=index), payments, Mailer(), Reader())
    app.state.services = services
    app.state.sessions = {}  # the last few visits, so the page can fetch the pictures a bag was checked against
    app.state.gate = Gate(services.faces, payments, Catalog.load(shop_file(DEFAULT_SHOP)).name)
    yield
    await services.sam.close()
    await services.verifier.close()
    await services.mailer.close()
    await payments.close()
    index.close()


app = FastAPI(title="SAM 3.1 checkout", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)


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
    # PayPal sends the shopper back to this server; "localhost" because it will not take a made-up host name
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
async def forget():
    """Delete every profile: faces (here and in the vector index), payment links and visit history. PayPal's sandbox keeps its vault."""
    return {"forgotten": app.state.services.faces.forget_all()}


@app.websocket("/ws")
async def ws(socket: WebSocket):
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
                    catalog = Catalog.from_dict(data["catalog"]) if data.get("catalog") else Catalog.load(shop_file(str(data.get("shop", ""))) or shop_file(DEFAULT_SHOP))
                    # "dry": everything runs except the charge and the email, for trying footage out
                    used = dataclasses.replace(services, payments=None, mailer=None) if data.get("dry") else services
                    session = Session(used, catalog, outbox, dry=bool(data.get("dry")))
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


app.mount("/samples", StaticFiles(directory=SAMPLES), name="samples")
app.mount("/static", StaticFiles(directory=WEB), name="static")

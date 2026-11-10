"""
The live app: serves the page, takes frames from the browser over a WebSocket, answers with checkout states.

    uv run uvicorn checkout.server:app --port 8765

Browser -> server   {"type": "start", "shop": "bookshop", "catalog": {...}, "auto_checkin": bool}   new visit; `catalog` overrides the shop file
                    {"type": "frame", "seq": n, "t": seconds}   then one binary message: the JPEG
                    {"type": "checkin", "person": id, "name": "...", "email": "...", "card": "pm_card_visa"}
                    {"type": "end"}                             the footage is over: everyone walks out and pays
Server -> browser   {"type": "ready", session}
                    {"type": "state", ...}       one per frame, in frame order: people, hands, items, events, carts, meta
                    {"type": "update", ...}      between frames: a check-in, a second look or a payment came through
                    {"type": "finished", ...}    after "end", once every payment has been attempted
                    {"type": "lost", seq, error} the model call for that frame failed

Every key (SAM, the verifier, Stripe) stays in this process. The server listens on localhost only.
"""
from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from .catalog import Catalog
from .identity import Profiles
from .mailer import Mailer
from .payments import TEST_CARDS, Payments
from .pipeline import Services, Session
from .reader import Reader
from .sam_client import MAX_FPS, MAX_PROMPTS, ROOT, SamClient
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
    services = Services(SamClient(), Verifier(), Profiles(), Payments(), Mailer(), Reader())
    app.state.services = services
    yield
    await services.sam.close()
    await services.verifier.close()
    await services.mailer.close()


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
def index():
    return FileResponse(WEB / "index.html", headers={"Cache-Control": "no-store"})


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


@app.delete("/api/profiles")
async def forget():
    """Delete every profile kept on this machine: faces, card links and visit history. Stripe's sandbox keeps its customers."""
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
                    session = Session(used, catalog, outbox, auto_checkin=bool(data.get("auto_checkin", False)))
                    pending, ending = None, False
                    outbox.put_nowait({"type": "ready", "session": session.id, "services": services.status()})
                elif kind == "frame":
                    pending = data
                elif kind == "checkin" and session:
                    problem = session.check_in(int(data.get("person", 0)), str(data.get("name", "")), str(data.get("card", "")), str(data.get("email", "")))
                    if problem:
                        outbox.put_nowait({"type": "refused", "session": session.id, "person": data.get("person"), "message": problem})
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

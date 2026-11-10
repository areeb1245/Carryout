"""
Run the checkout on a video file and write an annotated copy, the event log and the receipts.

    uv run python -m checkout.cli samples/49350-720.mp4 --fps 3 --catalog shops/supermarket.json

`--fps` is how many frames a second are analysed. The live app manages 2 to 3 on the shared A10G.
SAM answers are cached under .cache/sam, so re-running a clip while tuning costs no GPU time.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

import cv2
import numpy as np

from .catalog import Catalog
from .identity import Profiles
from .payments import Payments
from .pipeline import Services, Session
from .reader import Reader
from .sam_client import ROOT, SamClient
from .verify import Verifier

PALETTE = [(245, 135, 66), (80, 200, 120), (60, 170, 250), (200, 90, 230), (70, 220, 230), (120, 120, 250)]  # BGR


def colour(pid: int | None) -> tuple:
    return (170, 170, 170) if pid is None else PALETTE[(pid - 1) % len(PALETTE)]


def draw(frame: np.ndarray, state: dict, catalog: Catalog, log: list[str]) -> np.ndarray:
    h, w = frame.shape[:2]
    out = frame.copy()

    def pts(poly):
        return (np.array(poly) * [w, h]).astype(np.int32)

    def shade(polys, c, alpha, thick=2):
        nonlocal out
        layer = out.copy()
        for poly in polys:
            cv2.fillPoly(layer, [pts(poly)], c)
        out = cv2.addWeighted(layer, alpha, out, 1 - alpha, 0)
        for poly in polys:
            cv2.polylines(out, [pts(poly)], True, c, thick, cv2.LINE_AA)

    def tag(text, x, y, c):
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        y = max(th + 6, y)
        cv2.rectangle(out, (x, y - th - 6), (x + tw + 8, y), c, -1)
        cv2.putText(out, text, (x + 4, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1, cv2.LINE_AA)

    for poly in state.get("zone_all", []):
        cv2.polylines(out, [pts(poly)], True, (200, 200, 200), 1, cv2.LINE_AA)
    carts = {c["person"]: c for c in state["carts"]}
    for p in state["people"]:
        shade(p["polygons"], colour(p["id"]), 0.22)
        cart = carts.get(p["id"], {})
        tag(f"Shopper {p['id']}  {cart.get('count', 0)} items  {catalog.currency} {cart.get('total', 0):.2f}", int(p["box"][0] * w), int(p["box"][1] * h), colour(p["id"]))
    for hand in state["hands"]:
        for poly in hand["polygons"]:
            cv2.polylines(out, [pts(poly)], True, colour(hand["person"]), 1, cv2.LINE_AA)
    for item in state["items"]:
        c = colour(item["holder"]) if item["state"] == "held" else (235, 235, 235)
        shade(item["polygons"], c, 0.45 if item["state"] == "held" else 0.12, 2 if item["state"] == "held" else 1)
        if item["state"] != "rest":
            who = f" > Shopper {item['holder']}" if item["holder"] else " ?"
            tag(item["name"] + who, int(item["box"][0] * w), int(item["box"][1] * h) - 2, c)
    shown = log[-4:]
    if shown:
        strip = out[h - 12 - 20 * len(shown) :, :]
        strip[:] = (strip * 0.45).astype(np.uint8)
    for i, line in enumerate(shown):
        cv2.putText(out, line, (12, h - 14 - 20 * (len(shown) - 1 - i)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def describe(event: dict, state: dict) -> str:
    lines = {line["id"]: line for cart in state["carts"] for line in cart["lines"]}
    line = lines.get(event.get("entry"), {})
    name = line.get("name", "item")
    if line.get("unread"):
        name = f"a {name.lower()} (not yet told which)"
    elif event["type"] == "pick" and line.get("product") and not line.get("billable"):
        return f"{event['t']:5.1f}s  Shopper {event.get('person')} is holding their own {name.lower()} (not billed)"
    who = f"Shopper {event.get('person')}"
    t = f"{event['t']:5.1f}s"
    match event["type"]:
        case "enter":
            return f"{t}  {who} walked in"
        case "reenter":
            return f"{t}  {who} came back"
        case "exit":
            return f"{t}  {who} left, total {event['total']:.2f}"
        case "pick":
            how = {"carried_in": "arrived holding", "inferred": "took (inferred from the shelf)", "seen": "is carrying"}.get(event["origin"], "took")
            return f"{t}  {who} {how} {name}  ({event['confidence']:.0%})"
        case "return":
            return f"{t}  {who} returned {name}: {event['how']}  ({event['confidence']:.0%})"
        case "void":
            return f"{t}  (that pick was the same item as an earlier line: folded into it)"
        case "handover":
            return f"{t}  Shopper {event['from_person']} handed {name} to {who}"
        case "checkin":
            where = "checked in on the shop floor, not at the gate" if event.get("how") == "floor" else "let in at the gate"
            return f"{t}  {who} is {event['name']}, {where} ({event.get('method') or 'no payment method'})"
        case "unregistered":
            return f"{t}  {who} is not identified{' (resembles ' + event['name'] + ', match too weak)' if event.get('name') else ''}: nothing will be charged"
        case "audit":
            return f"{t}  Check before charging, {name}: {event['verdict']} ({event['why']})"
        case "invoice":
            return f"{t}  Invoice {event.get('number') or ''} {event['status']}: {event.get('url') or event.get('message') or ''}"
        case "verify":
            return f"{t}  Second model on {name}: {event['verdict']} ({event['why']})"
        case "email":
            return f"{t}  Receipt email to {event['to']}: {event['status']} {event.get('message') or ''}"
        case "payment":
            return f"{t}  {who} payment {event['status']}: {event['amount']:.2f} {event['currency']} for {event['items']} items {event.get('payment') or ''} {event.get('message') or ''}"
    return f"{t}  {event['type']}"


async def run(args) -> dict:
    catalog = Catalog.load(Path(args.catalog))
    sam = SamClient(cache_dir=None if args.no_cache else ROOT / ".cache" / "sam")
    services = Services(sam, Verifier() if args.verify else None, Profiles() if args.pay else None, Payments() if args.pay else None, None, Reader() if args.verify else None)
    session = Session(services, catalog)
    cap = cv2.VideoCapture(args.video)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25
    step = max(1, round(src_fps / args.fps))
    width = args.width or catalog.frame_width
    frames, seq, n = {}, 0, 0
    limit = asyncio.Semaphore(args.parallel)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if n % step == 0:
            k = min(1.0, width / max(frame.shape[:2]))  # the width caps the longer side
            small = cv2.resize(frame, (round(frame.shape[1] * k), round(frame.shape[0] * k)), interpolation=cv2.INTER_AREA)
            frames[seq] = (n / src_fps, small)
            seq += 1
        n += 1
    cap.release()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.video).stem
    writer = None
    log, events, last = [], [], None
    zone_all: list = []

    started = time.perf_counter()

    async def feed():
        for s, (t, small) in frames.items():
            if args.verify or args.pay:
                # the side jobs (second looks, reading) take real seconds: feed the footage no faster than it was shot
                await asyncio.sleep(max(0.0, t - (time.perf_counter() - started)))
            await limit.acquire()
            session.submit(s, t, cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes())

    feeder = asyncio.ensure_future(feed())
    for s, (t, small) in frames.items():
        state = await session.next_state()
        limit.release()
        if "error" in state:
            print(f"   frame {s} lost: {state['error']}")
            continue
        zone_all = state.get("zone", zone_all)
        state["zone_all"] = zone_all
        for e in state["events"]:
            log.append(describe(e, state))
            print(log[-1])
        events.extend(state["events"])
        if args.trace:
            held = [f"#{i['id']} {i['name']}>{i['holder']}" for i in state["items"] if i["state"] != "rest"]
            print(f"   t={t:5.2f} people={[p['id'] for p in state['people']]} hands={[x['person'] for x in state['hands']]} items={held} rest={sum(1 for i in state['items'] if i['state']=='rest')} ms={state['ms']}")
        picture = draw(small, state, catalog, log)
        if writer is None:
            writer = cv2.VideoWriter(str(out_dir / f"{stem}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, picture.shape[1::-1])
        writer.write(picture)
        if args.stills and s % args.stills == 0:
            cv2.imwrite(str(out_dir / f"{stem}-{s:03d}.jpg"), picture)
        last = state
    await feeder
    writer.release()
    if args.pay or args.verify:
        await session.read_shelf_now()
        # the footage is over: everyone walks out, and anything still being checked or charged is seen through
        await session.finish()
    last = {"carts": session.engine.carts()}
    printed = {e["id"] for e in events}
    for e in session.engine.events:  # what the side jobs added: check-ins, second looks, payments
        if e["id"] not in printed:
            events.append(e)
            print(describe(e, last))
    await sam.close()
    if services.verifier:
        await services.verifier.close()

    stock = [i for i in session.engine.items.values() if i.state == "rest" and catalog.family_of(i.seen_as)]
    if stock:
        print(f"\nShelf: {sum(1 for i in stock if i.settled)} of {len(stock)} items identified")
        for i in sorted(stock, key=lambda i: i.c[0]):
            print(f"  x={i.c[0] / session.engine.size[0]:.2f}  {(catalog.by_id[i.settled].name if i.settled else '(not read)'):<42} {i.read_how}")

    receipts = last["carts"] if last else []
    (out_dir / f"{stem}.json").write_text(json.dumps({"events": events, "carts": receipts}, indent=2))
    print("\nCarts")
    for cart in receipts:
        print(f"  Shopper {cart['person']} ({cart['state']}): {cart['count']} items, {catalog.currency} {cart['total']:.2f}")
        for line in cart["lines"]:
            price = "" if line["price"] is None else f"{line['price']:.2f}"
            print(f"     {line['state']:<9} {line['name']:<22} {price:>7}  {line['origin']:<10} {line['confidence']:.0%}  {line['verified'] or ''} {'paid' if line['paid'] else ''}")
    return {"events": events, "carts": receipts}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--catalog", default=str(ROOT / "shops" / "supermarket.json"), help="a shop's catalogue file, e.g. shops/bookshop.json")
    ap.add_argument("--fps", type=float, default=3.0, help="frames analysed per second of video")
    ap.add_argument("--width", type=int, default=0, help="longer side of the frames sent for analysis; default is the catalogue's")
    ap.add_argument("--out", default=str(ROOT / "out"))
    ap.add_argument("--parallel", type=int, default=3)
    ap.add_argument("--stills", type=int, default=0, help="also save every Nth annotated frame as a JPEG")
    ap.add_argument("--trace", action="store_true", help="print what the engine saw on every frame")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--verify", action="store_true", help="have the vision language model look at every pick")
    ap.add_argument("--pay", action="store_true", help="charge the shoppers who registered at the gate (their saved PayPal method) at the end")
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()

"""
Plays a clip to the running server the way the page does, in real time, and prints what comes back.

    uv run tools/live_client.py samples/49350-720.mp4 [--name Areeb --email you@example.com --card pm_card_visa] [--auto]

It sends a frame whenever fewer than --in-flight are in flight, never more than --fps a second, with the clip's own clock.
When the server reports a new face it checks that shopper in with the given name, email and test card (unless --auto).
"""
import argparse, asyncio, json, time, urllib.request
import cv2, websockets


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clip"); ap.add_argument("--url", default="ws://127.0.0.1:8765/ws")
    ap.add_argument("--name", default="Test shopper"); ap.add_argument("--email", default=""); ap.add_argument("--card", default="pm_card_visa")
    ap.add_argument("--auto", action="store_true", help="hands-free: the server checks new faces in by itself"); ap.add_argument("--record", help="write every message from the server to this JSON file")
    ap.add_argument("--dry", action="store_true", help="no charge and no email");
    ap.add_argument("--shop", default="bookshop"); ap.add_argument("--fps", type=float, default=5.0); ap.add_argument("--in-flight", type=int, default=4); ap.add_argument("--width", type=int, default=960)
    a = ap.parse_args()
    # like the page: wake the GPU first, so the clip is not half over before the first answer comes back
    health = a.url.replace("ws://", "http://").replace("/ws", "/api/health"); t_wake = time.perf_counter()
    print("model:", json.load(urllib.request.urlopen(health, timeout=180)), f"({time.perf_counter() - t_wake:.1f}s to wake)")
    cap = cv2.VideoCapture(a.clip); fps = cap.get(cv2.CAP_PROP_FPS) or 25; n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); length = n / fps
    async with websockets.connect(a.url, max_size=None) as ws:
        await ws.send(json.dumps({"type": "start", "shop": a.shop, "auto_checkin": a.auto, "dry": a.dry}))
        ready = json.loads(await ws.recv()); session = ready["session"]; print("services:", ready["services"])
        recorded = []; shelf = {}; printed = set()
        t0 = time.perf_counter(); in_flight = 0; last = -1.0; seq = 0; lat = []; asked = set(); sent_at = {}; names = {}; done = asyncio.Event()

        async def listen():
            nonlocal in_flight
            async for raw in ws:
                m = json.loads(raw)
                if m.get("session") != session: continue
                recorded.append(m)
                if m["type"] in ("state", "lost"): in_flight -= 1
                if m["type"] == "lost": print("  frame lost:", m.get("error")); continue
                if m["type"] == "refused": print("  check-in refused:", m["message"]); continue
                if m["type"] == "state": lat.append(time.perf_counter() - t0 - m["t"])
                lines = {l["id"]: l for c in m.get("carts", []) for l in c["lines"]}
                for pid, who in (m.get("meta", {}).get("people") or {}).items():
                    if who.get("profile"): names[int(pid)] = who["profile"]["name"]
                    if who["status"] == "new_face" and pid not in asked and not a.auto:
                        asked.add(pid)
                        print(f"  [{time.perf_counter()-t0:5.1f}s] new face on shopper {pid}: checking in as {a.name}")
                        await ws.send(json.dumps({"type": "checkin", "person": int(pid), "name": a.name, "email": a.email, "card": a.card}))
                for e in m.get("events", []):
                    if e["id"] in printed: continue
                    printed.add(e["id"])
                    who = names.get(e.get("person"), f"Shopper {e.get('person')}"); item = lines.get(e.get("entry"), {}).get("name", "")
                    extra = {k: v for k, v in e.items() if k not in ("id", "t", "type", "person", "entry", "item") and v not in (None, "")}
                    if "payment" in extra: extra["payment"] = extra["payment"][:6] + "…"
                    print(f"  [{time.perf_counter()-t0:5.1f}s wall | {e['t']:5.1f}s clip] {e['type']:<8} {who:<14} {item:<20} {extra}")
                if m.get("meta", {}).get("shelf") and m["meta"]["shelf"] != shelf.get("last"):
                    shelf["last"] = m["meta"]["shelf"]; print(f"  [{time.perf_counter()-t0:5.1f}s] shelf: {shelf['last']['read']} of {shelf['last']['items']} identified")
                if m["type"] == "finished":
                    print("\nfinal carts:")
                    for c in m["carts"]:
                        pay = m["meta"]["payments"].get(str(c["person"]), {})
                        print(f"  {names.get(c['person'], 'Shopper %d' % c['person'])}: {c['count']} item(s), total {c['total']:.2f}, payment {pay.get('status')} {pay.get('email', {}).get('status', '')}")
                        for l in c["lines"]: print(f"     {l['state']:<9} {l['name']:<20} charge={l['charge']} verified={l['verified']} paid={bool(l['paid'])} conf={l['confidence']}")
                    done.set(); return

        listener = asyncio.ensure_future(listen())
        while True:
            t = time.perf_counter() - t0
            if t >= length: break
            if in_flight < a.in_flight and t - last >= 1 / a.fps:
                cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000); ok, frame = cap.read()
                if not ok: break
                k = min(1.0, a.width / max(frame.shape[:2]))  # --width caps the longer side, as the page does
                small = cv2.resize(frame, (round(frame.shape[1] * k), round(frame.shape[0] * k)), interpolation=cv2.INTER_AREA)
                t = time.perf_counter() - t0; last = t; in_flight += 1
                await ws.send(json.dumps({"type": "frame", "seq": seq, "t": round(t, 3)})); await ws.send(cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()); seq += 1
            await asyncio.sleep(0.01)
        await ws.send(json.dumps({"type": "end"}))
        await asyncio.wait_for(done.wait(), 60); listener.cancel()
        if a.record: open(a.record, "w").write(json.dumps(recorded))
        lat.sort(); print(f"\n{seq} frames in {length:.1f}s = {seq/length:.1f} a second; answer delay median {lat[len(lat)//2]:.2f}s, worst {lat[-1]:.2f}s")

asyncio.run(main())

"""Probe: run prompts on single frames of a clip and draw what SAM 3.1 found.  uv run tools/probe.py clip.mp4 t1,t2 prompt1,prompt2"""
import asyncio, sys, time
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from checkout.sam_client import SamClient, ROOT

COLORS = [(66,135,245),(245,166,35),(80,200,120),(230,80,80),(170,90,230),(60,200,220),(240,220,70),(250,120,190)]

async def main():
    clip, times, prompts = sys.argv[1], [float(t) for t in sys.argv[2].split(",")], sys.argv[3].split(",")
    thr = float(sys.argv[4]) if len(sys.argv) > 4 else 0.3
    client = SamClient(cache_dir=ROOT / ".cache" / "sam")
    cap = cv2.VideoCapture(clip)
    for t in times:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        if not ok: continue
        frame = cv2.resize(frame, (960, 540))
        jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()
        t0 = time.time(); r = await client.segment(jpeg, prompts, thr); dt = time.time() - t0
        h, w = frame.shape[:2]
        print(f"t={t}: {len(jpeg)//1024} KB, server {r['ms']} ms, round trip {dt*1000:.0f} ms")
        for d in sorted(r["detections"], key=lambda d: -d["score"]):
            c = COLORS[prompts.index(d["label"]) % len(COLORS)]
            b = d["box"]
            print(f"   {d['label']:<22} {d['score']:.2f} box=({b[0]:.2f},{b[1]:.2f},{b[2]:.2f},{b[3]:.2f})")
            for poly in d.get("polygons", []):
                pts = (np.array(poly) * [w, h]).astype(np.int32)
                ov = frame.copy(); cv2.fillPoly(ov, [pts], c); frame = cv2.addWeighted(ov, 0.35, frame, 0.65, 0)
                cv2.polylines(frame, [pts], True, c, 2)
            cv2.putText(frame, f"{d['label']} {d['score']:.2f}", (int(b[0]*w), max(12, int(b[1]*h)-4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1, cv2.LINE_AA)
        out = ROOT / "scratch" / f"probe-{Path(clip).stem}-{t}.jpg"
        cv2.imwrite(str(out), frame); print("   ->", out)
    await client.close()
asyncio.run(main())

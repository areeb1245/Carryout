"""
SAM 3.1 for the walk-out checkout, on Modal: every phrase of a frame in one forward pass.

The shared `deck-twin-sam31` app runs the image through the vision backbone once and then the detector once per
phrase, one after the other, so eight phrases cost eight detector passes. Here the phrases are a batch: the backbone
output is reused for all of them and the detector runs once over the whole batch. The phrase embeddings are cached,
because a shop asks for the same phrases on every frame.

    BENCH_GPU=L40S modal run modal_app/sam_checkout.py   time one-by-one against batched on a GPU type
    modal deploy modal_app/sam_checkout.py               publish https://<workspace>--sam3-supermarket.modal.run

The weights are read from the volume the shared app already filled (`deck-twin-sam31`). The key comes from the Modal
secret `sam3-supermarket` (CHECKOUT_SAM_KEY).

    GET  /health        no key
    POST /v1/segment    {image_base64, prompts[1..16], threshold} -> {image, detections[{label, box, score, polygons}], ms}

Same answer shape as the shared app, so the client only needs another URL.
"""
import os
import threading

import modal

APP_NAME = "sam3-supermarket"
VOLUME = "/weights"
WEIGHTS = f"{VOLUME}/sam3.1/sam3.1_multiplex.pt"
SAM3_COMMIT = "f1647e8d0c"  # the commit the shared app was tested on
# Measured on one 960 px frame with eight phrases: A10G 575 ms one by one, 367 ms batched; L40S 290 ms and 139 ms.
GPU = os.environ.get("CHECKOUT_GPU", "L40S")
MAX_PROMPTS = 16
MAX_BODY = 8 * 1024 * 1024

app = modal.App(APP_NAME)
volume = modal.Volume.from_name("deck-twin-sam31")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "libgl1", "libglib2.0-0")
    .pip_install("torch==2.10.0", "torchvision", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install(f"git+https://github.com/facebookresearch/sam3.git@{SAM3_COMMIT}", "einops", "psutil", "pycocotools", "opencv-python-headless", "pillow", "httpx", "fastapi[standard]")
)


class Segmenter:
    """The model and the two ways of running it. Plain Python, so the benchmark and the web app share it."""

    def __init__(self):
        import torch
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model

        if not os.path.exists(WEIGHTS):
            raise RuntimeError(f"{WEIGHTS} is missing from the deck-twin-sam31 volume")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        self.model = build_sam3_image_model(checkpoint_path=WEIGHTS, load_from_HF=False)
        self.processor = Sam3Processor(self.model, confidence_threshold=0.3)
        self.text_cache: dict[tuple, dict] = {}
        self.lock = threading.Lock()  # one frame on the GPU at a time

    def _text(self, prompts: tuple) -> dict:
        if prompts not in self.text_cache:
            if len(self.text_cache) > 64:
                self.text_cache.clear()
            self.text_cache[prompts] = self.model.backbone.forward_text(list(prompts), device="cuda")
        return self.text_cache[prompts]

    def batched(self, jpeg: bytes, prompts: list[str], threshold: float) -> dict:
        import io

        import torch
        from PIL import Image
        from sam3.model.sam3_image_processor import FindStage
        from torch.nn.functional import interpolate

        picture = Image.open(io.BytesIO(jpeg)).convert("RGB")
        w, h = picture.size
        n = len(prompts)
        with self.lock, torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            state = self.processor.set_image(picture)
            backbone_out = {**state["backbone_out"], **self._text(tuple(prompts))}
            find = FindStage(
                img_ids=torch.zeros(n, device="cuda", dtype=torch.long),  # every phrase looks at image 0
                text_ids=torch.arange(n, device="cuda", dtype=torch.long),
                input_boxes=None,
                input_boxes_mask=None,
                input_boxes_label=None,
                input_points=None,
                input_points_mask=None,
            )
            out = self.model.forward_grounding(backbone_out=backbone_out, find_input=find, geometric_prompt=self.model._get_dummy_prompt(num_prompts=n), find_target=None)
            probs = (out["pred_logits"].sigmoid() * out["presence_logit_dec"].sigmoid().unsqueeze(1)).squeeze(-1)  # phrases x queries
            keep = probs > threshold
            which = keep.nonzero()[:, 0].tolist()
            scores = probs[keep].float().tolist()
            boxes = out["pred_boxes"][keep].float()  # cx, cy, w, h in 0..1
            masks = out["pred_masks"][keep]
            if len(which):
                # contours are traced at up to 960 px on the long side: enough for outlines, light on the CPU
                scale = min(1.0, 960 / max(w, h))
                size = (max(1, round(h * scale)), max(1, round(w * scale)))
                masks = (interpolate(masks.unsqueeze(1).float(), size, mode="bilinear", align_corners=False) > 0).squeeze(1).to(torch.uint8).cpu().numpy()
            boxes = boxes.cpu().tolist()
        detections = []
        for i, k in enumerate(which):
            cx, cy, bw, bh = boxes[i]
            box = [round(min(1.0, max(0.0, v)), 4) for v in (cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2)]
            detections.append({"label": prompts[k], "box": box, "score": round(scores[i], 4), "polygons": polygons(masks[i])})
        return {"model": "sam-3.1", "image": {"w": w, "h": h}, "detections": detections}

    def one_by_one(self, jpeg: bytes, prompts: list[str], threshold: float) -> dict:
        """What the shared app does. Kept for the benchmark."""
        import io

        import torch
        from PIL import Image

        picture = Image.open(io.BytesIO(jpeg)).convert("RGB")
        w, h = picture.size
        detections = []
        with self.lock, torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            self.processor.confidence_threshold = threshold
            state = self.processor.set_image(picture)
            for prompt in prompts:
                self.processor.reset_all_prompts(state)
                out = self.processor.set_text_prompt(state=state, prompt=prompt)
                masks = out["masks"].cpu().numpy()
                for i, (box, score) in enumerate(zip(out["boxes"].float().cpu().numpy(), out["scores"].float().cpu().numpy())):
                    x0, y0, x1, y1 = box.tolist()
                    detections.append({"label": prompt, "box": [round(x0 / w, 4), round(y0 / h, 4), round(x1 / w, 4), round(y1 / h, 4)], "score": round(float(score), 4), "polygons": polygons(masks[i].squeeze())})
        return {"model": "sam-3.1", "image": {"w": w, "h": h}, "detections": detections}


def polygons(mask) -> list:
    """Up to three outlines of a 0/1 mask, as normalised points."""
    import cv2
    import numpy as np

    h, w = mask.shape
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    found = []
    for c in sorted(contours, key=cv2.contourArea, reverse=True)[:3]:
        if cv2.contourArea(c) < 4:
            continue
        points = cv2.approxPolyDP(c, 0.003 * cv2.arcLength(c, True), True).reshape(-1, 2)
        if len(points) >= 3:
            found.append([[round(float(x) / w, 4), round(float(y) / h, 4)] for x, y in points])
    return found


@app.function(image=image, gpu=os.environ.get("BENCH_GPU", "A10G"), volumes={VOLUME: volume}, timeout=900)
def bench(jpeg: bytes, prompts: list[str], rounds: int = 8) -> dict:
    """Times both ways on one frame and checks they find the same things."""
    import inspect
    import time

    import torch

    seg = Segmenter()
    print("gpu:", torch.cuda.get_device_name(0), "| dummy prompt signature:", inspect.signature(seg.model._get_dummy_prompt))
    report = {"gpu": torch.cuda.get_device_name(0)}
    for name, run in (("one_by_one", seg.one_by_one), ("batched", seg.batched)):
        run(jpeg, prompts, 0.3)  # warm up
        times = []
        for _ in range(rounds):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            result = run(jpeg, prompts, 0.3)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
        counts: dict[str, int] = {}
        for d in result["detections"]:
            counts[d["label"]] = counts.get(d["label"], 0) + 1
        times.sort()
        report[name] = {"median_ms": round(times[len(times) // 2]), "best_ms": round(times[0]), "found": counts, "top": sorted(((d["label"], d["score"], d["box"]) for d in result["detections"]), key=lambda x: -x[1])[:6]}
        print(name, report[name]["median_ms"], "ms", counts)
    for n in (3, 5):
        seg.batched(jpeg, prompts[:n], 0.3)
        t0 = time.perf_counter()
        for _ in range(4):
            seg.batched(jpeg, prompts[:n], 0.3)
        report[f"batched_{n}_phrases_ms"] = round((time.perf_counter() - t0) / 4 * 1000)
    report["vram_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 1)
    return report


@app.local_entrypoint()
def main(frame: str = "scratch/bench-frame.jpg"):
    import json
    from pathlib import Path

    prompts = ["person", "hand", "item held in hand", "food package", "table tennis paddle", "bottle", "drink can", "mobile phone"]
    report = bench.remote(Path(frame).read_bytes(), prompts)
    print(json.dumps(report, indent=1))


@app.cls(
    image=image,
    gpu=GPU,
    volumes={VOLUME: volume},
    secrets=[modal.Secret.from_name("sam3-supermarket")],
    scaledown_window=300,  # the GPU is released five minutes after the last frame
    max_containers=1,  # one L40S keeps up with seven frames a second; a second container would only add cost
    timeout=120,
)
@modal.concurrent(max_inputs=6)
class Sam:
    @modal.enter()
    def load(self):
        self.seg = Segmenter()

    @modal.asgi_app(label=APP_NAME)
    def web(self):
        import base64
        import binascii
        import hmac
        import time

        from fastapi import FastAPI, Request
        from fastapi.responses import JSONResponse
        from pydantic import BaseModel, Field

        key = os.environ["CHECKOUT_SAM_KEY"]
        api = FastAPI(title="SAM 3.1 for the walk-out checkout", docs_url=None, redoc_url=None, openapi_url=None)

        class Body(BaseModel):
            image_base64: str
            prompts: list[str] = Field(..., min_length=1, max_length=MAX_PROMPTS)
            threshold: float = Field(0.3, ge=0, le=1)
            return_masks: bool = True

        @api.middleware("http")
        async def guard(request: Request, call_next):
            if request.url.path != "/health":
                given = request.headers.get("x-api-key") or ""
                if not hmac.compare_digest(given.encode(), key.encode()):
                    return JSONResponse({"error": "missing or invalid API key"}, status_code=401)
                length = request.headers.get("content-length", "")
                if request.method == "POST" and (not length.isdigit() or int(length) > MAX_BODY):
                    return JSONResponse({"error": "body missing a length, or too large"}, status_code=413)
            return await call_next(request)

        @api.get("/health")
        def health():
            return {"status": "ok", "model": "sam-3.1", "gpu": GPU, "batched": True}

        @api.post("/v1/segment")
        def segment(body: Body):
            data = body.image_base64.split(",", 1)[1] if body.image_base64.startswith("data:") else body.image_base64
            try:
                jpeg = base64.b64decode(data, validate=True)
            except (binascii.Error, ValueError):
                return JSONResponse({"error": "image_base64 is not valid base64"}, status_code=400)
            if jpeg[:3] != b"\xff\xd8\xff" and jpeg[:8] != b"\x89PNG\r\n\x1a\n":
                return JSONResponse({"error": "the image must be a JPEG or PNG"}, status_code=400)
            prompts = [p.strip()[:80] for p in body.prompts if p.strip()]
            if not prompts:
                return JSONResponse({"error": "no prompts"}, status_code=400)
            t0 = time.perf_counter()
            result = self.seg.batched(jpeg, prompts, body.threshold)
            return {**result, "ms": round((time.perf_counter() - t0) * 1000)}

        return api

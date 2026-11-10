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
    POST /v1/segment    {image_base64, prompts[0..16], threshold, exemplars?} -> {image, detections[{label, box, score, polygons}], ms}
    POST /v1/embed      {images_base64[1..16]} -> {vectors[[384 floats]], dim, ms}

Same answer shape as the shared app, so the client only needs another URL.

`exemplars` is SAM 3's visual prompt: boxes drawn round examples *in this same image*, with an optional phrase as a
hint, `[{prompt, boxes: [[x0, y0, x1, y1] in 0..1], labels: [true for "like this", false for "not this"]}]`. SAM
then outlines everything in the picture that looks like the positive boxes. A picture from elsewhere cannot be
given: SAM 3 reads the example from the image it is segmenting. Each exemplar is one more detector pass on the
same backbone output, about 20 ms.

`/v1/embed` turns pictures (a product cut out of footage, an item in a shopper's hand) into vectors with DINOv2
(ViT-S/14), L2-normalised, so the cosine of two of them says how alike they look.
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
MAX_EXEMPLARS = 4
MAX_EMBED = 16
EMBEDDER = "facebook/dinov2-small"
MAX_BODY = 8 * 1024 * 1024

app = modal.App(APP_NAME)
volume = modal.Volume.from_name("deck-twin-sam31")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "libgl1", "libglib2.0-0")
    .pip_install("torch==2.10.0", "torchvision", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install(f"git+https://github.com/facebookresearch/sam3.git@{SAM3_COMMIT}", "einops", "psutil", "pycocotools", "opencv-python-headless", "pillow", "httpx", "fastapi[standard]")
    .pip_install("transformers==4.57.1")
    .run_commands(f"python -c \"from transformers import AutoImageProcessor, AutoModel; AutoImageProcessor.from_pretrained('{EMBEDDER}'); AutoModel.from_pretrained('{EMBEDDER}')\"")
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
        from transformers import AutoImageProcessor, AutoModel

        self.embed_prep = AutoImageProcessor.from_pretrained(EMBEDDER)
        self.embedder = AutoModel.from_pretrained(EMBEDDER).to("cuda").eval()

    def embed(self, pictures: list[bytes]) -> list[list[float]]:
        """One vector per picture: DINOv2's class token and mean patch token side by side, L2-normalised."""
        import io

        import torch
        from PIL import Image

        images = [Image.open(io.BytesIO(b)).convert("RGB") for b in pictures]
        with self.lock, torch.inference_mode():
            batch = self.embed_prep(images=images, return_tensors="pt").to("cuda")
            hidden = self.embedder(**batch).last_hidden_state  # pictures x tokens x 384
            vec = hidden[:, 0] + hidden[:, 1:].mean(dim=1)
            vec = torch.nn.functional.normalize(vec.float(), dim=-1)
        return [[round(float(x), 5) for x in row] for row in vec.cpu().tolist()]

    def _text(self, prompts: tuple) -> dict:
        if prompts not in self.text_cache:
            if len(self.text_cache) > 64:
                self.text_cache.clear()
            self.text_cache[prompts] = self.model.backbone.forward_text(list(prompts), device="cuda")
        return self.text_cache[prompts]

    def batched(self, jpeg: bytes, prompts: list[str], threshold: float, exemplars: list[dict] | None = None) -> dict:
        import io

        import torch
        from PIL import Image
        from sam3.model.sam3_image_processor import FindStage

        picture = Image.open(io.BytesIO(jpeg)).convert("RGB")
        w, h = picture.size
        n = len(prompts)
        found: list[tuple[str, list, float, object, bool]] = []
        with self.lock, torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            state = self.processor.set_image(picture)
            if n:
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
                for k, box, score, mask in self._keep(out, threshold, w, h):
                    found.append((prompts[k], box, score, mask, False))
            for ex in exemplars or []:
                # SAM 3's own visual prompt, as Sam3Processor.add_geometric_prompt builds it: boxes in this image,
                # cx cy w h in 0..1, one sequence entry each, on top of the phrase (or "visual" with none)
                hint = (ex.get("prompt") or "").strip()[:80] or "visual"
                backbone_out = {**state["backbone_out"], **self._text((hint,))}
                geo = self.model._get_dummy_prompt()
                cxcywh = [[(b[0] + b[2]) / 2, (b[1] + b[3]) / 2, b[2] - b[0], b[3] - b[1]] for b in ex["boxes"]]
                labels = list(ex.get("labels") or [True] * len(cxcywh))
                geo.append_boxes(
                    torch.tensor(cxcywh, device="cuda", dtype=torch.float32).view(-1, 1, 4),
                    torch.tensor(labels, device="cuda", dtype=torch.bool).view(-1, 1),
                )
                out = self.model.forward_grounding(backbone_out=backbone_out, find_input=self.processor.find_stage, geometric_prompt=geo, find_target=None)
                label = ex.get("label") or (hint if hint != "visual" else "example")
                for _, box, score, mask in self._keep(out, threshold, w, h):
                    found.append((label, box, score, mask, True))
        detections = []
        for label, box, score, mask, by_example in found:
            d = {"label": label, "box": box, "score": score, "polygons": polygons(mask)}
            if by_example:
                d["exemplar"] = True
            detections.append(d)
        return {"model": "sam-3.1", "image": {"w": w, "h": h}, "detections": detections}

    @staticmethod
    def _keep(out: dict, threshold: float, w: int, h: int) -> list:
        """(query index, box x0 y0 x1 y1 in 0..1, score, 0/1 mask) for every detection over the threshold."""
        import torch
        from torch.nn.functional import interpolate

        probs = (out["pred_logits"].sigmoid() * out["presence_logit_dec"].sigmoid().unsqueeze(1)).squeeze(-1)  # queries x slots
        if probs.dim() == 1:
            probs = probs.unsqueeze(0)
        keep = probs > threshold
        which = keep.nonzero()[:, 0].tolist()
        if not which:
            return []
        scores = probs[keep].float().tolist()
        boxes = out["pred_boxes"][keep].float().cpu().tolist()  # cx, cy, w, h in 0..1
        masks = out["pred_masks"][keep]
        # contours are traced at up to 960 px on the long side: enough for outlines, light on the CPU
        scale = min(1.0, 960 / max(w, h))
        size = (max(1, round(h * scale)), max(1, round(w * scale)))
        masks = (interpolate(masks.unsqueeze(1).float(), size, mode="bilinear", align_corners=False) > 0).squeeze(1).to(torch.uint8).cpu().numpy()
        kept = []
        for i, k in enumerate(which):
            cx, cy, bw, bh = boxes[i]
            box = [round(min(1.0, max(0.0, v)), 4) for v in (cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2)]
            kept.append((k, box, round(scores[i], 4), masks[i]))
        return kept

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

        class Exemplar(BaseModel):
            prompt: str = ""
            label: str = ""
            boxes: list[list[float]] = Field(..., min_length=1, max_length=8)
            labels: list[bool] | None = None

        class Body(BaseModel):
            image_base64: str
            prompts: list[str] = Field(default_factory=list, max_length=MAX_PROMPTS)
            threshold: float = Field(0.3, ge=0, le=1)
            return_masks: bool = True
            exemplars: list[Exemplar] = Field(default_factory=list, max_length=MAX_EXEMPLARS)

        class Pictures(BaseModel):
            images_base64: list[str] = Field(..., min_length=1, max_length=MAX_EMBED)

        def picture(data: str) -> bytes | None:
            data = data.split(",", 1)[1] if data.startswith("data:") else data
            try:
                raw = base64.b64decode(data, validate=True)
            except (binascii.Error, ValueError):
                return None
            return raw if raw[:3] == b"\xff\xd8\xff" or raw[:8] == b"\x89PNG\r\n\x1a\n" else None

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
            return {"status": "ok", "model": "sam-3.1", "gpu": GPU, "batched": True, "exemplars": True, "embed": EMBEDDER}

        @api.post("/v1/segment")
        def segment(body: Body):
            jpeg = picture(body.image_base64)
            if jpeg is None:
                return JSONResponse({"error": "image_base64 must be a JPEG or PNG in base64"}, status_code=400)
            prompts = [p.strip()[:80] for p in body.prompts if p.strip()]
            exemplars = []
            for ex in body.exemplars:
                boxes = [[min(1.0, max(0.0, float(v))) for v in b] for b in ex.boxes if len(b) == 4]
                boxes = [b for b in boxes if b[2] - b[0] > 0.002 and b[3] - b[1] > 0.002]
                labels = ex.labels if ex.labels and len(ex.labels) == len(ex.boxes) else None
                if boxes and (labels is None or len(labels) == len(boxes)):
                    exemplars.append({"prompt": ex.prompt, "label": ex.label.strip()[:80], "boxes": boxes, "labels": labels})
            if not prompts and not exemplars:
                return JSONResponse({"error": "no prompts and no exemplars"}, status_code=400)
            t0 = time.perf_counter()
            result = self.seg.batched(jpeg, prompts, body.threshold, exemplars)
            return {**result, "ms": round((time.perf_counter() - t0) * 1000)}

        @api.post("/v1/embed")
        def embed(body: Pictures):
            pictures = [picture(b) for b in body.images_base64]
            if any(p is None for p in pictures):
                return JSONResponse({"error": "every picture must be a JPEG or PNG in base64"}, status_code=400)
            t0 = time.perf_counter()
            vectors = self.seg.embed(pictures)
            return {"vectors": vectors, "dim": len(vectors[0]) if vectors else 0, "model": EMBEDDER, "ms": round((time.perf_counter() - t0) * 1000)}

        return api

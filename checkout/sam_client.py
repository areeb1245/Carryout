"""
Client for a SAM 3.1 endpoint on Modal.

    POST /v1/segment  {image_base64, prompts, threshold, return_masks, exemplars?}
        -> {image: {w, h}, detections: [{label, box, score, polygons, exemplar?}], ms}
    POST /v1/embed    {images_base64} -> {vectors, dim, ms}      (sam3-supermarket only)

`exemplars` are SAM 3's visual prompts: boxes round examples in the same picture ("more like this one").

Two apps speak this: `sam3-supermarket` (modal_app/sam_checkout.py: all phrases in one pass on an L40S, up to 16 a
request) and the older shared `deck-twin-sam31` (one phrase at a time on an A10G, up to 8, 240 requests a minute).
`SAM_URL` and `SAM_API_KEY` choose which; `SAM_MAX_PROMPTS` and `SAM_MAX_FPS` say what it can take.

Boxes and polygon points are normalised to 0..1. The key stays in this process; the browser never sees it.
`cache_dir` stores each answer on disk by a hash of the request, so a clip can be replayed while tuning the logic
without paying for the GPU again.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent


def load_env(path: Path = ROOT / ".env") -> dict[str, str]:
    env = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                key, value = line.split("=", 1)
                env[key.strip()] = value.strip()
    # the process environment wins: a hosted copy (Render) has no .env, only environment variables
    return {**env, **os.environ}


MAX_PROMPTS = int(load_env().get("SAM_MAX_PROMPTS", 8))
MAX_FPS = float(load_env().get("SAM_MAX_FPS", 3))


class SamError(RuntimeError):
    pass


class SamClient:
    def __init__(self, url: str | None = None, api_key: str | None = None, cache_dir: Path | None = None, timeout: float = 180.0):
        env = load_env()
        self.url = (url or env.get("SAM_URL", "")).rstrip("/")
        self.api_key = api_key or env.get("SAM_API_KEY", "")
        if not self.url or not self.api_key:
            raise SamError("SAM_URL and SAM_API_KEY must be set in .env")
        self.cache_dir = cache_dir
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)
        # one keep-alive connection pool: a fresh TLS handshake per frame costs more than the model does
        self.http = httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=True,
            headers={"X-API-Key": self.api_key},
            limits=httpx.Limits(max_keepalive_connections=8, keepalive_expiry=120),
        )

    async def close(self) -> None:
        await self.http.aclose()

    async def health(self) -> dict:
        r = await self.http.get(f"{self.url}/health")
        r.raise_for_status()
        return r.json()

    async def segment(self, jpeg: bytes, prompts: list[str], threshold: float = 0.3, exemplars: list[dict] | None = None) -> dict:
        """All `prompts` on one image. More than 8 are split over parallel requests and merged. `exemplars`
        ([{prompt, label, boxes, labels}], boxes x0 y0 x1 y1 in 0..1 on this image) go with the first request."""
        chunks = [prompts[i : i + MAX_PROMPTS] for i in range(0, len(prompts), MAX_PROMPTS)] or [[]]
        parts = await asyncio.gather(*(self._segment(jpeg, chunk, threshold, exemplars if i == 0 else None) for i, chunk in enumerate(chunks)))
        merged = dict(parts[0])
        merged["detections"] = [d for part in parts for d in part["detections"]]
        merged["ms"] = max(part.get("ms", 0) for part in parts)
        return merged

    async def embed(self, pictures: list[bytes]) -> list[list[float]]:
        """A vector per picture (JPEG), L2-normalised: the cosine of two says how alike they look."""
        out: list[list[float]] = []
        for i in range(0, len(pictures), 16):
            batch = pictures[i : i + 16]
            r = await self._post("/v1/embed", {"images_base64": [base64.b64encode(b).decode() for b in batch]})
            out.extend(r["vectors"])
        return out

    async def _segment(self, jpeg: bytes, prompts: list[str], threshold: float, exemplars: list[dict] | None = None) -> dict:
        cache_file = None
        key = [prompts, threshold] + ([exemplars] if exemplars else [])  # text-only keys stay as they were
        if self.cache_dir:
            digest = hashlib.sha1(jpeg + json.dumps(key).encode()).hexdigest()
            cache_file = self.cache_dir / f"{digest}.json"
            if cache_file.exists():
                return json.loads(cache_file.read_text())
        body = {"image_base64": base64.b64encode(jpeg).decode(), "prompts": prompts, "threshold": threshold, "return_masks": True}
        if exemplars:
            body["exemplars"] = exemplars
        result = await self._post("/v1/segment", body)
        if cache_file:
            cache_file.write_text(json.dumps(result))
        return result

    async def _post(self, path: str, body: dict) -> dict:
        for attempt in range(4):
            try:
                r = await self.http.post(f"{self.url}{path}", json=body)
            except httpx.TransportError as e:
                if attempt == 3:
                    raise SamError(f"endpoint unreachable: {e}") from e
                await asyncio.sleep(0.5 * (attempt + 1))
                continue
            if r.status_code == 429:
                await asyncio.sleep(float(r.headers.get("retry-after", 2)) if attempt else 1.0)
                continue
            if r.status_code >= 500 and attempt < 3:
                await asyncio.sleep(0.5 * (attempt + 1))
                continue
            if r.status_code != 200:
                raise SamError(f"endpoint answered {r.status_code}: {r.text[:200]}")
            return r.json()
        raise SamError("endpoint kept refusing the request (rate limit)")

"""
A second opinion from a vision language model (the DeepSeek endpoint on Modal).

The engine decides picks from mask geometry. This asks a different kind of model to look at the same frame, with a box
drawn round the item, and say whether the shopper is holding something and which catalogue product it is. Where the
local OCR read any text off the item, the model is given that too. One answer is one "look"; the engine takes up to
three looks at a pick, on different frames, and settles the product or disputes the pick from the looks together.

`name_item` is the same question for an item still on its shelf (which of these books is this spine?), used to read
the stock before anyone touches it. Both run beside the frame loop, never in it.
"""
from __future__ import annotations

import base64
import json

import cv2
import httpx
import numpy as np

from .catalog import Product
from .sam_client import load_env

HELD = (
    "You check one frame from a cashierless shop's camera. A yellow box marks one object that the tracking system "
    "believes a shopper has in their hand. Reply with JSON only: "
    '{"holding": true or false, "product": one catalogue id or null, "why": at most 12 words}. '
    '"holding" is true when a person is carrying the object: in a hand, under an arm, or in a stack they hold. '
    'It is false when the object rests on a shelf, table or floor with nobody carrying it. '
    '"product" is null when the object is none of the catalogue entries. '
    "Where the object carries text (a title, a brand), go by the text you can read, not by colour alone."
)
HANDS = (
    "You check one frame from a shop's camera. A yellow box marks one shopper. "
    "List every catalogue item this person is carrying (in a hand, under an arm, in a stack they hold), going by "
    "the text you can read on each. Reply with JSON only: "
    '{"hands_visible": true or false, "products": [catalogue ids], "unreadable": number of carried items whose text '
    'you cannot read, "why": at most 16 words, quoting the text you read}. '
    "Leave out anything you cannot read well enough to be sure of, and anything standing on a shelf or lying on a table. "
    "An empty list is a fine answer."
)
SHELF = (
    "You check one frame from a shop's camera. A yellow box marks one item standing on a shelf. "
    "Say which catalogue entry it is, going by the text printed on it. Reply with JSON only: "
    '{"product": one catalogue id or null, "why": at most 12 words, quoting the text you read}. '
    '"product" is null when you cannot read enough to tell, or it is none of the entries. Do not guess.'
)


class Verifier:
    def __init__(self, env: dict | None = None, timeout: float = 30.0):
        env = load_env() if env is None else env
        self.url = env.get("LLM_URL", "").rstrip("/")
        self.model = env.get("LLM_MODEL", "deepseek-ai/DeepSeek-V4.1-Flash")
        self.enabled = bool(self.url and env.get("LLM_MODAL_KEY") and env.get("LLM_MODAL_SECRET"))
        self.http = httpx.AsyncClient(timeout=timeout, headers={"Modal-Key": env.get("LLM_MODAL_KEY", ""), "Modal-Secret": env.get("LLM_MODAL_SECRET", "")}) if self.enabled else None

    async def close(self) -> None:
        if self.http:
            await self.http.aclose()

    async def held_item(self, frame: np.ndarray, box: list[float], products: list[Product], ocr_text: str = "") -> dict | None:
        """`box` is normalised x0, y0, x1, y1. Returns {holding, product, why}, or None if there was no usable answer."""
        answer = await self._ask(HELD, frame, box, products, ocr_text)
        if answer is None:
            return None
        return {"holding": bool(answer.get("holding")), "product": answer["product"], "why": answer["why"]}

    async def carried(self, frame: np.ndarray, box: list[float], products: list[Product]) -> dict | None:
        """Which of `products` the boxed shopper is carrying. Returns {products: [ids], unreadable, hands_visible, why}, or None."""
        answer = await self._ask(HANDS, frame, box, products, "")
        if answer is None:
            return None
        known = {p.id for p in products}
        listed = answer.get("products") if isinstance(answer.get("products"), list) else []
        try:
            unreadable = max(0, int(answer.get("unreadable") or 0))
        except (TypeError, ValueError):
            unreadable = 0
        return {"products": [p for p in dict.fromkeys(listed) if isinstance(p, str) and p in known], "unreadable": unreadable, "hands_visible": answer.get("hands_visible") is not False, "why": answer["why"]}

    async def name_item(self, frame: np.ndarray, box: list[float], products: list[Product], ocr_text: str = "") -> dict | None:
        """Which of `products` the boxed shelf item is. Returns {product, why}, or None."""
        return await self._ask(SHELF, frame, box, products, ocr_text)

    async def _ask(self, system: str, frame: np.ndarray, box: list[float], products: list[Product], ocr_text: str) -> dict | None:
        if not self.enabled:
            return None
        h, w = frame.shape[:2]
        marked = frame if w <= 1280 else cv2.resize(frame, (1280, round(h * 1280 / w)), interpolation=cv2.INTER_AREA)
        marked = marked.copy()
        mh, mw = marked.shape[:2]
        pad = 0.012
        cv2.rectangle(marked, (int(max(0, box[0] - pad) * mw), int(max(0, box[1] - pad) * mh)), (int(min(1, box[2] + pad) * mw), int(min(1, box[3] + pad) * mh)), (0, 255, 255), 2)
        whole = cv2.imencode(".jpg", marked, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()
        # a close-up of the item and what is round it, at full resolution, so small print can still be read
        bw, bh = box[2] - box[0], box[3] - box[1]
        grow = 0.8 if max(bw, bh) < 0.25 else 0.25
        x0, y0 = int(max(0, box[0] - grow * bw) * w), int(max(0, box[1] - grow * bh) * h)
        x1, y1 = int(min(1, box[2] + grow * bw) * w), int(min(1, box[3] + grow * bh) * h)
        close = frame[y0:y1, x0:x1]
        scale = 448 / max(close.shape[:2])
        if scale > 1:
            close = cv2.resize(close, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        zoom = cv2.imencode(".jpg", close, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes()

        entries = []
        for p in products:
            entry = {"id": p.id, "name": p.name + ("" if p.billable else " (the shopper's own, not for sale)")}
            if p.author:
                entry["by"] = p.author
            entry["looks_like"] = p.cues or p.prompt
            entries.append(entry)
        text = "Catalogue: " + json.dumps(entries, ensure_ascii=False)
        if ocr_text:
            text += f'\nText an OCR program read inside the box (it may contain mistakes): "{ocr_text[:200]}"'
        text += "\nFirst picture: the whole frame with the yellow box. Second picture: a close-up of the boxed area."
        body = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 200,
            "reasoning_effort": "none",
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text},
                        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(whole).decode()}},
                        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(zoom).decode()}},
                    ],
                },
            ],
        }
        try:
            r = await self.http.post(f"{self.url}/v1/chat/completions", json=body)
            r.raise_for_status()
            answer = json.loads(r.json()["choices"][0]["message"]["content"])
            product = answer.get("product")
            known = {p.id for p in products}
            return {**answer, "product": product if isinstance(product, str) and product in known else None, "why": str(answer.get("why", ""))}
        except (httpx.HTTPError, KeyError, IndexError, ValueError, TypeError, AttributeError):
            return None

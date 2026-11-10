"""
A second opinion from a vision language model (the DeepSeek endpoint on Modal).

The engine decides picks from mask geometry. This asks a different kind of model to look at the same frame, with a box
drawn round the item, and say whether the shopper is holding something and which catalogue product it is. Where the
local OCR read any text off the item, the model is given that too. One answer is one "look"; the engine takes up to
three looks at a pick, on different frames, and settles the product or disputes the pick from the looks together.

`name_item` is the same question for an item still on its shelf (which of these books is this spine?), used to read
the stock before anyone touches it. Both run beside the frame loop, never in it.

`audit` is the last word before money moves: one call per walk-out with the shopper's bill and the pictures in which
they were seen carrying things. It answers line by line: kept, put back, or unclear. "Put back" is never charged;
"unclear" is charged only when the title was already read in their hands on the floor (pipeline._audit).
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
STILL = (
    "You check one frame from a shop's camera, taken as a shopper leaves. A yellow box marks the shopper. "
    "Say which catalogue items this person is still carrying: in a hand, under an arm, or held against the body. "
    "Go by each item's look (shape, colour, packaging) and any text printed on it; catalogue pictures of some "
    "products follow the frame. Reply with JSON only: "
    '{"hands_seen": how many of this shopper\'s hands you can see clearly, 0, 1 or 2, "products": [catalogue ids], '
    '"unknown": number of carried things you cannot match to any catalogue entry, "why": at most 16 words}. '
    "The shopper's own things (marked as such) are in the catalogue so you can tell them apart: list them too when "
    "carried. An item resting on a shelf, table or floor, or in somebody else's hand, is not carried. "
    "An empty list is a fine answer."
)
SHELF = (
    "You check one frame from a shop's camera. A yellow box marks one item standing on a shelf. "
    "Say which catalogue entry it is, going by the text printed on it. Reply with JSON only: "
    '{"product": one catalogue id or null, "why": at most 12 words, quoting the text you read}. '
    '"product" is null when you cannot read enough to tell, or it is none of the entries. Do not guess.'
)
AUDIT = (
    "You are the last check before a shopper at a cashierless shop is charged. You get their bill and pictures of them "
    "from the shop's camera, oldest first, each marked with a letter in its top left corner. The earlier pictures are "
    "moments at which items were read in their hands. The last picture is the whole scene the last time they were seen. "
    "For every bill line report two observations. "
    '"held_in": the letter of a picture in which this shopper is holding that exact item (in a hand, under an arm, or '
    "in a stack they hold), or null if no picture shows that. Where the item carries text, such as a book, only count "
    'a picture in which you can read its title on it, and give the words you read as "read". '
    '"put_down_in": the letter of a later picture in which you can see that same item at rest again on a shelf, '
    "table or counter with nobody's hand on it; otherwise null. An item still in a hand is not put down, even when "
    "that hand is near a shelf. An item you cannot see, because a hand, their body, a bag or another item hides it or "
    "it is out of the picture, is not put down. "
    '"count": how many of that product the shopper holds at once, in the picture that shows the most of them. '
    'Reply with JSON only: {"lines": [{"line": bill line number, "held_in": letter or null, "read": words read on '
    'the item or "", "put_down_in": letter or null, "count": number, "why": at most 14 words on what you saw}]}. '
    "One entry for every bill line and no others. Report only what the pictures show; when unsure, use null."
)
DESCRIBE = (
    "You help a shop put a product into the catalogue of its camera checkout. You get one frame from the shop's camera "
    "with a yellow box round one object, and a close-up of the box. Describe the whole object inside the box as it "
    "would be sold, including its packaging: two tissue rolls wrapped together are one pack of two, not a roll. "
    "Reply with JSON only: "
    '{"name": a short product name for a shelf label, with the brand if you can read it, "maker": the brand or maker '
    'or "", "phrases": 3 or 4 short noun phrases (1 to 4 words, lower case) an object detector could be asked for to '
    "find this object, most specific first and the most general last, each naming the whole packaged object, "
    '"category": a shop category, "section": a section within it, "cues": at most 14 words on what it looks like '
    '(colour, shape, packaging, printed text), "text": words printed on it that you can read, or ""}. '
    "Where the shop already has categories and sections that fit, use those names."
)
VERDICTS = ("kept", "put_back", "unclear")


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

    async def held_item(self, frame: np.ndarray, box: list[float], products: list[Product], ocr_text: str = "", refs: list[tuple[str, bytes]] | None = None) -> dict | None:
        """`box` is normalised x0, y0, x1, y1. `refs` are catalogue pictures, (product id, JPEG), shown after the frame
        so the model can compare. Returns {holding, product, why}, or None if there was no usable answer."""
        answer = await self._ask(HELD, frame, box, products, ocr_text, refs)
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

    async def still_carried(self, frame: np.ndarray, box: list[float], products: list[Product], refs: list[tuple[str, bytes]] | None = None) -> dict | None:
        """At the door, for products told apart by their look rather than a title: which of `products` the boxed
        shopper still carries. Returns {products: [ids], unknown, hands_seen: 0..2, why}, or None."""
        answer = await self._ask(STILL, frame, box, products, "", refs)
        if answer is None:
            return None
        known = {p.id for p in products}
        listed = answer.get("products") if isinstance(answer.get("products"), list) else []

        def number(key: str, top: int) -> int:
            try:
                return min(top, max(0, int(answer.get(key) or 0)))
            except (TypeError, ValueError):
                return 0

        return {"products": [p for p in dict.fromkeys(listed) if isinstance(p, str) and p in known], "unknown": number("unknown", 9), "hands_seen": number("hands_seen", 2), "why": answer["why"]}

    async def name_item(self, frame: np.ndarray, box: list[float], products: list[Product], ocr_text: str = "") -> dict | None:
        """Which of `products` the boxed shelf item is. Returns {product, why}, or None."""
        return await self._ask(SHELF, frame, box, products, ocr_text)

    async def audit(self, lines: list[dict], pictures: list[tuple[float, np.ndarray]]) -> dict | None:
        """The check before a charge. `lines` are {line, name, by, looks_like}; `pictures` are (seconds into the
        visit, image), oldest first. Returns {line: {verdict: kept | put_back | unclear, why}} with an entry for
        every line, or None when the model gave no usable answer."""
        if not self.enabled or not lines or not pictures:
            return None
        content = [{"type": "text", "text": "Bill: " + json.dumps(lines, ensure_ascii=False) + "\nPictures, oldest first: " + ", ".join(f"{chr(65 + i)} at {t:.0f} s" for i, (t, _) in enumerate(pictures))}]
        for i, (_, image) in enumerate(pictures):
            content.append({"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(self.lettered(image, chr(65 + i))).decode()}})
        body = {
            "model": self.model, "temperature": 0, "max_tokens": 120 + 60 * len(lines), "reasoning_effort": "none", "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": AUDIT}, {"role": "user", "content": content}],
        }
        try:
            r = await self.http.post(f"{self.url}/v1/chat/completions", json=body)
            r.raise_for_status()
            answer = json.loads(r.json()["choices"][0]["message"]["content"])
            rows = answer["lines"]
            asked = {line["line"] for line in lines}
            letters = [chr(65 + i) for i in range(len(pictures))]
            out = {}
            for row in rows:
                number = int(row.get("line"))
                if number not in asked or number in out:
                    continue
                held, down = self._letter(row.get("held_in"), letters), self._letter(row.get("put_down_in"), letters)
                read, why = str(row.get("read") or "").strip()[:60], str(row.get("why") or "").strip()[:100]
                if held is None:
                    out[number] = {"verdict": "unclear", "why": why or "Not visible in hand in any picture"}
                elif down is not None and letters.index(down) > letters.index(held):
                    out[number] = {"verdict": "put_back", "why": f"In hand in picture {held}, back on the shelf in picture {down}"}
                else:
                    seen = f': "{read}"' if read else ""
                    out[number] = {"verdict": "kept", "why": f"In hand in picture {held}{seen}"}
            for number in asked - set(out):  # a line the model skipped is a line it did not confirm
                out[number] = {"verdict": "unclear", "why": "No answer for this line"}
            # A product on the bill twice is two items, and one in the hand says "held" for both lines. So the copies
            # beyond the most of it seen in a hand at once are not confirmed: never bill two for one.
            counts: dict[str, int] = {}
            for row in rows:
                line = next((x for x in lines if x["line"] == int(row.get("line", -1))), None)
                if line is not None:
                    try:
                        counts[line["name"]] = max(counts.get(line["name"], 0), int(row.get("count") or 0))
                    except (TypeError, ValueError):
                        pass
            nth: dict[str, int] = {}
            for line in sorted(lines, key=lambda x: x["line"]):
                if out[line["line"]]["verdict"] != "kept":
                    continue
                name = line["name"]
                nth[name] = nth.get(name, 0) + 1
                many = sum(1 for x in lines if x["name"] == name)
                held = max(1, counts.get(name, 0))  # seen held at all is one, whatever the count says
                if many > 1 and nth[name] > held:
                    out[line["line"]] = {"verdict": "unclear", "why": f"Only {held} in hand at once, {many} on the bill"}
            return out
        except (httpx.HTTPError, KeyError, IndexError, ValueError, TypeError, AttributeError):
            return None

    async def describe(self, frame: np.ndarray, box: list[float], categories: list[str]) -> dict | None:
        """What the boxed object is, for adding it to the catalogue: {name, maker, phrases, category, section, cues, text}."""
        if not self.enabled:
            return None
        h, w = frame.shape[:2]
        marked = frame.copy()
        cv2.rectangle(marked, (int(box[0] * w), int(box[1] * h)), (int(box[2] * w), int(box[3] * h)), (0, 255, 255), max(2, w // 400))
        k = min(1.0, 1280 / max(h, w))
        whole = cv2.resize(marked, None, fx=k, fy=k, interpolation=cv2.INTER_AREA) if k < 1 else marked
        bw, bh = box[2] - box[0], box[3] - box[1]
        close = frame[int(max(0, box[1] - 0.15 * bh) * h): int(min(1, box[3] + 0.15 * bh) * h), int(max(0, box[0] - 0.15 * bw) * w): int(min(1, box[2] + 0.15 * bw) * w)]
        if close.size == 0:
            return None
        scale = 512 / max(close.shape[:2])
        if scale > 1:
            close = cv2.resize(close, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        pictures = [cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tobytes() for img in (whole, close)]
        text = "The shop's categories and sections so far: " + (", ".join(categories[:40]) or "none yet")
        body = {
            "model": self.model, "temperature": 0, "max_tokens": 260, "reasoning_effort": "none", "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": DESCRIBE},
                {"role": "user", "content": [{"type": "text", "text": text}, *({"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(b).decode()}} for b in pictures)]},
            ],
        }
        try:
            r = await self.http.post(f"{self.url}/v1/chat/completions", json=body)
            r.raise_for_status()
            answer = json.loads(r.json()["choices"][0]["message"]["content"])
            phrases = [str(x).strip().lower()[:60] for x in answer.get("phrases") or [] if str(x).strip()]
            return {
                "name": str(answer.get("name") or "").strip()[:80],
                "maker": str(answer.get("maker") or "").strip()[:80],
                "phrases": list(dict.fromkeys(phrases))[:4],
                "category": str(answer.get("category") or "").strip()[:40],
                "section": str(answer.get("section") or "").strip()[:40],
                "cues": str(answer.get("cues") or "").strip()[:160],
                "text": str(answer.get("text") or "").strip()[:120],
            }
        except (httpx.HTTPError, KeyError, IndexError, ValueError, TypeError, AttributeError):
            return None

    @staticmethod
    def _letter(value, letters: list[str]) -> str | None:
        """A picture letter the model named, if it is one of the pictures it was shown."""
        text = str(value or "").strip().upper()
        return text[:1] if text[:1] in letters and len(text) <= 2 else None

    @staticmethod
    def lettered(image: np.ndarray, letter: str, side: int = 896) -> bytes:
        """The picture at a size the model reads well, with its letter in the top left corner."""
        h, w = image.shape[:2]
        k = min(1.0, side / max(h, w))
        small = cv2.resize(image, (round(w * k), round(h * k)), interpolation=cv2.INTER_AREA) if k < 1 else image.copy()
        cv2.rectangle(small, (0, 0), (44, 44), (0, 255, 255), -1)
        cv2.putText(small, letter, (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 3, cv2.LINE_AA)
        return cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tobytes()

    async def _ask(self, system: str, frame: np.ndarray, box: list[float], products: list[Product], ocr_text: str, refs: list[tuple[str, bytes]] | None = None) -> dict | None:
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
        refs = refs or []
        if refs:
            text += (
                "\nAfter those, catalogue pictures of some products, in this order: " + ", ".join(pid for pid, _ in refs)
                + ". They show what those products look like; the same product can be seen from another side or in other light."
            )
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
                        *({"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()}} for _, jpeg in refs),
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

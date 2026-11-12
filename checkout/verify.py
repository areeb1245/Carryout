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
# The check before charging, as separate questions, each about one picture (Verifier.check_visit).
ASK_HELD = (
    "You check one picture from a cashierless shop's camera. Question: is the shopper holding the item named below "
    "(in a hand, under an arm, or in a stack they hold)? Where the item carries a title, answer \"yes\" only if you "
    "can read that title on an item in their hands. Reply with JSON only: "
    '{"answer": "yes" or "no" or "unknown", "read": the words you read on it or "", "why": at most 14 words}. '
    '"no" only when you can see their hands clearly and that item is not in them. When unsure, answer "unknown".'
)
ASK_SHELF = (
    "You check one picture from a cashierless shop's camera, taken after a shopper was seen holding the items listed "
    "below. Question: is a hand putting an item onto or into a shelf here? Count it as placing when a hand is pushing "
    "or sliding an item in among the things on the shelf, is resting it on the shelf, or has only just let go of it "
    "there, whether or not the hand has let go yet. An item held up or carried away from the shelf, not touching it, "
    "is not being placed. If it is being placed, which catalogue item is it? Reply with JSON only: "
    '{"placing": "yes" or "no" or "unknown", "product": one catalogue id or "unknown", "read": words you read on '
    'that item or "", "why": at most 14 words}. Give a product id only when you can read its title or it is '
    'unmistakable; otherwise "unknown". When unsure, use "unknown".'
)
ASK_LEAVING = (
    "You check one picture from a cashierless shop's camera, one of the last before a shopper leaves. Question: what "
    "is in the shopper's hands (in a hand, under an arm, or in a stack they hold)? List each item once. Reply with "
    'JSON only: {"hands": "visible" or "hidden" or "unknown", "items": [{"product": one catalogue id or "unknown", '
    '"read": the words you read on it or ""}], "why": at most 16 words}. Give a product id only when you can read its '
    'title or it is unmistakable; list an item you cannot name as "unknown". Leave out anything standing on a shelf '
    "or lying on a table. An empty list is a fine answer."
)
ASK_WHICH = (
    "You check one picture from a cashierless shop's camera, one of the last before a shopper leaves. They are "
    "carrying a book whose title cannot be read here. Question: which of the candidate books below is it? First say the "
    "main colour of that book's cover or spine, then pick the candidate whose description has that colour (a book's "
    "spine and cover share its colour). Use its picture, shape and any letters you can make out to choose between "
    "candidates of the same colour. A book in their hands whose title you can read is not it. Reply with JSON only: "
    '{"colour": the main colour of that book, "product": one candidate id or "unknown", "why": at most 14 words on what '
    'you see}. Answer "unknown" when no candidate has that colour, or two or more fit it equally well.'
)
ASK_PLACED = (
    "You check one picture from a cashierless shop's camera. A shopper is putting a book onto a shelf here, or is about "
    "to, or has only just let go of it there. Its title may not be readable. Question: which of the candidate books "
    "below is it? First say the main colour of that book's cover or spine, then pick the candidate whose description "
    "has that colour (a book's spine and cover share its colour). Use its picture, shape and any letters you can make "
    "out to choose between candidates of the same colour. Reply with JSON only: "
    '{"colour": the main colour of the book, "product": one candidate id or "unknown", "why": at most 14 words}. '
    'Answer "unknown" when no candidate has that colour, or two or more fit it equally well.'
)
ANSWERS = ("yes", "no", "unknown")
CHECK_CALLS = 12  # questions to the model per walk-out, at most
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

    async def carried(self, frame: np.ndarray, box: list[float], products: list[Product], hands: list[float] | None = None) -> dict | None:
        """Which of `products` the boxed shopper is carrying. `hands`, when given, is the part of the frame with their
        hands and what is in them: the close-up shows that, enlarged, rather than the whole shopper. Returns
        {products: [ids], unreadable, hands_visible, why}, or None."""
        answer = await self._ask(HANDS, frame, box, products, "", zoom=hands)
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

    async def check_visit(self, lines: list[dict], evidence: dict, catalogue: list[dict], budget: int = CHECK_CALLS, doubt: dict | None = None) -> dict | None:
        """The check before charging, step by step: separate questions about separate pictures, checked against each
        other and against what the floor saw, with one more look where they disagree. At most `budget` calls.

        `lines`     {line, product, name, by, looks_like, door, door_why, last_read_t, titled}. `titled`: told apart
                    by its title (a book); a product without one, held and not seen put down, is kept. `door` is "put_back"
                    when the door look already found the item gone; such a line can only stay off the bill.
        `evidence`  {held: {line: (t, image)}, pick: {line: (t, image)}, views: [(t, image)] the last close views of
                    their hands, oldest first, shelf: [(t, image)] views at moments the floor saw something go back,
                    scene: (t, image) or None}.
        `catalogue` the products the questions may name: {id, name, by, looks_like}.
        `doubt`     {spare, candidates: [{product, name, looks_like, line or None}]}: the door look counted `spare`
                    unreadable books in their hands and more titles that could be them. `line` None is a title the
                    floor saw go back after it was last read, which comes back on the bill (`reinstate`) only if
                    both views say the unreadable book looks like it.

        Returns {verdicts: {line: {verdict, why, absent, pictures}}, reinstate: [{product, why}], steps: [{step, line,
        ask, t, answer, why}], pictures: [(t, image, caption)]}, or None when no question got an answer.

        The steps.
        0  Which      (with `doubt`) which candidate the unreadable book in their hands looks like, in the last
                      two views
        1  Leaving    what is in their hands in each of the last two views
        2  Held       for each line: is this item in their hands, in the picture it was last read in?
        3  Shelf      for each line: in the first view after that, is something being put on a shelf, and what?
                      (an unreadable book going back: which of the titles does it look like?)
        4  Cross-check, with one more look (the newest view) where the answers disagree
        5  Verdict    put back when seen put on the shelf, or found gone at the door, and not in hand as they left;
                      kept when in hand as they left; otherwise unclear. "Unclear" is never charged unless the
                      caller knows the title was confirmed in their hands earlier (and `absent` is not set)."""
        if not self.enabled or not lines:
            return None
        known = {c["id"]: c["name"] for c in catalogue}
        steps: list[dict] = []
        pictures: list[tuple[float, np.ndarray, str]] = []
        calls = {"n": 0, "answered": 0}
        views = list(evidence.get("views") or [])
        if not views and evidence.get("scene"):
            views = [evidence["scene"]]

        async def ask(kind: str, line: dict | None, t: float, image: np.ndarray, system: str, question: str, listing: list | None = None) -> dict | None:
            if calls["n"] >= budget or image is None:
                return None
            calls["n"] += 1
            step = len(steps) + 1
            text = question + ("\nCatalogue: " + json.dumps(catalogue, ensure_ascii=False) if listing is None else "")
            answer = await self._one(system, text, image)
            if answer is not None:
                calls["answered"] += 1
            short = self._gist(kind, answer, known)
            steps.append({"step": step, "line": line["line"] if line else None, "name": line["name"] if line else None, "ask": kind, "t": round(t, 2), "answer": short, "why": str((answer or {}).get("why") or "")[:100]})
            pictures.append((t, image, f"step {step}: {self._question(kind, line)} {short}"))
            return answer

        def about(line: dict) -> str:
            return json.dumps({k: line[k] for k in ("name", "by", "looks_like") if line.get(k)}, ensure_ascii=False)

        candidates = {c["product"]: c for c in (doubt or {}).get("candidates", [])}
        room = max(0, int((doubt or {}).get("spare", 0)))

        # 1. leaving: the last two views of their hands
        leaving: dict[str, list[tuple[float, str]]] = {}
        unknown_out, hands_out, counts = 0, False, {}
        for t, image in views[-2:]:
            answer = await ask("leaving", None, t, image, ASK_LEAVING, "What is in the shopper's hands as they leave?")
            if not answer:
                continue
            hands_out = hands_out or str(answer.get("hands")).lower() == "visible"
            items = answer.get("items") if isinstance(answer.get("items"), list) else []
            seen_here: dict[str, int] = {}
            unknown_here = 0
            for it in items:
                pid = it.get("product") if isinstance(it, dict) else None
                if isinstance(pid, str) and pid in known:
                    leaving.setdefault(pid, []).append((t, str(it.get("read") or "")[:60]))
                    seen_here[pid] = seen_here.get(pid, 0) + 1
                else:
                    unknown_here += 1
            unknown_out = max(unknown_out, unknown_here)
            for pid, n in seen_here.items():
                counts[pid] = max(counts.get(pid, 0), n)

        # 2. held, and 3. shelf, line by line
        held: dict[int, tuple[str, float]] = {}
        shelved: dict[str, tuple[float, str]] = {}  # product -> when it was seen going onto the shelf, and how it was told
        shelf_seen: dict[float, tuple[str | None, str]] = {}  # one shelf question per picture, whichever line asks
        placings: list[tuple[float, str | None]] = []  # something seen going onto a shelf: when, and which if told
        everyone = [*lines, *candidates.values()]

        async def on_shelf(line: dict, t: float, image: np.ndarray) -> tuple[str | None, str]:
            """Is something going onto the shelf in this picture, and which of the titles is it?"""
            if t in shelf_seen:
                return shelf_seen[t]
            shelf_seen[t] = (None, "")
            others = ", ".join(x["name"] for x in everyone)
            answer = await ask("shelf", line, t, image, ASK_SHELF, "Is an item being put on a shelf, and which? The shopper was last seen holding: " + others)
            if not answer or self._yes(answer.get("placing")) != "yes":
                return shelf_seen[t]
            # not a title read in their hands after this picture (whatever went onto the shelf, not that one), nor one
            # plainly in their hands as they leave
            possible = {x["product"]: x for x in everyone if x["product"] not in leaving and not ((x.get("last_read_t") or -1.0) > t)}
            titles = {x["product"]: x for x in everyone}
            named = answer.get("product") if answer.get("product") in titles else None
            read = str(answer.get("read") or "")[:60]
            if possible:
                # Which title does it look like, colour first? A title named on a book moving in a hand is often a guess
                # (or the book in the other hand), so the look decides when it gives an answer. All titles are
                # candidates, the ones still in their hands too: a book that only looks like one of those is not
                # going back, and a placing named as one of those that no other title fits is not believed.
                listing = [{"id": x["product"], "name": x["name"], "looks_like": x.get("looks_like", "")} for x in titles.values()]
                again = await ask("placed", line, t, image, ASK_PLACED, "Candidates: " + json.dumps(listing, ensure_ascii=False), listing)
                looked = (again or {}).get("product")
                if looked in titles:
                    named, read = looked, f"looks like {titles[looked]['name']}"
            if named in titles and named not in possible:
                return shelf_seen[t]  # a book still in their hands after this: the placing is not believed
            product = named if named in possible else None
            shelf_seen[t] = (product, read)
            placings.append((t, product))
            return shelf_seen[t]

        # the titles the door could not account for that are not lines of the bill (their outline went back) get
        # the shelf question too: seeing one go back is what tells the others apart
        extra = [{**c, "door": None} for p, c in candidates.items() if not c.get("line")]
        for line in [*lines, *extra]:
            n = line["line"]
            shot = (evidence.get("held") or {}).get(n) or (evidence.get("pick") or {}).get(n) if n else None
            if shot and line.get("door") != "put_back" and line["product"] not in leaving:  # read as they left: kept anyway
                answer = await ask("held", line, shot[0], shot[1], ASK_HELD, "Is this item in the shopper's hands? Item: " + about(line))
                if answer:
                    held[n] = (self._yes(answer.get("answer")), shot[0])
            since = line.get("last_read_t")
            if since is None and shot:
                since = shot[0]
            if since is None:
                continue
            after = sorted((v for v in [*(evidence.get("shelf") or []), *views] if v[0] > since), key=lambda v: v[0])
            # the first view after it was last seen in hand, and, for a line not read in hand as they left, the next
            # three too: a book going back takes a second or two
            for t, image in after[: 1 if line["product"] in leaving else 4]:
                product, read = await on_shelf(line, t, image)
                if product:
                    shelved.setdefault(product, (t, read))
                    break

        placed = {x["line"]: shelved[x["product"]] for x in lines if x["product"] in shelved}

        # 4a. which: the book they could not read as they left, against the titles the door said it could be. Titles
        # seen going onto the shelf are not it. If only as many titles are left as there are unreadable books, those
        # are the ones; otherwise the last two views are asked which one it looks like.
        chosen: list[str] = []
        left = [p for p in candidates if p not in shelved]
        if candidates and left and len(left) <= room and len(left) < len(candidates):
            chosen = left
            how = {p: "the other titles it could be were seen going onto the shelf" for p in chosen}
        elif candidates and left and len(left) <= room and not placings and all(candidates[p].get("line") for p in left):
            # as many unreadable books in their hands as titles on the bill they could be, and nothing seen going
            # onto a shelf: a stack whose lower books cannot be read, the benefit of the doubt as before
            chosen = left
            how = {p: "as many unreadable books in their hands as such titles, none seen going onto a shelf" for p in chosen}
        elif candidates and left and views:
            listing = [{"id": p, "name": candidates[p]["name"], "looks_like": candidates[p].get("looks_like", "")} for p in left]
            votes: dict[str, int] = {}
            for t, image in views[-2:]:
                answer = await ask("which", None, t, image, ASK_WHICH, "Candidates: " + json.dumps(listing, ensure_ascii=False), listing)
                product = (answer or {}).get("product")
                if product in left:
                    votes[product] = votes.get(product, 0) + 1
            named = sorted(votes, key=lambda p: -votes[p])
            if len(named) > room:  # the views name more books than there are unreadable ones: only those clearly ahead
                named = [p for p in named if votes[p] > votes[named[room]]]
            # a title back onto the bill, or one the door saw go missing, needs both views; one still on it, one
            chosen = [p for p in named if votes[p] >= (1 if candidates[p].get("line") and candidates[p].get("sure") else 2)][:room]
            how = {p: f"the book looks like {candidates[p]['name']}" for p in chosen}

        if calls["answered"] == 0:
            return None
        newest = views[-1] if views else None

        def put_down_after(t: float | None, product: str) -> bool:
            """Was something seen going onto a shelf after `t` that is not told apart as another item? Without that,
            a line missing from the last pictures is under others in a stack, not put down."""
            return any(at > (t if t is not None else -1.0) and p in (None, product) for at, p in placings)

        # 4 and 5. cross-check, one more look where the answers disagree, then the verdict
        verdicts: dict[int, dict] = {}
        for line in lines:
            n, product = line["line"], line["product"]
            out = leaving.get(product)
            gone = line.get("door") == "put_back"
            down = placed.get(n)
            if (gone or down) and out and newest is not None:
                # put down, yet read in hand in the last views: one more look, at the newest view
                again = await ask("held", line, newest[0], newest[1], ASK_HELD, "Is this item in the shopper's hands? Item: " + about(line))
                if again and self._yes(again.get("answer")) == "yes":
                    verdicts[n] = {"verdict": "kept", "why": f"In hand as they left ({again.get('read') or out[-1][1] or 'read'})", "absent": False}
                    continue
                out = None
            if down and not out:
                verdicts[n] = {"verdict": "put_back", "why": f"Seen put on the shelf at {down[0]:.1f} s" + (f" ({down[1]})" if down[1] else ""), "absent": True}
            elif gone and not out and put_down_after(line.get("last_read_t"), product):
                # gone from the door's last pictures, and something not told apart as another title went onto a
                # shelf after it was last read
                verdicts[n] = {"verdict": "put_back", "why": line.get("door_why") or "Not in hand as they left", "absent": True}
            elif gone and not out:
                # gone from the last pictures, yet nothing seen going onto a shelf after it was last read: a book under
                # others in a stack. Not absent, so a title read twice on the floor still stands
                verdicts[n] = {"verdict": "unclear", "why": f"Not read in hand after {line.get('last_read_t') or 0:.1f} s; nothing seen going onto a shelf", "absent": False}
            elif out:
                read = next((r for _, r in reversed(out) if r), "")
                verdicts[n] = {"verdict": "kept", "why": f"In hand as they left at {out[-1][0]:.1f} s" + (f': "{read}"' if read else ""), "absent": False}
            else:
                seen = held.get(n)
                verdict = {"verdict": "unclear", "why": "Not seen in hand in any picture", "absent": False}
                if seen and seen[0] == "yes":
                    verdict["why"] = f"In hand at {seen[1]:.1f} s, title not readable as they left"
                    if newest is not None and newest[0] > seen[1] and product not in candidates:  # those are settled below
                        again = await ask("held", line, newest[0], newest[1], ASK_HELD, "Is this item in the shopper's hands? Item: " + about(line))
                        said = self._yes((again or {}).get("answer"))
                        if said == "yes":
                            verdict = {"verdict": "kept", "why": f"In hand as they left at {newest[0]:.1f} s", "absent": False}
                        elif said == "no" and hands_out and unknown_out == 0 and put_down_after(seen[1], product):
                            verdict = {"verdict": "unclear", "why": f"In hand at {seen[1]:.1f} s, not in their hands as they left", "absent": True}
                elif seen and seen[0] == "no":
                    verdict["why"] = f"Not in hand in the picture at {seen[1]:.1f} s"
                if verdict["verdict"] == "unclear" and not verdict["absent"] and line.get("titled") is False and seen and seen[0] == "yes":
                    # a product without a title: held, and not seen put down afterwards, is kept (as the one-look check)
                    verdict = {"verdict": "kept", "why": f"In hand at {seen[1]:.1f} s, not seen put down", "absent": False}
                verdicts[n] = verdict
        # the titles the door could not tell apart: the one the unreadable book looks like is with them, the others not
        for c in candidates.values():
            n = c.get("line")
            if not n or verdicts[n]["verdict"] == "kept" and verdicts[n]["why"].startswith("In hand as they left"):
                continue
            if verdicts[n]["verdict"] == "put_back" and verdicts[n].get("absent"):
                continue  # seen going onto the shelf
            names = ", ".join(candidates[p]["name"] for p in chosen)
            if c["product"] in chosen:
                verdicts[n] = {"verdict": "kept", "why": f"In hand as they left, title not readable; {how[c['product']]}", "absent": False}
            elif chosen:
                verdicts[n] = {"verdict": "put_back", "why": f"Not in hand as they left; the unreadable book there is {names}", "absent": True}
            else:
                # Not told apart. Off their hands only if something was seen going onto a shelf after it was last read
                # that could be it; otherwise a stack whose books cannot be counted, and a title read twice on the
                # floor stands (not absent)
                down = put_down_after(c.get("last_read_t"), c["product"])
                verdicts[n] = {"verdict": "unclear", "why": "Not readable as they left, and the book in hand could be any of " + ", ".join(x["name"] for x in candidates.values())
                               + ("" if down else "; nothing seen going onto a shelf"), "absent": down}
        reinstate = [{"product": p, "why": f"In hand as they left, title not readable; {how[p]}"} for p in chosen if not candidates[p].get("line")]
        # A product on the bill twice is two items: copies beyond the most seen in hand at once are not confirmed.
        nth: dict[str, int] = {}
        for line in sorted(lines, key=lambda x: x["line"]):
            if verdicts[line["line"]]["verdict"] != "kept":
                continue
            nth[line["product"]] = nth.get(line["product"], 0) + 1
            many = sum(1 for x in lines if x["product"] == line["product"])
            most = max(1, counts.get(line["product"], 0))
            if many > 1 and nth[line["product"]] > most:
                verdicts[line["line"]] = {"verdict": "unclear", "why": f"Only {most} in hand at once, {many} on the bill", "absent": False}
        for n, verdict in verdicts.items():
            verdict["pictures"] = [s["step"] for s in steps if s["line"] in (n, None)]
        return {"verdicts": verdicts, "reinstate": reinstate, "steps": steps, "pictures": pictures, "calls": calls["n"]}

    @staticmethod
    def _yes(value) -> str:
        text = str(value or "").strip().lower()
        return text if text in ANSWERS else "unknown"

    @staticmethod
    def _question(kind: str, line: dict | None) -> str:
        name = line["name"] if line else ""
        return {"leaving": "What is in the hands as they leave?", "held": f"Is {name} in the hand?", "shelf": "Is an item being put on the shelf, and which?",
                "which": "Which book does the unreadable one in the hand look like?", "placed": "Which book is going onto the shelf?",
}[kind]

    @staticmethod
    def _gist(kind: str, answer: dict | None, known: dict[str, str]) -> str:
        """An answer in a few words, for the visit's steps."""
        if not answer:
            return "No answer"
        if kind == "held":
            read = str(answer.get("read") or "").strip()
            return Verifier._yes(answer.get("answer")).capitalize() + (f' ("{read[:40]}")' if read else "")
        if kind in ("which", "placed"):
            return known.get(answer.get("product"), "Unknown")
        if kind == "shelf":
            placing = Verifier._yes(answer.get("placing"))
            product = answer.get("product") if answer.get("product") in known else None
            return {"yes": f"Yes: {known[product] if product else 'title not readable'}", "no": "No", "unknown": "Unknown"}[placing]
        items = answer.get("items") if isinstance(answer.get("items"), list) else []
        names = [(known[i["product"]] if isinstance(i, dict) and i.get("product") in known else "unreadable item") for i in items]
        return ", ".join(names) if names else ("Nothing" if str(answer.get("hands")).lower() == "visible" else "Hands not visible")

    async def _one(self, system: str, text: str, image: np.ndarray) -> dict | None:
        """One question about one picture, answered in JSON."""
        h, w = image.shape[:2]
        k = min(1.0, 896 / max(h, w))
        small = cv2.resize(image, (max(2, round(w * k)), max(2, round(h * k))), interpolation=cv2.INTER_AREA) if k < 1 else image
        jpeg = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tobytes()
        body = {
            "model": self.model, "temperature": 0, "max_tokens": 260, "reasoning_effort": "none", "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": [{"type": "text", "text": text}, {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()}}]},
            ],
        }
        try:
            r = await self.http.post(f"{self.url}/v1/chat/completions", json=body)
            r.raise_for_status()
            answer = json.loads(r.json()["choices"][0]["message"]["content"])
            return answer if isinstance(answer, dict) else None
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

    async def _ask(self, system: str, frame: np.ndarray, box: list[float], products: list[Product], ocr_text: str, refs: list[tuple[str, bytes]] | None = None,
                   zoom: list[float] | None = None) -> dict | None:
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
        # (or, given `zoom`, of the hands and what they hold, enlarged: a title on a book in a hand is a few pixels tall)
        near = zoom or box
        bw, bh = near[2] - near[0], near[3] - near[1]
        grow = 0.8 if max(bw, bh) < 0.25 else 0.25
        x0, y0 = int(max(0, near[0] - grow * bw) * w), int(max(0, near[1] - grow * bh) * h)
        x1, y1 = int(min(1, near[2] + grow * bw) * w), int(min(1, near[3] + grow * bh) * h)
        close = frame[y0:y1, x0:x1]
        scale = min(3.0, (1024 if zoom else 448) / max(close.shape[:2]))
        if scale > 1:
            close = cv2.resize(close, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        zoom_jpeg = cv2.imencode(".jpg", close, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes()

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
        text += "\nFirst picture: the whole frame with the yellow box. Second picture: " + (
            "a close-up of the boxed shopper's hands and what they hold, enlarged." if zoom else "a close-up of the boxed area.")
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
                        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(zoom_jpeg).decode()}},
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

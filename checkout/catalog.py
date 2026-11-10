"""
A shop's catalogue: what SAM is asked to look for, and what each thing costs.

Each product has a name, a price, a `category` and a `section` within it (Books > Biography > "Elon Musk"), and a
`prompt`, a short noun phrase SAM 3.1 segments directly ("drink can"). Products that share a phrase
form a family: SAM is asked for the phrase once ("book"), and which member an item is gets decided afterwards by
reading it (reader.py) and asking the second model (verify.py). A family's head stands in for "one of these, not
yet known which"; it has no price and is never billed.

Entries under `belongings` are things shoppers bring with them (a phone, a wallet): they are tracked so they are never
mistaken for stock, and never billed.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

PERSON = "person"
HAND = "hand"
#: catches anything in a hand, whatever it is, so an item outside the catalogue is still seen and flagged
HELD = "item held in hand"
DEFAULT_ZONES = ["shelf", "refrigerator", "display rack"]


@dataclass
class Product:
    id: str
    name: str
    prompt: str
    price: float = 0.0
    billable: bool = True
    author: str = ""  # or maker: a second line of text that helps tell family members apart
    cues: str = ""  # what it looks like, for the second model
    category: str = ""  # "Books"
    section: str = ""  # the subcategory within it: "Biography"
    meta: dict = field(default_factory=dict)  # anything else worth keeping: format, publisher, language
    family: str | None = None  # id of the family head, for members
    head: bool = False  # the stand-in for "some member of this family"


@dataclass
class Catalog:
    products: list[Product] = field(default_factory=list)
    currency: str = "AED"
    zones: list[str] = field(default_factory=lambda: list(DEFAULT_ZONES))
    name: str = "Shop"
    frame_width: int = 960  # longer side of the frames sent for analysis; reading titles needs more than outlines do

    def __post_init__(self):
        self.by_id: dict[str, Product] = {p.id: p for p in self.products}
        self.by_prompt: dict[str, Product] = {}
        self.members: dict[str, list[Product]] = {}  # family head id -> its products
        groups: dict[str, list[Product]] = {}
        for p in self.products:
            groups.setdefault(p.prompt, []).append(p)
        for prompt, group in groups.items():
            if len(group) == 1:
                self.by_prompt[prompt] = group[0]
                continue
            head = Product(f"any:{prompt}", prompt.capitalize(), prompt, billable=False, head=True)
            self.by_id[head.id] = head
            self.by_prompt[prompt] = head
            self.members[head.id] = group
            for p in group:
                p.family = head.id

    def family_of(self, product_id: str | None) -> str | None:
        product = self.by_id.get(product_id)
        if product is None:
            return None
        return product.id if product.head else product.family

    def same_kind(self, a: str | None, b: str | None) -> bool:
        """Could two labels be the same physical item? Yes if either is unknown, they are equal, or one is a
        family ("a book") and the other a member of it ("Deep Work")."""
        if a is None or b is None or a == b:
            return True
        pa, pb = self.by_id.get(a), self.by_id.get(b)
        if pa is None or pb is None:
            return False
        return (pa.head and pb.family == pa.id) or (pb.head and pa.family == pb.id)

    @classmethod
    def from_dict(cls, data: dict) -> "Catalog":
        products, seen = [], set()
        for i, raw in enumerate(data.get("products", [])):
            prompt = str(raw.get("prompt") or raw.get("name") or "").strip().lower()[:80]
            if not prompt:
                continue
            name = str(raw.get("name") or prompt).strip()[:80]
            pid = str(raw.get("id") or re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or f"p{i}")
            while pid in seen:
                pid += "-2"
            seen.add(pid)
            products.append(
                Product(
                    pid, name, prompt, float(raw.get("price") or 0),
                    author=str(raw.get("author") or "")[:80], cues=str(raw.get("cues") or "")[:160],
                    category=str(raw.get("category") or "").strip()[:40], section=str(raw.get("section") or "").strip()[:40],
                    meta={str(k)[:24]: str(v)[:80] for k, v in (raw.get("meta") or {}).items()} if isinstance(raw.get("meta"), dict) else {},
                )
            )
        for i, raw in enumerate(data.get("belongings", [])):
            prompt = str(raw if isinstance(raw, str) else raw.get("prompt", "")).strip().lower()[:80]
            if prompt:
                products.append(Product(f"own{i}", prompt.capitalize(), prompt, 0.0, billable=False))
        zones = [str(z).strip().lower()[:80] for z in data.get("zones", DEFAULT_ZONES) if str(z).strip()]
        width = min(1920, max(640, int(data.get("frame_width") or 960)))
        return cls(products, str(data.get("currency", "AED"))[:6], zones, str(data.get("name") or "Shop")[:40], width)

    @classmethod
    def load(cls, path: Path) -> "Catalog":
        return cls.from_dict(json.loads(Path(path).read_text()))

    def to_dict(self) -> dict:
        def row(p: Product) -> dict:
            out = {"id": p.id, "name": p.name, "prompt": p.prompt, "price": p.price}
            if p.author:
                out["author"] = p.author
            if p.cues:
                out["cues"] = p.cues
            if p.category:
                out["category"] = p.category
            if p.section:
                out["section"] = p.section
            if p.meta:
                out["meta"] = p.meta
            return out

        return {
            "name": self.name,
            "currency": self.currency,
            "frame_width": self.frame_width,
            "products": [row(p) for p in self.products if p.billable],
            "belongings": [p.prompt for p in self.products if not p.billable],
            "zones": self.zones,
        }

    def frame_prompts(self) -> list[str]:
        """Asked on every frame: one phrase per product or family."""
        return [PERSON, HAND, HELD, *self.by_prompt]

    def zone_prompts(self) -> list[str]:
        """Asked every few seconds: where the shelves are."""
        return list(self.zones)

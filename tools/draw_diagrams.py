"""
Draws the README's diagrams as SVG files in docs/diagrams/, in the app's own look (web/styles.css): paper and bottle
green, Newsreader for titles, Instrument Sans for everything else. The two fonts are embedded, so the pictures look
the same on GitHub as here.

    uv run python tools/draw_diagrams.py
"""
from __future__ import annotations

import base64
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "diagrams"

BG, SURFACE, INK, MUTED, LINE, WASH = "#f8faf8", "#ffffff", "#141a16", "#535a56", "#dbdfdc", "#eef2ef"
GREEN, DEEP, GWASH, HELD, HWASH, EDGE = "#1a5439", "#0e3f2a", "#e1f4e9", "#804d0c", "#fdf0dc", "#929b96"
CHART = "#25855b"
ON_GREEN, ON_GREEN_MUTED = "#ffffff", "#bfe0cf"


def wrap(s: str, width: int) -> list[str]:
    lines, line = [], ""
    for word in s.split():
        if line and len(line) + 1 + len(word) > width:
            lines.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    return lines + [line]


def fonts() -> str:
    out = []
    for family, name, weight in (("NR", "newsreader-latin-wght-normal.woff2", "200 800"), ("IS", "instrument-sans-latin-wght-normal.woff2", "400 700")):
        data = base64.b64encode((ROOT / "web" / "fonts" / name).read_bytes()).decode()
        out.append(f'@font-face{{font-family:"{family}";src:url(data:font/woff2;base64,{data}) format("woff2");font-weight:{weight};}}')
    return "".join(out)


class Svg:
    def __init__(self, w: int, h: int, title: str, subtitle: str):
        self.w, self.h, self.parts = w, h, []
        self.rect(0.5, 0.5, w - 1, h - 1, BG, LINE, 16)
        self.text(40, 62, title, 28, 500, INK, serif=True)
        self.text(40, 92, subtitle, 15, 400, MUTED)

    def add(self, s: str) -> None:
        self.parts.append(s)

    def rect(self, x, y, w, h, fill, stroke=None, r=8, dash=None, sw=1):
        s = f' stroke="{stroke}" stroke-width="{sw}"' if stroke else ""
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}"{s}{d}/>')

    def text(self, x, y, s, size=13, weight=400, fill=INK, anchor="start", serif=False, mono=False):
        family = "NR, Georgia, serif" if serif else ("ui-monospace, SFMono-Regular, Menlo, monospace" if mono else "IS, system-ui, -apple-system, Segoe UI, sans-serif")
        self.add(f'<text x="{x}" y="{y}" font-family="{family}" font-size="{size}" font-weight="{weight}" fill="{fill}" text-anchor="{anchor}">{escape(s)}</text>')

    def group(self, x, y, w, h, label, fill=WASH, stroke=LINE, dash=None, color=MUTED):
        self.rect(x, y, w, h, fill, stroke, 12, dash)
        self.text(x + 20, y + 26, label, 13, 600, color)

    def node(self, x, y, w, h, title, lines=(), mono=None, kind="plain"):
        fill, stroke, tc, sc = {
            "plain": (SURFACE, LINE, INK, MUTED),
            "core": (GREEN, GREEN, ON_GREEN, ON_GREEN_MUTED),
            "money": (SURFACE, GREEN, INK, MUTED),
            "infra": (WASH, LINE, INK, MUTED),
        }[kind]
        self.rect(x, y, w, h, fill, stroke, 8, sw=1.5 if kind == "money" else 1)
        ty = y + 30
        self.text(x + 16, ty, title, 15, 600, tc)
        if mono:
            ty += 19
            self.text(x + 16, ty, mono, 11.5, 400, sc, mono=True)
        for line in lines:
            ty += 18
            self.text(x + 16, ty, line, 13, 400, sc)

    def chip(self, x, y, s, fill=WASH, color=MUTED, anchor="start") -> float:
        w = len(s) * 6.4 + 20
        if anchor == "end":
            x -= w
        self.rect(x, y, w, 22, fill, None, 11)
        self.text(x + w / 2, y + 15, s, 11.5, 500, color, "middle")
        return w

    def pill(self, cx, cy, s, color=MUTED, border=LINE):
        w = len(s) * 6.5 + 20
        self.rect(cx - w / 2, cy - 11, w, 22, BG, border, 11)
        self.text(cx, cy + 4, s, 11.5, 500, color, "middle")

    def path(self, d, color=EDGE, dash=None, head=True, width=1.5):
        marker = f' marker-end="url(#{"g" if color == GREEN else ("h" if color == HELD else "e")})"' if head else ""
        dd = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}" stroke-linejoin="round"{dd}{marker}/>')

    def badge(self, cx, cy, n, fill=GWASH, color=GREEN):
        self.add(f'<circle cx="{cx}" cy="{cy}" r="13" fill="{fill}"/>')
        self.text(cx, cy + 4.5, str(n), 12.5, 700, color, "middle")

    def chevron(self, cx, cy, color=EDGE):
        self.add(f'<path d="M{cx - 4} {cy - 8} L{cx + 4} {cy} L{cx - 4} {cy + 8}" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>')

    def save(self, name: str) -> None:
        defs = "".join(
            f'<marker id="{i}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="{c}"/></marker>'
            for i, c in (("e", EDGE), ("g", GREEN), ("h", HELD))
        )
        body = "\n".join(self.parts)
        svg = (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{self.h}" viewBox="0 0 {self.w} {self.h}">\n'
            f"<style>{fonts()}</style>\n<defs>{defs}</defs>\n{body}\n</svg>\n"
        )
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / name).write_text(svg)


def architecture() -> None:
    s = Svg(1280, 870, "How it fits together", "Cameras send frames to one web service on Render. Vision runs on Modal GPUs, faces are matched in Neon, and money moves through PayPal.")

    s.group(40, 118, 1200, 138, "In the shop")
    s.node(80, 160, 250, 76, "Gate camera", ["browser or phone, at the entrance"])
    s.node(400, 160, 250, 76, "Floor camera", ["browser or phone, at the shelf"])
    s.node(720, 160, 250, 76, "Staff browser", ["floor, catalogue, dashboard"])
    for i, line in enumerate(("The browser only sends frames", "and draws the answers. All", "analysis runs on the server", "and on Modal.")):
        s.text(1020, 176 + i * 18, line, 13, 400, MUTED)

    s.group(40, 296, 1200, 180, "Render · one Python web service", GWASH, "#c4e3d2", color=GREEN)
    s.node(80, 344, 250, 80, "Gate", ["recognise or register"], mono="checkout/gate.py")
    s.node(400, 344, 250, 80, "Visit pipeline", ["tracks, picks, carts, checks"], mono="pipeline.py · engine.py", kind="core")
    s.node(720, 344, 250, 80, "Staff screens and API", ["sign-in, dashboard, refunds"], mono="checkout/server.py")
    s.node(1020, 316, 200, 70, "On-server models", ["YuNet + SFace · RapidOCR"], kind="infra")
    s.node(1020, 396, 200, 70, "SQLite on a 1 GB disk", ["visits, ledger, catalogues"], kind="infra")

    s.group(40, 610, 1200, 210, "Cloud services")
    s.group(590, 648, 410, 152, "Modal · GPU", SURFACE)
    s.node(80, 690, 210, 90, "Neon Postgres", ["pgvector face index", "HNSW, cosine"])
    s.node(320, 690, 240, 90, "PayPal sandbox", ["Vault, Orders v2,", "Invoicing, Refunds"], kind="money")
    s.node(610, 690, 180, 90, "SAM 3.1 + DINOv2", ["L40S, phrases batched"])
    s.node(810, 690, 170, 90, "DeepSeek vision", ["looks, final check"])
    s.node(1030, 690, 190, 90, "Resend", ["invoice email"])

    for x, label in ((205, "1 frame/s · HTTPS"), (525, "frames · WebSocket"), (845, "JWT sign-in")):
        s.path(f"M{x} 236 V342")
        s.pill(x, 276, label)

    s.path("M170 424 V688")
    s.path("M300 424 V510 H400 V688", GREEN)
    s.path("M480 424 V688", GREEN)
    s.path("M570 424 V550 H700 V688")
    s.path("M605 424 V520 H895 V688")
    s.path("M640 424 V490 H1125 V688")
    s.pill(170, 570, "nearest face")
    s.pill(350, 510, "save method", GREEN, GREEN)
    s.pill(480, 590, "charge · invoice", GREEN, GREEN)
    s.pill(635, 550, "frames + phrases")
    s.pill(760, 520, "pictures + bill")
    s.pill(880, 490, "invoice email")

    s.path("M40 846 H76", GREEN, head=False, width=2)
    s.text(86, 850, "money moves", 12.5, 500, MUTED)
    s.path("M200 846 H236", EDGE, head=False, width=2)
    s.text(246, 850, "data", 12.5, 500, MUTED)
    s.save("architecture.svg")


def visit() -> None:
    s = Svg(1280, 600, "A visit, end to end", "From the gate to the invoice email. Faces, frames and pictures are read by the services named under each step.")
    phases = (
        ("Entry", "Gate camera", [
            ("Face found and read", "YuNet + SFace"),
            ("Nearest face looked up", "Neon pgvector"),
            ("New face: name, email, method", "PayPal vault"),
            ("Visit opened with the gate picture", "Ledger"),
        ]),
        ("In store", "Floor camera, 3 to 4 frames a second", [
            ("Shoppers, hands, products outlined", "SAM 3.1"),
            ("Picks and returns from the masks", "Engine"),
            ("Each pick looked at again", "DeepSeek vision"),
            ("Face matched, cart linked", "SFace"),
        ]),
        ("Checkout", "Out of view for 3 seconds", [
            ("Bill checked against pictures", "DeepSeek vision"),
            ("One payment for the kept lines", "PayPal Orders v2"),
            ("Invoice written, email sent", "PayPal + Resend"),
            ("Steps and proof written down", "Ledger"),
        ]),
    )
    n = 1
    for i, (name, where, steps) in enumerate(phases):
        x = 40 + i * 410
        last = i == 2
        s.rect(x, 124, 380, 440, SURFACE, LINE, 12)
        s.rect(x, 124, 380, 72, GREEN if last else GWASH, None, 12)
        s.rect(x, 172, 380, 24, GREEN if last else GWASH, None, 0)
        s.text(x + 24, 158, name, 22, 500, ON_GREEN if last else DEEP, serif=True)
        s.text(x + 24, 182, where, 12.5, 500, ON_GREEN_MUTED if last else GREEN)
        for j, (title, chip) in enumerate(steps):
            y = 226 + j * 84
            s.badge(x + 37, y + 14, n)
            s.text(x + 62, y + 19, title, 14.5, 600, INK)
            s.chip(x + 62, y + 32, chip, GWASH if "PayPal" in chip else WASH, GREEN if "PayPal" in chip else MUTED)
            if j < len(steps) - 1:
                s.add(f'<line x1="{x + 37}" y1="{y + 30}" x2="{x + 37}" y2="{y + 82}" stroke="{LINE}" stroke-width="2"/>')
            n += 1
        if i < 2:
            s.chevron(x + 395, 344)
    s.save("visit.svg")


def verification() -> None:
    s = Svg(1280, 620, "How a bill is verified", "A line is charged only when it passes every stage. Anything doubtful is held for review, never guessed.")
    cards = (
        ("Gate", "Who is shopping", ["Face matched at gate", "Score 0.45, lead 0.1", "PayPal method on file"], "No method, no entry"),
        ("Floor", "What was taken", ["SAM masks each frame", "Leaves the shelf in a hand", "Most contact wins"], "No pick, no line"),
        ("Floor", "Which product", ["SAM phrase first", "Up to 3 vision looks", "Likeness 0.55 to the catalogue pictures"], None),
        ("Door", "Still with them", ["Seen with them in their last 6 seconds", "Or seen on them at the door"], None),
        ("Door", "Final check", ["Bill + 5 pictures", "Kept, put back or unclear", "Counts each product"], None),
        ("PayPal", "Charged", ["One Orders v2 payment", "Never charged twice", "Invoice, email, proof"], None),
    )
    for i, (where, title, items, note) in enumerate(cards):
        x, y = 40 + i * 204, 128
        core = i == 5
        s.rect(x, y, 180, 244, GREEN if core else SURFACE, GREEN if core else LINE, 10)
        s.badge(x + 29, y + 30, i + 1, "#2c6b4f" if core else GWASH, ON_GREEN if core else GREEN)
        s.text(x + 50, y + 35, where, 12, 600, ON_GREEN_MUTED if core else MUTED)
        s.text(x + 18, y + 78, title, 17, 600, ON_GREEN if core else INK)
        ty = y + 110
        for item in items:
            s.add(f'<circle cx="{x + 21}" cy="{ty - 4.5}" r="2.5" fill="{ON_GREEN_MUTED if core else CHART}"/>')
            for line in wrap(item, 21):
                s.text(x + 30, ty, line, 12.5, 400, ON_GREEN_MUTED if core else MUTED)
                ty += 19
            ty += 7
        if i < 5:
            s.chevron(x + 192, y + 122)
        if note:
            s.path(f"M{x + 90} {y + 244} V{y + 286}", EDGE, dash="3 4", head=False)
            s.text(x + 90, y + 306, note, 12.5, 500, MUTED, "middle")
    s.rect(448, 470, 588, 98, HWASH, "#e7c897", 12)
    s.text(472, 506, "Needs review", 17, 600, HELD)
    s.text(472, 532, "Held lines are never charged on a guess. Staff see the pictures", 13, 400, MUTED)
    s.text(472, 551, "behind the bill, then press Charge or mark it Not taken.", 13, 400, MUTED)
    for i, label in ((2, "name disputed"), (3, "not linked at exit"), (4, "unclear")):
        cx = 40 + i * 204 + 90
        s.path(f"M{cx} 372 V468", HELD, dash="4 4")
        s.pill(cx, 420, label, HELD, "#e7c897")
    s.text(1170, 412, "Refunds: one line or", 12.5, 500, GREEN, "middle")
    s.text(1170, 430, "the whole bill, once", 12.5, 500, GREEN, "middle")
    s.save("verification.svg")


def catalogue() -> None:
    s = Svg(1280, 330, "Adding a product", "Hold it up to a camera and turn it. The form fills itself in; only the price is typed.")
    steps = (
        ("Hold it up", ["Turn it slowly in", "front of any camera"]),
        ("SAM outlines it", ["About every 0.7 s,", "up to ten clear views"]),
        ("Best four views", ["The most different,", "by DINOv2 vectors"]),
        ("Model names it", ["Name, maker, category,", "section, how it looks"]),
        ("Phrase test", ["The phrase SAM", "outlines it by"]),
        ("Type the price", ["Saved to the shop file,", "live on the next Start"]),
    )
    s.add(f'<line x1="70" y1="156" x2="1210" y2="156" stroke="{LINE}" stroke-width="2"/>')
    for i, (title, lines) in enumerate(steps):
        x = 40 + i * 204
        core = i == 5
        s.badge(x + 30, 156, i + 1, GREEN if core else GWASH, ON_GREEN if core else GREEN)
        s.rect(x, 186, 180, 104, GREEN if core else SURFACE, GREEN if core else LINE, 10)
        s.text(x + 18, 218, title, 15.5, 600, ON_GREEN if core else INK)
        for j, line in enumerate(lines):
            s.text(x + 18, 246 + j * 20, line, 13, 400, ON_GREEN_MUTED if core else MUTED)
    s.save("catalogue.svg")


def deployment() -> None:
    s = Svg(1280, 640, "Where it runs", "Two deploys from one repository: a Render Blueprint for the web service and a Modal app for the GPU.")

    s.rect(40, 262, 220, 150, SURFACE, LINE, 12)
    s.text(60, 296, "GitHub repository", 15, 600, INK)
    for j, f in enumerate(("render.yaml", "modal_app/", "checkout/, web/", "models/*.onnx")):
        s.text(60, 324 + j * 20, f, 12, 400, MUTED, mono=True)

    s.rect(400, 128, 440, 222, GWASH, "#c4e3d2", 12)
    s.text(424, 162, "Render", 22, 500, DEEP, serif=True)
    s.text(816, 160, "Virginia", 12.5, 600, GREEN, "end")
    s.node(424, 180, 392, 66, "Web service: paypal-hackathon", ["uv sync from the lockfile, then uvicorn"])
    x = 424
    for chip in ("Starter, 512 MB", "1 instance", "/healthz"):
        x += s.chip(x, 262, chip, SURFACE, GREEN) + 8
    s.node(424, 296, 392, 40, "Disk at /var/data, 1 GB", kind="infra")

    s.rect(400, 390, 440, 222, SURFACE, LINE, 12)
    s.text(424, 424, "Modal", 22, 500, INK, serif=True)
    s.text(816, 422, "GPU", 12.5, 600, MUTED, "end")
    s.node(424, 442, 392, 66, "App sam3-supermarket", ["SAM 3.1 + DINOv2, phrases batched on one L40S"])
    x = 424
    for chip in ("1 container", "off after 5 idle min", "weights on a volume"):
        x += s.chip(x, 524, chip) + 8
    s.node(424, 558, 392, 40, "DeepSeek vision endpoint", kind="infra")

    s.node(940, 128, 300, 120, "Neon Postgres", ["pgvector, HNSW on cosine", "embeddings and profile ids only"], mono="checkout_faces_render")
    s.node(940, 280, 300, 100, "PayPal sandbox", ["vault, Orders v2, invoices, refunds"], mono="api-m.sandbox.paypal.com", kind="money")
    s.node(940, 412, 300, 80, "Resend", ["invoice email"])

    s.path("M260 310 H320 V215 H398")
    s.text(330, 205, "Blueprint deploy", 12, 500, MUTED)
    s.path("M260 364 H320 V475 H398")
    s.text(330, 497, "modal deploy", 12, 500, MUTED)
    s.path("M620 350 V388")
    s.pill(690, 369, "x-api-key")
    s.path("M840 188 H938")
    s.text(850, 180, "DATABASE_URL", 11.5, 500, MUTED)
    s.path("M840 300 H880 V330 H938", GREEN)
    s.path("M840 322 H866 V452 H938")
    s.save("deployment.svg")


def walkthrough() -> None:
    s = Svg(1280, 640, "Pick it up, walk out, get billed", "The cameras and models decide what each shopper carried out. PayPal holds the payment method, takes one charge at the door and writes the invoice.")
    cols = (
        ("Register", "Faces the gate camera and picks PayPal or a card",
         "Face found and embedded", ["YuNet + SFace", "Neon pgvector"],
         "Vault saves the method", ["/v3/vault/setup-tokens", "/v3/vault/payment-tokens"], True),
        ("Walk in", "The gate opens for them by name",
         "Two matching looks open a visit", ["SFace, score 0.45"],
         "Method on file shown", ["saved vault token"], True),
        ("Pick up", "Takes items off the shelf, puts some back",
         "Masks show who took what, models name it", ["SAM 3.1", "DeepSeek", "DINOv2"],
         "Nothing charged while shopping", [], False),
        ("Walk out", "Leaves with what they want. No till, no scan",
         "Final check per line: kept, put back, unclear", ["DeepSeek vision"],
         "One charge at the door", ["/v2/checkout/orders", "PayPal-Request-Id"], True),
        ("Afterwards", "Gets the PayPal invoice in their inbox",
         "Every step and picture kept as proof", ["Ledger"],
         "Invoice marked paid, refunds by line", ["/v2/invoicing/invoices", "/v2/payments/…/refund"], True),
    )
    lanes = (("Shopper", "what they do", 170, 96), ("AI senses", "cameras and models", 286, 140), ("PayPal acts", "the money", 446, 150))
    for name, sub, y, h in lanes:
        s.rect(40, y, 150, h, GREEN if name == "PayPal acts" else WASH, None, 10)
        s.text(60, y + 34, name, 19, 500, ON_GREEN if name == "PayPal acts" else INK, serif=True)
        s.text(60, y + 56, sub, 12.5, 500, ON_GREEN_MUTED if name == "PayPal acts" else MUTED)
    for i, (when, does, ai, models, pp, api, money) in enumerate(cols):
        x = 210 + i * 208
        s.badge(x + 14, 140, i + 1, GREEN, ON_GREEN)
        s.text(x + 36, 145, when, 15, 600, INK)
        if i < 4:
            s.add(f'<line x1="{x + 150}" y1="140" x2="{x + 214}" y2="140" stroke="{LINE}" stroke-width="2"/>')
        s.rect(x, 170, 196, 96, SURFACE, LINE, 10)
        ty = 198
        for line in wrap(does, 25):
            s.text(x + 16, ty, line, 13.5, 500, INK)
            ty += 19
        s.rect(x, 286, 196, 140, GWASH, "#c4e3d2", 10)
        ty = 314
        for line in wrap(ai, 24):
            s.text(x + 16, ty, line, 13.5, 600, DEEP)
            ty += 19
        cx, cy = x + 16, 370
        for m in models:
            w = len(m) * 6.4 + 20
            if cx + w > x + 186:
                cx, cy = x + 16, cy + 28
            s.chip(cx, cy, m, SURFACE, GREEN)
            cx += w + 6
        s.rect(x, 446, 196, 150, SURFACE if money else WASH, GREEN if money else LINE, 10, sw=1.5 if money else 1, dash=None if money else "4 4")
        ty = 474
        for line in wrap(pp, 24):
            s.text(x + 16, ty, line, 13.5, 600, INK if money else MUTED)
            ty += 19
        for line in api:
            ty += 4
            s.text(x + 16, ty + 14, line, 11, 400, GREEN, mono=True)
            ty += 18
        for y0, y1 in ((266, 284), (426, 444)):
            s.path(f"M{x + 98} {y0} V{y1}", GREEN if y0 == 426 and money else EDGE, width=1.5)
    s.save("walkthrough.svg")


def gpu() -> None:
    s = Svg(1280, 700, "Inside the GPU: SAM 3.1 on Modal", "Every analysed frame is one request. The image goes through the backbone once, and every phrase the shop asks for is detected in one batch.")

    s.node(40, 150, 200, 180, "POST /v1/segment", ["one JPEG frame", "person, hand,", "item held in hand,", "every catalogue phrase", "optional example boxes"], mono="x-api-key")
    s.node(40, 380, 200, 100, "POST /v1/embed", ["up to 16 pictures"], mono="x-api-key")

    s.group(270, 118, 740, 390, "Modal container · one L40S GPU", SURFACE, "#c4e3d2", color=GREEN)
    s.text(990, 144, "one frame at a time · bf16", 12, 500, MUTED, "end")
    steps = (
        ("Backbone", ["once per frame,", "shared by every", "phrase"], "plain"),
        ("Text encoder", ["phrase vectors,", "cached: the shop", "asks the same", "every frame"], "infra"),
        ("Detect + mask", ["every phrase in", "one batched pass"], "core"),
        ("Example boxes", ["visual prompt,", "one pass each,", "about 20 ms"], "dashed"),
        ("Outlines", ["kept over the", "threshold, sent", "as polygons"], "plain"),
    )
    for i, (title, lines, kind) in enumerate(steps):
        x = 282 + i * 146
        if kind == "dashed":
            s.rect(x, 166, 136, 150, SURFACE, EDGE, 8, dash="4 4")
            s.text(x + 12, 196, title, 14, 600, INK)
            for j, line in enumerate(lines):
                s.text(x + 12, 220 + j * 18, line, 12, 400, MUTED)
        else:
            fill, stroke, tc, sc = {"plain": (SURFACE, LINE, INK, MUTED), "infra": (WASH, LINE, INK, MUTED), "core": (GREEN, GREEN, ON_GREEN, ON_GREEN_MUTED)}[kind]
            s.rect(x, 166, 136, 150, fill, stroke, 8)
            s.text(x + 12, 196, title, 14, 600, tc)
            for j, line in enumerate(lines):
                s.text(x + 12, 220 + j * 18, line, 12, 400, sc)
        if i < 4:
            s.chevron(x + 141, 241)
    s.path("M240 240 H280")
    s.path("M1010 240 H1038")

    s.text(290, 364, "Picture vectors", 13, 600, MUTED)
    vec = (("DINOv2 ViT-S/14", ["small vision model,", "same container"]), ("Class + mean patch", ["token summed, then", "L2-normalised"]), ("384 numbers", ["the cosine of two", "says how alike"]))
    for i, (title, lines) in enumerate(vec):
        x = 290 + i * 240
        s.rect(x, 378, 224, 104, WASH, LINE, 8)
        s.text(x + 14, 406, title, 14, 600, INK)
        for j, line in enumerate(lines):
            s.text(x + 14, 430 + j * 18, line, 12, 400, MUTED)
        if i < 2:
            s.chevron(x + 232, 430)
    s.path("M240 430 H288")
    s.path("M1010 430 H1038")

    s.node(1040, 150, 200, 180, "Answer", ["for each find: label,", "box, score, outline", "", "about 139 ms for", "eight phrases"], mono="detections[], ms")
    s.node(1040, 380, 200, 100, "Vectors", ["likeness check,", "best four views"], mono="vectors[], dim 384")

    s.text(40, 556, "Running it", 15, 600, INK)
    x, y = 40, 572
    for chip in ("weights on a Modal volume", "key from a Modal secret", "1 container at most", "off after 5 idle minutes", "woken by the floor page"):
        w = len(chip) * 6.4 + 20
        if x + w > 640:
            x, y = 40, y + 30
        s.chip(x, y, chip, WASH, MUTED)
        x += w + 8

    s.text(700, 556, "One frame, eight phrases", 15, 600, INK)
    bars = (("A10G, one by one", 575, False), ("A10G, batched", 367, False), ("L40S, one by one", 290, False), ("L40S, batched (deployed)", 139, True))
    for i, (label, ms, best) in enumerate(bars):
        y = 576 + i * 28
        s.text(700, y + 14, label, 12, 500, INK if best else MUTED)
        w = ms / 575 * 300
        s.rect(870, y, w, 18, GREEN if best else "#c9d3cd", None, 4)
        s.text(870 + w + 8, y + 14, f"{ms} ms", 12, 600, GREEN if best else MUTED)
    s.save("gpu.svg")


if __name__ == "__main__":
    for draw in (walkthrough, architecture, visit, verification, gpu, catalogue, deployment):
        draw()
    print("\n".join(sorted(p.name for p in OUT.glob("*.svg"))))

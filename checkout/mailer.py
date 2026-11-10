"""
The invoice by email, through Resend, once a walk-out payment has gone through: a thank-you, the things bought with
their category and section, the total, the link to the invoice on the payment provider's side, and a goodbye.

With the default sender (`onboarding@resend.dev`) Resend only delivers to the address that owns the Resend account;
any other recipient is refused with a 403 until a domain is verified. That refusal is reported, not hidden.
"""
from __future__ import annotations

import html
import re
from datetime import datetime

import httpx

from .sam_client import load_env

DEFAULT_FROM = "Walk-out checkout <onboarding@resend.dev>"
EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
#: addresses that must never get automated mail from this machine (the owner's standing rule)
NEVER = {"abdullahweb24bd@gmail.com"}


def valid_email(address: str) -> bool:
    address = address.strip().lower()
    return bool(EMAIL.match(address)) and len(address) <= 120 and address not in NEVER


class Mailer:
    def __init__(self, env: dict | None = None):
        env = load_env() if env is None else env
        self.key = env.get("RESEND_API_KEY", "")
        self.sender = env.get("RESEND_FROM", DEFAULT_FROM)
        self.enabled = bool(self.key)
        self.http = httpx.AsyncClient(timeout=20, headers={"Authorization": f"Bearer {self.key}"}) if self.enabled else None

    async def close(self) -> None:
        if self.http:
            await self.http.aclose()

    async def receipt(self, to: str, name: str, lines: list[tuple], payment: dict, key: str) -> dict:
        """`lines` are (item, price) or (item, price, "Category · Section"). Returns {status: sent | failed | skipped, to, id, message}."""
        lines = [(line[0], line[1], line[2] if len(line) > 2 else "") for line in lines]
        to = to.strip()
        if not self.enabled:
            return {"status": "skipped", "to": to, "id": None, "message": "RESEND_API_KEY is not set"}
        if not valid_email(to):
            return {"status": "skipped", "to": to, "id": None, "message": "not an address this app may write to"}
        currency = payment["currency"].upper()
        when = datetime.now().strftime("%d %B %Y, %H:%M")
        invoice = payment.get("invoice") or {}
        number = invoice.get("number") or "INV-" + datetime.now().strftime("%Y%m%d") + "-" + (payment.get("id") or "000000")[-6:].upper()
        card = payment.get("method") or f"{(payment.get('brand') or 'card').capitalize()} ending {payment.get('last4') or '····'}"
        provider = "PayPal" if invoice else "the payment provider"
        link_text = "View your PayPal invoice" if invoice else "View your receipt"
        cell = "padding:10px 0;border-bottom:1px solid #e6e3de"
        small = "display:block;color:#6b6862;font-size:13px"
        rows = "".join(
            f'<tr><td style="{cell}">{html.escape(item)}' + (f'<span style="{small}">{html.escape(where)}</span>' if where else "") + f'</td><td style="{cell};text-align:right;white-space:nowrap;vertical-align:top">{currency} {price:.2f}</td></tr>'
            for item, price, where in lines
        )
        link = f'<p style="margin:16px 0 0;font-size:14px"><a href="{html.escape(payment["receipt_url"])}" style="color:#1f5c45;font-weight:600">{link_text}</a></p>' if payment.get("receipt_url") else ""
        muted = "color:#6b6862;font-size:14px"
        body = f"""<div style="font-family:-apple-system,Segoe UI,Roboto,sans-serif;color:#262421;max-width:480px;margin:0 auto;padding:32px 16px;font-size:16px;line-height:1.5">
<h1 style="font-family:Georgia,serif;font-weight:500;font-size:26px;margin:0 0 8px">Thank you for purchasing with us</h1>
<p style="margin:0 0 24px">Hello {html.escape(name)}, these are the things you bought today.</p>
<p style="margin:0 0 4px;{muted}">Invoice {number} · {when}</p>
<table style="width:100%;border-collapse:collapse">{rows}
<tr><td style="padding:12px 0;font-weight:600">Total paid</td><td style="padding:12px 0;text-align:right;font-weight:600;white-space:nowrap">{currency} {payment['amount']:.2f}</td></tr></table>
<p style="margin:8px 0 0;{muted}">Paid with {html.escape(card)} through {provider}.</p>{link}
<p style="margin:32px 0 0">Looking forward to meeting you again.</p>
<p style="margin:24px 0 0;{muted}">This invoice comes from a demo running on a payment sandbox. No money moved.</p></div>"""
        text = f"Thank you for purchasing with us\n\nHello {name}, these are the things you bought today.\n\nInvoice {number} · {when}\n\n"
        text += "\n".join(f"{item}{f' ({where})' if where else ''}  {currency} {price:.2f}" for item, price, where in lines)
        text += f"\n\nTotal paid  {currency} {payment['amount']:.2f}\nPaid with {card} through {provider}.\n"
        text += (f"{link_text}: {payment['receipt_url']}\n" if payment.get("receipt_url") else "") + "\nLooking forward to meeting you again.\n\nThis invoice comes from a demo running on a payment sandbox. No money moved.\n"
        subject = f"Thank you for your purchase: invoice {number}"
        try:
            r = await self.http.post(
                "https://api.resend.com/emails",
                headers={"Idempotency-Key": key},
                json={"from": self.sender, "to": [to], "subject": subject, "html": body, "text": text},
            )
        except httpx.HTTPError as e:
            return {"status": "failed", "to": to, "id": None, "message": str(e) or type(e).__name__}
        if r.status_code >= 300:
            try:
                message = r.json().get("message", r.text)
            except ValueError:
                message = r.text
            return {"status": "failed", "to": to, "id": None, "message": str(message)[:240]}
        return {"status": "sent", "to": to, "id": r.json().get("id"), "message": ""}

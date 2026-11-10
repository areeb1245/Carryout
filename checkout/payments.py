"""
A payment method on file and the charge at the door, on PayPal's sandbox.

Registration saves a payment method in PayPal's vault: a setup token for a card or a PayPal account, which becomes a
payment token (`usage_type: MERCHANT`) once it is approved. A card is approved at once; a PayPal account is approved
by its owner in a PayPal window. Walking out creates one Orders v2 payment against that token with the shopper not
present, and `PayPal-Request-Id` makes a retry of the same walk-out charge once, not twice. After the charge an
invoice is written with the Invoicing API and marked as paid, so the shopper has a PayPal page to open.

Only the sandbox is ever called: the host is fixed below, and `PAYPAL_ENV` must say `sandbox` or payments stay off.
The Stripe version of this file (same `save_card` and `charge`) is kept on the `stripe` branch.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from datetime import date

import httpx

from .sam_client import load_env

API = "https://api-m.sandbox.paypal.com"  # never the live host
#: what PayPal can charge in (https://developer.paypal.com/api/rest/reference/currency-codes/); AED is not one of them
CURRENCIES = {"AUD", "BRL", "CAD", "CNY", "CZK", "DKK", "EUR", "HKD", "HUF", "ILS", "JPY", "MYR", "MXN", "TWD", "NZD", "NOK", "PHP", "PLN", "GBP", "SGD", "SEK", "CHF", "THB", "USD"}

#: PayPal's published sandbox card numbers, offered at the gate so nobody types a real one
_SANDBOX_CARDS = {
    "visa": {"number": "4012888888881881", "expiry": "2030-12", "security_code": "123"},
    "mastercard": {"number": "5425233430109903", "expiry": "2030-12", "security_code": "123"},
}
TEST_CARDS = {"visa": "Visa test card", "mastercard": "Mastercard test card"}
#: issues PayPal reports when the payment method itself said no, as opposed to the request being wrong
_DECLINES = {"INSTRUMENT_DECLINED", "PAYMENT_SOURCE_DECLINED_BY_PROCESSOR", "CARD_EXPIRED", "TRANSACTION_REFUSED", "PAYER_ACTION_REQUIRED", "PAYEE_BLOCKED_TRANSACTION", "PAYER_CANNOT_PAY", "PAYMENT_DENIED", "AGREEMENT_ALREADY_CANCELLED", "REDIRECT_PAYER_FOR_ALTERNATE_FUNDING"}


class PaymentsOff(RuntimeError):
    pass


class PayPalError(RuntimeError):
    def __init__(self, status: int, issue: str, message: str):
        super().__init__(message)
        self.status, self.issue = status, issue


@dataclass
class Card:
    """A saved payment method. For a PayPal account `brand` is "paypal", `last4` is empty and `payer` is its email."""

    customer: str
    payment_method: str  # the vault id
    brand: str
    last4: str
    payer: str = ""


def money(amount: float, currency: str) -> dict:
    return {"currency_code": currency.upper(), "value": f"{amount:.2f}"}


class Payments:
    name = "paypal"

    def __init__(self, env: dict | None = None, http: httpx.AsyncClient | None = None):
        env = load_env() if env is None else env
        self.client_id, self.secret = env.get("PAYPAL_CLIENT_ID", ""), env.get("PAYPAL_CLIENT_SECRET", "")
        mode = env.get("PAYPAL_ENV", "").strip().lower()
        self.enabled, self.problem = bool(self.client_id and self.secret), ""
        if not self.enabled:
            self.problem = "PAYPAL_CLIENT_ID and PAYPAL_CLIENT_SECRET are not set in .env; payments are off"
        elif mode != "sandbox":
            # this app charges on its own say-so, so it only ever talks to the sandbox
            self.enabled, self.problem = False, "PAYPAL_ENV is not 'sandbox'; this demo refuses to run against live PayPal"
        self.http = http or (httpx.AsyncClient(base_url=API, timeout=40) if self.enabled else None)
        self._token, self._token_until = "", 0.0
        self._lock = asyncio.Lock()

    async def close(self) -> None:
        if self.http:
            await self.http.aclose()

    # ------------------------------------------------------------------ plumbing

    async def _bearer(self) -> str:
        if not self.enabled:
            raise PaymentsOff(self.problem)
        async with self._lock:
            if time.time() >= self._token_until:
                r = await self.http.post("/v1/oauth2/token", auth=(self.client_id, self.secret), data={"grant_type": "client_credentials"})
                if r.status_code != 200:
                    raise PayPalError(r.status_code, "AUTH", "PayPal did not accept the sandbox client id and secret")
                body = r.json()
                self._token, self._token_until = body["access_token"], time.time() + int(body.get("expires_in", 300)) - 60
            return self._token

    async def _call(self, method: str, path: str, body: dict | None = None, key: str | None = None) -> dict:
        headers = {"Authorization": f"Bearer {await self._bearer()}", "Content-Type": "application/json", "Prefer": "return=representation"}
        if key:
            headers["PayPal-Request-Id"] = key[:108]
        r = await self.http.request(method, path, json=body, headers=headers)
        try:
            data = r.json() if r.content else {}
        except ValueError:
            data = {}
        if r.status_code >= 300:
            detail = (data.get("details") or [{}])[0]
            issue = detail.get("issue") or data.get("name") or f"HTTP_{r.status_code}"
            raise PayPalError(r.status_code, issue, detail.get("description") or data.get("message") or issue)
        return data

    @staticmethod
    def _method(token: dict) -> Card:
        source = token.get("payment_source") or {}
        customer = (token.get("customer") or {}).get("id", "")
        if "card" in source:
            return Card(customer, token["id"], str(source["card"].get("brand", "card")).lower(), source["card"].get("last_digits", ""))
        wallet = source.get("paypal") or {}
        return Card(customer, token["id"], "paypal", "", wallet.get("email_address", ""))

    # ------------------------------------------------------------------ saving a payment method

    async def save_card(self, name: str, test_card: str, shopper_id: str, email: str | None = None, details: dict | None = None) -> Card:
        """Vault a card for charging later. `test_card` picks one of PayPal's sandbox numbers; `details`
        ({number, expiry "YYYY-MM", security_code}) is what a shopper typed instead."""
        if details is None:
            if test_card not in _SANDBOX_CARDS:
                raise ValueError("not one of the offered test cards")
            details = _SANDBOX_CARDS[test_card]
        card = {"number": "".join(ch for ch in str(details.get("number", "")) if ch.isdigit()), "expiry": str(details.get("expiry", "")), "name": name[:120] or "Shopper"}
        if details.get("security_code"):
            card["security_code"] = str(details["security_code"])
        setup = await self._call("POST", "/v3/vault/setup-tokens", {"payment_source": {"card": card}}, key=f"setup-{shopper_id}")
        if setup.get("status") != "APPROVED":
            raise PayPalError(422, setup.get("status", "NOT_APPROVED"), "PayPal did not approve this card for saving")
        return await self.finish_wallet(setup["id"])

    async def begin_wallet(self, shopper_id: str, return_url: str, cancel_url: str, shop: str = "Walk-out checkout") -> dict:
        """Start saving a PayPal account. Returns {id, approve_url}: the shopper approves at that address, PayPal sends
        them back to `return_url`, and `finish_wallet(id)` then turns the approval into a payment token."""
        wallet = {
            "description": f"Pay at {shop} when you walk out"[:128],
            "usage_type": "MERCHANT",
            "usage_pattern": "IMMEDIATE",
            "customer_type": "CONSUMER",
            "permit_multiple_payment_tokens": False,
            "experience_context": {"return_url": return_url, "cancel_url": cancel_url, "brand_name": shop[:127], "shipping_preference": "NO_SHIPPING"},
        }
        setup = await self._call("POST", "/v3/vault/setup-tokens", {"payment_source": {"paypal": wallet}}, key=f"setup-{shopper_id}")
        approve = next((link["href"] for link in setup.get("links", []) if link.get("rel") == "approve"), None)
        if not approve:
            raise PayPalError(422, "NO_APPROVE_LINK", "PayPal gave no address to approve at")
        return {"id": setup["id"], "approve_url": approve}

    async def finish_wallet(self, setup_token: str) -> Card:
        token = await self._call("POST", "/v3/vault/payment-tokens", {"payment_source": {"token": {"id": setup_token, "type": "SETUP_TOKEN"}}}, key=f"token-{setup_token}")
        return self._method(token)

    # ------------------------------------------------------------------ the charge at the door

    async def charge(self, card: Card, amount: float, currency: str, description: str, key: str, metadata: dict | None = None, lines: list[tuple] | None = None) -> dict:
        """One payment with the shopper not present. `key` makes a retry of the same walk-out charge once, not twice.
        `lines` are (name, price, "Category · Section") and show up on the PayPal side.

        Returns {status: succeeded | declined | failed, id, amount, currency, receipt_url, message}.
        """
        currency = currency.upper()
        base = {"amount": round(amount, 2), "currency": currency.lower(), "id": None, "receipt_url": None}
        if currency not in CURRENCIES:
            return {**base, "status": "failed", "message": f"PayPal cannot charge in {currency}."}
        unit: dict = {"reference_id": key[:256], "description": description[:127], "amount": money(amount, currency)}
        if metadata and metadata.get("shopper"):
            unit["custom_id"] = str(metadata["shopper"])[:127]
        if lines and round(sum(line[1] for line in lines), 2) == round(amount, 2):
            unit["amount"]["breakdown"] = {"item_total": money(amount, currency)}
            unit["items"] = [{"name": line[0][:127], "quantity": "1", "unit_amount": money(line[1], currency), "category": "PHYSICAL_GOODS"} for line in lines]
        if card.brand == "paypal":
            source = {"paypal": {"vault_id": card.payment_method}}
        else:
            source = {"card": {"vault_id": card.payment_method, "stored_credential": {"payment_initiator": "MERCHANT", "payment_type": "UNSCHEDULED", "usage": "SUBSEQUENT"}}}
        try:
            order = await self._call("POST", "/v2/checkout/orders", {"intent": "CAPTURE", "purchase_units": [unit], "payment_source": source}, key=key)
        except PayPalError as e:
            declined = e.issue in _DECLINES
            return {**base, "status": "declined" if declined else "failed", "message": "The payment method was declined." if declined else f"{e.issue}: {e}"[:200]}
        except (httpx.HTTPError, PaymentsOff) as e:
            return {**base, "status": "failed", "message": str(e) or type(e).__name__}
        captures = (((order.get("purchase_units") or [{}])[0].get("payments") or {}).get("captures")) or []
        capture = captures[0] if captures else {}
        if order.get("status") == "COMPLETED" and capture.get("status") == "COMPLETED":
            return {**base, "status": "succeeded", "id": capture["id"], "order": order.get("id"), "message": ""}
        if capture.get("status") == "DECLINED":
            return {**base, "status": "declined", "id": capture.get("id"), "order": order.get("id"), "message": "The payment method was declined."}
        return {**base, "status": "failed", "id": capture.get("id"), "order": order.get("id"), "message": f"PayPal left the payment as {capture.get('status') or order.get('status')}"}

    # ------------------------------------------------------------------ the invoice

    async def invoice(self, name: str, email: str, lines: list[tuple], currency: str, payment_id: str, shop: str, key: str) -> dict:
        """Write a PayPal invoice for a walk-out that has been paid, and mark it as paid. PayPal itself sends nothing:
        the thank-you email (mailer.py) carries the link. Returns {status: created | failed, id, number, url, message}."""
        currency = currency.upper()
        total = round(sum(line[1] for line in lines), 2)
        first, _, last = name.strip().partition(" ")
        recipient = {"name": {"given_name": first[:140] or "Shopper", "surname": last[:140] or "-"}}
        if email:
            recipient["email_address"] = email
        draft = {
            "detail": {
                "currency_code": currency,
                "reference": payment_id[:120],
                "invoice_date": date.today().isoformat(),
                "note": "Thank you for purchasing with us. Looking forward to meeting you again.",
                "memo": f"Paid at walk-out. PayPal capture {payment_id}.",
                "payment_term": {"term_type": "DUE_ON_RECEIPT"},
            },
            "invoicer": {"business_name": shop[:300]},
            "primary_recipients": [{"billing_info": recipient}],
            "items": [{"name": line[0][:200], "description": (line[2] if len(line) > 2 else "")[:1000], "quantity": "1", "unit_amount": money(line[1], currency)} for line in lines],
        }
        try:
            made = await self._call("POST", "/v2/invoicing/invoices", draft, key=f"inv-{key}")
            invoice_id = made["id"]
            await self._call("POST", f"/v2/invoicing/invoices/{invoice_id}/send", {"send_to_recipient": False, "send_to_invoicer": False}, key=f"send-{key}")
            await self._call("POST", f"/v2/invoicing/invoices/{invoice_id}/payments", {"method": "PAYPAL", "payment_date": date.today().isoformat(), "amount": money(total, currency), "note": f"PayPal capture {payment_id}"}, key=f"paid-{key}")
            shown = await self._call("GET", f"/v2/invoicing/invoices/{invoice_id}")
        except PayPalError as e:
            return {"status": "failed", "id": None, "number": None, "url": None, "message": f"{e.issue}: {e}"[:200]}
        except (httpx.HTTPError, PaymentsOff, KeyError) as e:
            return {"status": "failed", "id": None, "number": None, "url": None, "message": str(e) or type(e).__name__}
        detail = shown.get("detail") or {}
        url = (detail.get("metadata") or {}).get("recipient_view_url") or f"https://www.sandbox.paypal.com/invoice/p/#{invoice_id}"
        return {"status": "created", "id": invoice_id, "number": detail.get("invoice_number"), "url": url, "state": shown.get("status"), "message": ""}

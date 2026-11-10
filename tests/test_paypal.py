"""
The PayPal provider against a stand-in for PayPal's API: what it sends when it saves a method, charges it and writes
the invoice, and that it will not run against anything but the sandbox.
"""
import asyncio
import json

import httpx

from checkout.payments import API, Card, Payments

ENV = {"PAYPAL_CLIENT_ID": "client", "PAYPAL_CLIENT_SECRET": "secret", "PAYPAL_ENV": "sandbox"}


class FakePayPal:
    """Answers like the sandbox does, and keeps every request it was sent."""

    def __init__(self):
        self.calls: list[tuple[str, str, dict, dict]] = []
        self.decline = None  # an issue code to refuse the next order with
        self.captures: dict[str, str] = {}

    def __call__(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/v1/oauth2/token":
            self.calls.append(("POST", path, {}, dict(request.headers)))
            return httpx.Response(200, json={"access_token": "token-1", "expires_in": 3600})
        body = json.loads(request.content) if request.content else {}
        self.calls.append((request.method, path, body, dict(request.headers)))
        assert request.headers["authorization"] == "Bearer token-1"
        if path == "/v3/vault/setup-tokens":
            source = body["payment_source"]
            if "card" in source:
                return httpx.Response(201, json={"id": "SETUP-CARD", "status": "APPROVED", "customer": {"id": "CUST1"}})
            return httpx.Response(201, json={"id": "SETUP-WALLET", "status": "PAYER_ACTION_REQUIRED", "links": [{"rel": "approve", "href": "https://www.sandbox.paypal.com/agreements/approve?approval_session_id=S1"}]})
        if path == "/v3/vault/payment-tokens":
            if body["payment_source"]["token"]["id"] == "SETUP-CARD":
                return httpx.Response(201, json={"id": "VAULT-CARD", "customer": {"id": "CUST1"}, "payment_source": {"card": {"brand": "VISA", "last_digits": "1881"}}})
            return httpx.Response(201, json={"id": "VAULT-WALLET", "customer": {"id": "CUST2"}, "payment_source": {"paypal": {"email_address": "buyer@example.com"}}})
        if path == "/v2/checkout/orders":
            if self.decline:
                return httpx.Response(422, json={"name": "UNPROCESSABLE_ENTITY", "details": [{"issue": self.decline, "description": "The instrument presented was declined."}]})
            key = request.headers["paypal-request-id"]
            capture = self.captures.setdefault(key, f"CAP{len(self.captures) + 1}")  # the same key is the same payment
            unit = body["purchase_units"][0]
            return httpx.Response(201, json={"id": "ORDER1", "status": "COMPLETED", "purchase_units": [{"payments": {"captures": [{"id": capture, "status": "COMPLETED", "amount": unit["amount"]}]}}]})
        if path.startswith("/v2/payments/captures/") and path.endswith("/refund"):
            if path.split("/")[4] not in self.captures.values():
                return httpx.Response(404, json={"name": "RESOURCE_NOT_FOUND", "details": [{"issue": "INVALID_RESOURCE_ID", "description": "Specified resource ID does not exist."}]})
            key = request.headers["paypal-request-id"]
            self.refunds = getattr(self, "refunds", {})
            refund = self.refunds.setdefault(key, f"REF{len(self.refunds) + 1}")  # the same key is the same refund
            return httpx.Response(201, json={"id": refund, "status": "COMPLETED", "amount": body["amount"]})
        if path == "/v2/invoicing/invoices" and request.method == "POST":
            return httpx.Response(201, json={"id": "INV2-AAAA", "status": "DRAFT"})
        if path.endswith("/send") or path.endswith("/payments"):
            return httpx.Response(200, json={})
        if path == "/v2/invoicing/invoices/INV2-AAAA":
            return httpx.Response(200, json={"id": "INV2-AAAA", "status": "MARKED_AS_PAID", "detail": {"invoice_number": "0007", "metadata": {"recipient_view_url": "https://www.sandbox.paypal.com/invoice/p/#AAAA"}}})
        return httpx.Response(404, json={"name": "RESOURCE_NOT_FOUND"})

    def sent(self, path):
        return [(body, headers) for _, p, body, headers in self.calls if p == path]


def provider():
    fake = FakePayPal()
    return Payments(ENV, http=httpx.AsyncClient(base_url=API, transport=httpx.MockTransport(fake))), fake


def run(coro):
    return asyncio.run(coro)


def test_only_the_sandbox_is_ever_used():
    assert API == "https://api-m.sandbox.paypal.com"
    assert Payments(ENV).http.base_url.host == "api-m.sandbox.paypal.com"
    live = Payments({**ENV, "PAYPAL_ENV": "live"})
    assert not live.enabled and "refuses" in live.problem and live.http is None
    assert not Payments({**ENV, "PAYPAL_ENV": ""}).enabled  # not saying is not sandbox
    missing = Payments({"PAYPAL_ENV": "sandbox"})
    assert not missing.enabled and "not set" in missing.problem

    async def refused():
        result = await live.charge(Card("c", "v", "visa", "1881"), 10.0, "USD", "x", "k")
        assert result["status"] == "failed" and result["id"] is None

    run(refused())


def test_a_card_is_saved_through_a_setup_token():
    async def scenario():
        pay, paypal = provider()
        card = await pay.save_card("Ada Lovelace", "visa", "shopper-1", "ada@example.com")
        assert card == Card("CUST1", "VAULT-CARD", "visa", "1881")
        (setup, _), = paypal.sent("/v3/vault/setup-tokens")
        assert setup["payment_source"]["card"]["number"] == "4012888888881881" and setup["payment_source"]["card"]["name"] == "Ada Lovelace"
        (token, _), = paypal.sent("/v3/vault/payment-tokens")
        assert token == {"payment_source": {"token": {"id": "SETUP-CARD", "type": "SETUP_TOKEN"}}}
        typed = await pay.save_card("Ada", "", "shopper-2", details={"number": "4012 8888 8888 1881", "expiry": "2031-01", "security_code": "999"})
        assert typed.last4 == "1881" and paypal.sent("/v3/vault/setup-tokens")[-1][0]["payment_source"]["card"]["number"] == "4012888888881881"
        try:
            await pay.save_card("Ada", "amex-gold", "shopper-3")
        except ValueError:
            pass
        else:
            raise AssertionError("an unknown test card was accepted")
        assert len(paypal.sent("/v1/oauth2/token")) == 1  # one sign-in for all of that

    run(scenario())


def test_a_paypal_account_is_saved_for_charging_without_the_shopper():
    async def scenario():
        pay, paypal = provider()
        begun = await pay.begin_wallet("shopper-1", "http://localhost:8765/gate/return?reg=r1", "http://localhost:8765/gate/return?reg=r1&cancelled=1", "Bookshop")
        assert begun == {"id": "SETUP-WALLET", "approve_url": "https://www.sandbox.paypal.com/agreements/approve?approval_session_id=S1"}
        wallet = paypal.sent("/v3/vault/setup-tokens")[0][0]["payment_source"]["paypal"]
        assert wallet["usage_type"] == "MERCHANT" and wallet["experience_context"]["return_url"].endswith("reg=r1")
        saved = await pay.finish_wallet(begun["id"])
        assert saved == Card("CUST2", "VAULT-WALLET", "paypal", "", "buyer@example.com")

    run(scenario())


def test_the_walk_out_charge_uses_the_saved_method_once():
    async def scenario():
        pay, paypal = provider()
        lines = [("Elon Musk", 79.0, "Books · Biography"), ("Blue Ocean Strategy", 85.0, "Books · Business"), ("The 7 Habits of Highly Effective People", 85.0, "Books · Self-help")]
        paid = await pay.charge(Card("CUST1", "VAULT-CARD", "visa", "1881"), 249.0, "USD", "Walk-out checkout", "sess-1-0", {"shopper": "p1"}, lines=lines)
        assert paid == {"amount": 249.0, "currency": "usd", "id": "CAP1", "receipt_url": None, "status": "succeeded", "order": "ORDER1", "message": ""}
        order, headers = paypal.sent("/v2/checkout/orders")[0]
        assert order["intent"] == "CAPTURE" and headers["paypal-request-id"] == "sess-1-0"
        assert order["payment_source"] == {"card": {"vault_id": "VAULT-CARD", "stored_credential": {"payment_initiator": "MERCHANT", "payment_type": "UNSCHEDULED", "usage": "SUBSEQUENT"}}}
        unit = order["purchase_units"][0]
        assert unit["amount"]["value"] == "249.00" and unit["amount"]["currency_code"] == "USD" and unit["custom_id"] == "p1"
        assert [i["name"] for i in unit["items"]] == [line[0] for line in lines] and unit["amount"]["breakdown"]["item_total"]["value"] == "249.00"

        again = await pay.charge(Card("CUST1", "VAULT-CARD", "visa", "1881"), 249.0, "USD", "Walk-out checkout", "sess-1-0")
        assert again["id"] == "CAP1"  # a retry with the same key is the same payment, not a second one

        wallet = await pay.charge(Card("CUST2", "VAULT-WALLET", "paypal", "", "buyer@example.com"), 18.5, "usd", "Walk-out checkout", "sess-1-1")
        assert wallet["status"] == "succeeded" and paypal.sent("/v2/checkout/orders")[-1][0]["payment_source"] == {"paypal": {"vault_id": "VAULT-WALLET"}}

    run(scenario())


def test_a_charge_that_cannot_go_through_says_why():
    async def scenario():
        pay, paypal = provider()
        card = Card("CUST1", "VAULT-CARD", "visa", "1881")
        dirhams = await pay.charge(card, 249.0, "AED", "Walk-out checkout", "k1")
        assert dirhams["status"] == "failed" and "AED" in dirhams["message"] and not paypal.sent("/v2/checkout/orders")
        paypal.decline = "INSTRUMENT_DECLINED"
        declined = await pay.charge(card, 249.0, "USD", "Walk-out checkout", "k2")
        assert declined["status"] == "declined" and declined["id"] is None
        paypal.decline = "INVALID_REQUEST"
        broken = await pay.charge(card, 249.0, "USD", "Walk-out checkout", "k3")
        assert broken["status"] == "failed" and "INVALID_REQUEST" in broken["message"]

    run(scenario())


def test_the_invoice_is_written_marked_paid_and_not_emailed_by_paypal():
    async def scenario():
        pay, paypal = provider()
        lines = [("Elon Musk", 79.0, "Books · Biography"), ("Deep Work", 59.0, "Books · Productivity")]
        made = await pay.invoice("Ada Lovelace", "ada@example.com", lines, "USD", "CAP1", "Bookshop", "sess-1-0")
        assert made == {"status": "created", "id": "INV2-AAAA", "number": "0007", "url": "https://www.sandbox.paypal.com/invoice/p/#AAAA", "state": "MARKED_AS_PAID", "message": ""}
        draft = paypal.sent("/v2/invoicing/invoices")[0][0]
        assert draft["detail"]["currency_code"] == "USD" and draft["detail"]["reference"] == "CAP1"
        assert draft["primary_recipients"][0]["billing_info"] == {"name": {"given_name": "Ada", "surname": "Lovelace"}, "email_address": "ada@example.com"}
        assert [(i["name"], i["description"], i["unit_amount"]["value"]) for i in draft["items"]] == [("Elon Musk", "Books · Biography", "79.00"), ("Deep Work", "Books · Productivity", "59.00")]
        assert paypal.sent("/v2/invoicing/invoices/INV2-AAAA/send")[0][0] == {"send_to_recipient": False, "send_to_invoicer": False}
        assert paypal.sent("/v2/invoicing/invoices/INV2-AAAA/payments")[0][0]["amount"] == {"currency_code": "USD", "value": "138.00"}

    run(scenario())


def test_a_refund_goes_back_once_however_often_it_is_asked_for():
    async def scenario():
        pay, paypal = provider()
        paid = await pay.charge(Card("CUST1", "VAULT-CARD", "visa", "1881"), 249.0, "USD", "Walk-out checkout", "s1-1-0")
        back = await pay.refund(paid["id"], 79.0, "USD", "refund-7-31-0", "Refund for Elon Musk")
        assert back == {"amount": 79.0, "currency": "usd", "id": "REF1", "status": "refunded", "message": ""}
        (body, headers), = paypal.sent(f"/v2/payments/captures/{paid['id']}/refund")
        assert body == {"amount": {"currency_code": "USD", "value": "79.00"}, "note_to_payer": "Refund for Elon Musk"}
        assert headers["paypal-request-id"] == "refund-7-31-0"
        again = await pay.refund(paid["id"], 79.0, "USD", "refund-7-31-0")  # a retry of the same refund
        assert again["id"] == "REF1"
        missing = await pay.refund("NOPE", 5.0, "USD", "refund-x")
        assert missing["status"] == "failed" and "INVALID_RESOURCE_ID" in missing["message"]
        nothing = await pay.refund(paid["id"], 0.0, "USD", "refund-y")
        assert nothing["status"] == "failed" and len(paypal.sent(f"/v2/payments/captures/{paid['id']}/refund")) == 2  # never sent

    run(scenario())

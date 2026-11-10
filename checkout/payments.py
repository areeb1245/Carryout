"""
Card on file and the charge at the door, on Stripe's sandbox.

Check-in saves a card to a new Stripe Customer with a SetupIntent (usage off_session). Walking out creates one
PaymentIntent for the cart, confirmed off-session: the shopper is not there to approve anything.

Only test keys are accepted. A key that does not start with `sk_test_` is refused, so this demo cannot move real money.
Cards are Stripe's own test payment methods (`pm_card_visa` and friends); no card number is typed or stored here.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass

import stripe

from .sam_client import load_env

#: Stripe test payment methods offered at check-in. The last one saves fine and is declined when charged.
TEST_CARDS = {
    "pm_card_visa": "Visa",
    "pm_card_mastercard": "Mastercard",
    "pm_card_visa_debit": "Visa debit",
    "pm_card_chargeCustomerFail": "Card that is declined at the door",
}
#: handed out in turn to shoppers who are checked in automatically (this sandbox account takes Visa and Mastercard)
ROTATION = ["pm_card_visa", "pm_card_mastercard", "pm_card_visa_debit"]


class PaymentsOff(RuntimeError):
    pass


@dataclass
class Card:
    customer: str
    payment_method: str
    brand: str
    last4: str


class Payments:
    def __init__(self, env: dict | None = None):
        env = load_env() if env is None else env
        key = env.get("STRIPE_SECRET_KEY", "")
        self.enabled = bool(key)
        self.problem = ""
        if key and not key.startswith("sk_test_"):
            # never a live key: this app charges cards on its own say-so
            self.enabled, self.problem = False, "STRIPE_SECRET_KEY is not a test key (sk_test_...); payments are off"
        elif not key:
            self.problem = "STRIPE_SECRET_KEY is not set in .env; payments are off"
        self.client = stripe.StripeClient(key) if self.enabled else None

    def _need(self) -> stripe.StripeClient:
        if not self.client:
            raise PaymentsOff(self.problem)
        return self.client

    async def save_card(self, name: str, test_card: str, shopper_id: str, email: str | None = None) -> Card:
        """New Customer with one of Stripe's test cards saved for charging later."""
        if test_card not in TEST_CARDS:
            raise ValueError("not one of the offered test cards")
        client = self._need()

        def work() -> Card:
            who = {"name": name, "metadata": {"app": "sam3-supermarket", "shopper": shopper_id}}
            if email:
                who["email"] = email
            customer = client.v1.customers.create(params=who)
            setup = client.v1.setup_intents.create(
                params={
                    "customer": customer.id,
                    "payment_method": test_card,
                    "confirm": True,
                    "usage": "off_session",
                    "automatic_payment_methods": {"enabled": True, "allow_redirects": "never"},
                }
            )
            method = client.v1.payment_methods.retrieve(setup.payment_method)
            return Card(customer.id, method.id, method.card.brand, method.card.last4)

        return await asyncio.to_thread(work)

    async def charge(self, card: Card, amount: float, currency: str, description: str, key: str, metadata: dict | None = None) -> dict:
        """One off-session charge. `key` makes a retry of the same walk-out charge once, not twice.

        Returns {status: succeeded | declined | failed, id, amount, currency, receipt_url, message}.
        """
        client = self._need()
        minor = round(amount * 100)
        base = {"amount": round(amount, 2), "currency": currency.lower()}

        def work() -> dict:
            try:
                intent = client.v1.payment_intents.create(
                    params={
                        "amount": minor,
                        "currency": currency.lower(),
                        "customer": card.customer,
                        "payment_method": card.payment_method,
                        "off_session": True,
                        "confirm": True,
                        "description": description,
                        "metadata": metadata or {},
                        "expand": ["latest_charge"],
                    },
                    options={"idempotency_key": key},
                )
            except stripe.CardError as e:
                intent_id = (e.error.payment_intent or {}).get("id") if e.error else None
                return {**base, "status": "declined", "id": intent_id, "receipt_url": None, "message": e.user_message or "The card was declined."}
            except stripe.StripeError as e:
                return {**base, "status": "failed", "id": None, "receipt_url": None, "message": e.user_message or str(e)}
            charge = intent.latest_charge
            ok = intent.status == "succeeded"
            return {**base, "status": "succeeded" if ok else "failed", "id": intent.id, "receipt_url": getattr(charge, "receipt_url", None), "message": "" if ok else f"Stripe left the payment as {intent.status}"}

        return await asyncio.to_thread(work)

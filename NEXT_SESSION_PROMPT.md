# Prompt for the next session: PayPal, a registration gate, and a check before charging

Paste everything below the line into a new chat opened in this folder.

---

You are continuing work on the walk-out checkout demo in `/Users/areebpasha/Desktop/PROJECTS/sam3-supermarket`. Read `README.md` first: it explains the whole system and its limits. Then skim `checkout/pipeline.py`, `checkout/payments.py`, `checkout/identity.py`, `checkout/server.py` and `web/app.js`.

## Where things stand

- A camera feed goes through SAM 3.1 (my own Modal app `sam3-supermarket`, an L40S that bills only while it is up). The engine tracks shoppers and what they take.
- The bookshop catalogue (`shops/bookshop.json`, 27 books) is the default shop. A book is billed only when the DeepSeek model has read its title in the shopper's hands on two looks. One line per title per shopper. Anything not identified is shown as not charged.
- Faces are matched locally (OpenCV YuNet + SFace). Profiles, visits and purchases are in SQLite (`data/checkout.db`).
- Payment today is Stripe sandbox: card saved at check-in, charged when the shopper walks out, invoice emailed through Resend.
- My test clip `samples/IMG_6401.MOV` must come out as Elon Musk, Blue Ocean Strategy and The 7 Habits of Highly Effective People, AED 249.00. `uv run pytest` passes 33 tests.

## What I want built

Think through the customer's experience end to end first, write me a short plan of the journey and the screens, then build it. It should feel seamless, the way a real customer would move through a shop.

### 1. Keep Stripe on a branch, put PayPal on main

- This folder is not a git repository. I do not want a GitHub repo created. Ask me before running `git init`; if I agree, keep it local only (no remote, no push), commit the current Stripe version, and create a branch named `stripe` that keeps it.
- If a remote is ever needed, it must be under the GitHub account `areeb1501`.
- On main, PayPal replaces Stripe as the payment provider. Keep the provider behind the same small interface `checkout/payments.py` has now (save a payment method, charge it) so the two stay swappable.

### 2. PayPal sandbox as the payment provider

- Credentials are already in `.env` as `PAYPAL_CLIENT_ID`, `PAYPAL_CLIENT_SECRET` and `PAYPAL_ENV=sandbox`. Never print them, never put them anywhere but `.env`, and refuse to run against live PayPal.
- Use the PayPal vault: the shopper saves a payment method once at registration (setup token, approved by the shopper, exchanged for a payment token with `usage_type: MERCHANT`), and it is charged later with the shopper not present (Orders v2 with the saved `vault_id`, a `PayPal-Request-Id` for idempotency).
- For now registration can simply take whatever test details the shopper enters: a PayPal sandbox test card, or approval with a sandbox PayPal account. Plan for "tap a card or scan a code to register" as the real-world version, but do not build hardware integration.
- Check the current PayPal docs through Context7 before writing calls. In particular confirm which currencies the sandbox accepts: I believe AED is not supported, in which case switch the catalogue currency (USD is fine) and keep the prices.

### 3. A registration gate before shopping

Shopping must not begin until the person is registered.

- A gate screen, separate from the shop-floor screen. I suggested a subdomain: serve it at `gate.localhost:8765` if that is simple, and at `/gate` either way.
- The person stands in front of the gate camera. If their face matches a profile, greet them by name, show that a payment method is on file, and let them in.
- If no profile matches, ask them to register right there: name, email, payment method (step 2), and their face captured from the camera. Then let them in.
- Only people who came through the gate have an open shopping visit. On the shop-floor camera a person with no open visit is flagged as not registered, and nothing is charged to them.
- Test credentials only, throughout.

### 4. One model check before charging

When a shopper finishes (walks out), and before any charge:

- Make one call to the DeepSeek model (`checkout/verify.py` has the client) with the evidence for that visit: the cart lines and the frames in which the shopper was seen carrying those items.
- Ask it to confirm, line by line, that the shopper really took each item. Charge only the lines it confirms. Show the rest as held back, with the reason.
- Keep the existing rules: never charge the same title twice, never charge an item that was put back, never guess a title.

### 5. Charge, then email the invoice

- Charge the confirmed lines to the vaulted PayPal method in one payment.
- Create the invoice with PayPal's Invoicing API and email the shopper: thank you for purchasing with us, the items with their category and section, the total, the link to the PayPal invoice, and "looking forward to meeting you again".
- Email goes through Resend as it does now. With Resend's default sender only `demoareeb@gmail.com` is deliverable, so use that for tests.

## Rules for working

- Before any run that charges or emails, replay with `uv run tools/live_client.py <clip> --dry` until the cart is right several times in a row. I care more about never over-charging than about catching every item.
- Keep the 33 tests passing and add tests for the gate, the PayPal provider (with a fake client), and the pre-charge check.
- Verify the result on `samples/IMG_6401.MOV`: three books, the right total, one real sandbox charge, one invoice email.
- Tell me plainly what you verified and what you could not (for example, the PayPal approval step needs me to click through the sandbox window).
- Update `README.md` when the behaviour changes.

## Done means

1. Stripe version preserved on the `stripe` branch (if I agreed to local git), PayPal on main.
2. Gate page works: known face is let in, unknown face registers with a PayPal sandbox method and is let in.
3. Shop floor only bills people with an open visit.
4. The pre-charge model check runs on every walk-out and its verdict shows on the cart.
5. One sandbox PayPal charge and one invoice email with the PayPal invoice link, both matching the cart, on my test clip.

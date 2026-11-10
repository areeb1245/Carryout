# Walk-out checkout on SAM 3.1

A cashierless-shop demo with two stations, each with its own camera.

1. **The gate.** A shopper stands in front of the gate camera. A face the shop knows, with a payment method on file,
   is let in. A new face registers on the spot: name, email and a PayPal payment method. Nobody shops unregistered.
2. **The shop floor.** A second camera watches the shelf. SAM 3.1 outlines every shopper, hand and product, and the
   shopper whose hand takes an item off the shelf gets it on their bill. Put it back and it comes off.
3. **The door.** When a shopper walks out, one model call checks their bill line by line against the pictures of them
   carrying things: took it, put it back, or unclear. Only "took it" is charged, in one PayPal payment. A PayPal
   invoice is written and emailed.

Two shops come with it (`shops/`): a supermarket, where each product has its own phrase, and a bookshop, where every
book is just "book" to SAM and the title is read off the book in the shopper's hands.

Payments run on **PayPal's sandbox**. The earlier Stripe version is kept on the `stripe` branch; both sit behind the
same two calls in `checkout/payments.py` (save a method, charge it).

## Run it

```bash
uv run uvicorn checkout.server:app --port 8765
```

| Screen | Address | For |
|---|---|---|
| The gate | http://localhost:8765/gate (also http://gate.localhost:8765) | shoppers: be recognised or register, then walk in |
| The shop floor | http://localhost:8765 | the shop: the camera, each shopper's bag, what happened at the door |

On the shop floor pick a shop, then the footage (a camera plugged into this computer, a sample clip or a video
file), and press Start. **Rehearsal** runs everything except the charge, the invoice and the email. The first frame
after the GPU has been idle waits for it to wake (up to a minute); the page says so. Each screen has its own camera
chooser, so the gate and the shop floor can be tied to different cameras. Products are added in
**Shop settings → Catalogue → Add a product**; a new product is found by the phrase you give it, or by its name.

Keys live in `.env`, which git ignores:

| Variable | For |
|---|---|
| `SAM_URL`, `SAM_API_KEY` | the SAM 3.1 endpoint: the `sam3-supermarket` Modal app (below) |
| `SAM_MAX_PROMPTS`, `SAM_MAX_FPS` | what that endpoint takes: phrases a request, frames a second |
| `SAM_SHARED_URL`, `SAM_SHARED_API_KEY` | the older shared `deck-twin-sam31` app. Copy them over `SAM_URL` and `SAM_API_KEY`, with 8 and 3, to fall back to it |
| `LLM_URL`, `LLM_MODAL_KEY`, `LLM_MODAL_SECRET` | the DeepSeek vision model on Modal: reads titles in the hands, and makes the check before charging |
| `PAYPAL_CLIENT_ID`, `PAYPAL_CLIENT_SECRET`, `PAYPAL_ENV` | PayPal. `PAYPAL_ENV` must be `sandbox`; anything else and payments stay off. The code only ever calls `api-m.sandbox.paypal.com` |
| `DATABASE_URL` | Postgres with pgvector (Neon): the face embeddings the gate searches. Without it faces are matched on this machine |
| `RESEND_API_KEY`, `RESEND_FROM` | invoice emails |
| `DEMO_EMAIL` | prefilled in the gate's registration form |

PayPal, the database and email are each optional: without one, that part switches off and the screens say which.

## The gate

`checkout/gate.py`, `web/gate.html`. The gate camera sends a frame about once a second.

- **Known face, method on file:** "Welcome back", the method on file is shown, and a visit opens (a row in
  `admissions`). The same answer has to come twice running before the gate acts on it.
- **New face:** the form asks for a name, an email address and how to pay:
  - *PayPal account*: a vault setup token with `usage_type: MERCHANT`; the shopper approves it in a PayPal window,
    PayPal sends them back to `/gate/return`, and the approval is exchanged for a payment token.
  - *Card*: a sandbox test card saved the same way (setup token, then payment token). This needs a sandbox app
    whose business account can take cards (a US one does); otherwise PayPal answers
    `NOT_ENABLED_TO_VAULT_PAYMENT_SOURCE` and the gate says so.
  Nothing is saved until the payment method is: no method, no profile, no way in.
- **Known face without a usable method** (for example a profile from the Stripe version): asked to add one.
- **No camera:** "Use a photo" does the same with a picture, which is also how to try the gate on a laptop.

The real-world version of the form is "tap a card or scan a code"; no hardware is wired up here.

### Faces

Faces are found with OpenCV's YuNet and turned into a 128-number embedding with SFace, both on this machine in about
15 ms. Each profile keeps up to six embeddings from different angles and one small face picture.

With `DATABASE_URL` set the embeddings are also stored in Postgres with pgvector (`checkout/vectors.py`, table
`checkout_faces`, HNSW index on cosine distance), and the gate asks that index whose face it sees: one
nearest-neighbour query however many people have registered, about 110 ms to Neon in Frankfurt. Only embeddings and
profile ids go there. Pictures, names, emails and payment links stay in `data/checkout.db`. The shop floor matches
against the copy in memory, so it never waits on the network.

## What happens to a shopper on the floor

1. **Walks in.** The person mask becomes a tracked shopper with a stable id and colour.
2. **Face.** Their face is matched to the profiles. Someone with an open visit shows by name, with their payment
   method. Anyone else, an unknown face or a profile that did not come through the gate this time, is flagged
   **Not registered**, and nothing is ever charged to them.
3. **Picks and returns.** Decided from the masks (below). Each pick is also shown to the DeepSeek model on up to
   three frames; together those looks name the product, or dispute the pick.
4. **Walks out.** Three seconds out of view counts as leaving (as does the end of a clip, or Stop).
5. **The check before charging.** One call to the DeepSeek model with the bill and up to five pictures: for each
   line the picture in which that item was read in their hands, then the latest close view of their hands, then the
   whole scene the last time they were seen. For every line the model reports the picture it sees the item held in
   and, if any, a later picture in which it sees the item at rest again. From that:
   - **kept**: held in a picture, not seen put down afterwards. Charged.
   - **put back**: seen at rest again in a later picture. Not charged.
   - **unclear**: no picture shows it in their hands. Not charged.
   No answer from the model, or no picture kept, also means not charged. The bag shows each verdict with its reason
   and the pictures the check was shown.
6. **Charge.** The kept lines, in one Orders v2 payment against the saved method (`vault_id`), shopper not present,
   with a `PayPal-Request-Id` so a retry cannot charge twice.
7. **Invoice.** A PayPal invoice (Invoicing API) with the lines and their category and section, marked as paid. The
   email through Resend says thank you, lists the items and the total, links the PayPal invoice and ends with
   "looking forward to meeting you again". PayPal itself emails nobody.
8. **Visit closed.** The next visit starts at the gate again.

Lines that are doubtful are never charged: not in the catalogue, a book whose title could not be read, under 50 %
confidence with no confirmation, disputed by the second model, or not confirmed by the check before charging. They
stay on the bag marked as not charged, with the reason.

## The SAM endpoint

`modal_app/sam_checkout.py` is this app's own SAM 3.1 deployment. It runs every phrase of a frame through the detector
as one batch instead of one after another, and caches the phrase embeddings. Measured on one frame with eight phrases:

| | one phrase at a time | batched |
|---|---|---|
| A10G (what the shared app uses) | 575 ms | 367 ms |
| L40S (what this app is deployed on) | 290 ms | 139 ms |

```bash
modal deploy modal_app/sam_checkout.py              # https://<workspace>--sam3-supermarket.<region>.modal.run
BENCH_GPU=L40S modal run modal_app/sam_checkout.py  # the timing above, on a GPU of your choice
```

It reads the weights from the shared app's volume and its key from the Modal secret `sam3-supermarket`. One L40S
container at most, released five minutes after the last frame. Live, the app now analyses 3 to 4 frames a second with
answers about 0.4 s behind (it was 1.5 a second and 1.2 s on the shared endpoint).

## Books: who took which title

In `shops/bookshop.json` every book has a name, author, category (`Books`), a section within it, a price and some
metadata (format, publisher, what the spine looks like). All 27 share the phrase `book`: SAM is asked for it once and
outlines each book, and which title a shopper took is settled in two steps.

1. **Is it sure they took something?** The frame loop decides that, from the masks: a book counts as taken only when
   its outline leaves the shelf area in the shopper's hand. Touching, tilting and browsing on the shelf do not count.
2. **Which titles are they carrying?** Once something is taken, the DeepSeek model is shown that shopper about once
   a second and asked which catalogue titles are in their hands. Their cart follows those answers:
   - a title read on **two looks** goes on the cart (onto a pick that has no title yet, or as its own line if the
     outlines missed that pick, which happens when a book slides straight into a stack);
   - a title that drops out of **three looks in a row**, with the hands in view, comes off again: it was put down;
   - **one line per title per shopper**. A second line with the same title is the same book seen twice, and is never
     charged.

What is not charged is shown as such: a pick whose title was never read in the hands stays on the cart as
"Not identified, so not charged"; a name that was only read on the shelf (local OCR, `checkout/reader.py`) is shown
but held back until it is read in the hands. On a tight shelf the outline that follows a hand out is not always the
one that was read, so a shelf reading alone never bills anyone.

The shelf itself is read by a local OCR (RapidOCR, about 100 ms a spine) with the second model as a fallback. On the
photo of the real shelf (`samples/books.jpg`) that names all 27 books; the labels are drawn on the spines.

A title the outlines wrongly took for put back (a book held right next to the shelf) comes back onto the bill only
when it is read in the shopper's hands on two more looks.

**Checked on real footage** (`samples/IMG_6401.MOV`, hand-held phone, three books taken and stacked): Elon Musk,
Blue Ocean Strategy and The 7 Habits of Highly Effective People, USD 249.00. On 10 November 2026, 5 of 5 live
rehearsals in a row, and one from the page, ended with those three books, that total, and all three confirmed by the
check before charging. One real run then charged USD 249.00 to the saved sandbox Visa, wrote PayPal invoice 0002
for the same three lines, and sent the invoice email.

## A camera that moves

Picks are decided by items leaving their resting places, which only works if resting places stay put. For hand-held
footage the engine follows corner points on the background from frame to frame (people masked out) and carries every
remembered position, mask and shelf area along with the shift, turn and zoom it finds. Inferring a pick from an item
that has simply vanished is switched off while the camera is moving.

## How a pick is decided

Per analysed frame, one request asks SAM for `person`, `hand`, `item held in hand` and every catalogue phrase
(`catalog.json`). `checkout/engine.py` then:

- gives each hand to the shopper whose mask it lies in (and keeps the owner from the previous frame when one shopper
  reaches across another);
- fuses catalogue detections with the generic held-item mask into item observations, each knowing which hands touch it;
- tracks items as *candidate*, *at rest* or *held*, and keeps a decaying tally of hand contact per shopper on each.

| Decision | Evidence |
|---|---|
| Pick | a resting item leaves its place while a hand is on it; or an item first seen in a hand stays there across frames |
| Inferred pick | a steady shelf item is gone, its spot is in view, and a hand had just closed over it |
| Return | the item sits still with no hand on it, on a shelf or away from the shopper |
| Inferred return | last seen being put on a shelf, and the shopper's hands have been empty since |
| Hand-over | only another shopper's hands touch it for over a second |
| Same item again | an item that was hidden for a moment and reappears in the same shopper's hand is not counted twice |

The pick goes to the shopper with the largest share of the contact tally, not to whoever is nearest in one frame.
Thresholds are fractions of the frame diagonal and seconds, so they do not depend on resolution or frame rate.

## Files

| Path | What |
|---|---|
| `checkout/engine.py` | masks to picks, returns and carts |
| `checkout/people.py` | shopper tracking and re-identification by clothing |
| `checkout/gate.py` | the gate: recognise or register, then open a visit |
| `checkout/identity.py` | profiles, faces, gate admissions and visit history in SQLite |
| `checkout/vectors.py` | face embeddings in Postgres with pgvector |
| `checkout/payments.py` | PayPal sandbox: save a method in the vault, charge it, write the invoice |
| `checkout/reader.py` | local OCR and matching the text to catalogue titles |
| `checkout/verify.py` | the vision language model: second looks, titles in the hands, the check before charging |
| `checkout/mailer.py` | the invoice email through Resend |
| `modal_app/sam_checkout.py` | the batched SAM 3.1 endpoint on Modal |
| `shops/` | one catalogue per shop |
| `checkout/pipeline.py` | one visit: frame order, plus the jobs that run beside it |
| `checkout/server.py` | the local web server and WebSocket |
| `web/` | the two screens: `index.html` + `app.js` (shop floor), `gate.html` + `gate.js` (gate), one `styles.css` |
| `checkout/cli.py` | run a video file offline and write an annotated copy |
| `tools/live_client.py` | play a clip to the running server in real time, from the terminal |
| `models/` | the two OpenCV face models |
| `data/checkout.db` | profiles, face pictures and embeddings, admissions, visits. Stays on this machine |

## Tests

```bash
uv run pytest
```

`tests/test_engine.py` covers the pick logic on synthetic masks (two shoppers at one shelf, put-backs, bags,
hand-overs, re-entry). `tests/test_crowd.py` runs five shoppers at once and checks that only what was carried out is
charged, once. `tests/test_books.py` covers books: a title is billed only once read in the hands, never twice, an
unread one is not charged, and stray outlines are folded away. `tests/test_gate.py` covers the gate (known face let
in, new face registered with a card or a PayPal account, the vector index) and that the floor bills open visits
only. `tests/test_paypal.py` runs the PayPal provider against a stand-in for PayPal's API, and checks it refuses
anything but the sandbox. `tests/test_last_check.py` covers the check before charging.

```bash
uv run tools/live_client.py samples/IMG_6401.MOV --shop bookshop --width 1600 --dry   # rehearsal: no charge, no invoice, no email
uv run tools/live_client.py samples/IMG_6401.MOV --shop bookshop --width 1600         # for real: charges the registered shopper
uv run python -m checkout.cli samples/49350-720.mp4 --fps 3 --verify                  # offline, annotated video in out/
```

Rehearse before any real run: a real run charges the sandbox method and sends the invoice email. Only someone the
gate has let in is charged, so register at the gate screen first (a rehearsal leaves the visit open).

## Limits worth knowing

- **Frame rate.** 3 to 4 analysed frames a second, picture about 1.5 s behind the footage. A grab that is over in a
  fraction of a second can still be missed.
- **Books lean on the second model.** Titles are billed from what DeepSeek reads in the hands. A book held for under
  two seconds, or with its title hidden the whole time, is not identified and so not charged. Two copies of one
  title in the same hands are billed once.
- **One real book clip so far.** One shopper, three books, a moving camera. Several shoppers at one bookshelf, and
  putting a book back, are covered only by synthetic tests.
- **Products are found by phrase.** A vague phrase scores low; the second model fixes many of those, but a real shop
  would give each product its own trained signature.
- **One camera.** An item put in a trolley parked in front of a shelf looks like an item put back on it.
- **PayPal sandbox, USD.** PayPal does not charge in AED, so the catalogues are priced in USD (same numbers).
  Saving a card needs a sandbox app on a business account that takes cards (US); saving a PayPal account works
  on any.
- **The check before charging confirms "taken" well and "put back" weakly.** It reliably refuses an item that was
  never in the shopper's hands. It only calls a put-back when a picture shows the item at rest again, which a
  far-off shelf of spines rarely does; put-backs are mostly caught earlier, by the outlines and by the title
  dropping out of the hands.
- **One camera per station.** The floor recognises a shopper by their face, so someone who never faces the floor
  camera stays "not registered" and is not charged.
- **GPU cost.** The L40S bills only while it is up: from the first frame until five minutes after the last.
- **Resend's default sender** only delivers to the address that owns the Resend account.
- **Faces are biometric data.** The pictures never leave this machine. With `DATABASE_URL` set the embeddings are
  stored in that Postgres database. "Delete all shoppers" removes both, but anyone filmed for real needs to have
  agreed to it.

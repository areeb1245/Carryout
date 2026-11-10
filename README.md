# Walk-out checkout on SAM 3.1

A cashierless-shop demo. A camera watches the shelf, SAM 3.1 outlines every shopper, hand and product, and the shopper
whose hand takes an item off the shelf gets it on their bill. Put it back and it comes off. Walk out and the card on
your profile is charged on Stripe's sandbox, and an invoice is emailed.

Two shops come with it (`shops/`): a supermarket, where each product has its own phrase, and a bookshop, where every
book is just "book" to SAM and the title is read off the spine.

## Run it

```bash
uv run uvicorn checkout.server:app --port 8765
```

Open http://localhost:8765 (it opens on the bookshop), pick a shop, then a sample clip, the camera or a video file, and press Start. The first
frame after the GPU has been idle waits for it to wake (up to a minute); the page says so.

Keys live in `.env`, which git ignores:

| Variable | For |
|---|---|
| `SAM_URL`, `SAM_API_KEY` | the SAM 3.1 endpoint: the `sam3-supermarket` Modal app (below) |
| `SAM_MAX_PROMPTS`, `SAM_MAX_FPS` | what that endpoint takes: phrases a request, frames a second |
| `SAM_SHARED_URL`, `SAM_SHARED_API_KEY` | the older shared `deck-twin-sam31` app. Copy them over `SAM_URL` and `SAM_API_KEY`, with 8 and 3, to fall back to it |
| `LLM_URL`, `LLM_MODAL_KEY`, `LLM_MODAL_SECRET` | the DeepSeek vision model on Modal, used as the second opinion |
| `STRIPE_SECRET_KEY` | Stripe sandbox. Anything that is not `sk_test_...` is refused |
| `RESEND_API_KEY`, `RESEND_FROM` | invoice emails |
| `DEMO_EMAIL` | the address put on profiles that hands-free mode creates, and prefilled in the check-in form |

Each of the last three is optional: without it that part switches off and the page says which.

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

## What happens to a shopper

1. **Walks in.** The person mask becomes a tracked shopper with a stable id and colour.
2. **Face.** Their face is read on this machine (OpenCV YuNet + SFace, about 15 ms) and compared with the saved
   profiles. A match pulls up the profile: name, card, visits, what they bought last time.
3. **New face.** The page asks for a name, an email address and a card (one of Stripe's test cards). That becomes a
   profile: a row in `data/checkout.db` and a Stripe customer with the card saved for later. The "hands-free" switch
   does this without asking, for unattended demos.
4. **Picks and returns.** Decided from the masks (below). Each pick is also shown to the DeepSeek model on up to three
   frames; together those looks name the product, or dispute the pick.
5. **Walks out.** Three seconds out of view counts as leaving (as does the end of a clip, or Stop). The lines still
   in the cart, and that held up, are charged in one off-session payment. Then the invoice is emailed: a thank-you,
   the things bought, the total, and "looking forward to meeting you again".

Lines that are doubtful are never charged: not in the catalogue, a book whose title could not be read, under 50 %
confidence with no confirmation, or disputed by the second model. They stay on the cart marked "held back" for a
person to look at. Every charged line says how it was verified.

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

**Checked on real footage** (`samples/IMG_6401.MOV`, hand-held phone, three books taken and stacked): Elon Musk,
Blue Ocean Strategy and The 7 Habits of Highly Effective People, AED 249.00, in 5 of 5 offline runs and 5 of 5 live
runs, with the charge and the invoice matching.

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
| `checkout/identity.py` | profiles, faces and visit history in SQLite |
| `checkout/payments.py` | Stripe: save a card, charge it off-session |
| `checkout/reader.py` | local OCR and matching the text to catalogue titles |
| `checkout/verify.py` | the second opinion from the vision language model |
| `checkout/mailer.py` | the invoice email through Resend |
| `modal_app/sam_checkout.py` | the batched SAM 3.1 endpoint on Modal |
| `shops/` | one catalogue per shop |
| `checkout/pipeline.py` | one visit: frame order, plus the jobs that run beside it |
| `checkout/server.py` | the local web server and WebSocket |
| `web/` | the page |
| `checkout/cli.py` | run a video file offline and write an annotated copy |
| `tools/live_client.py` | play a clip to the running server in real time, from the terminal |
| `models/` | the two OpenCV face models |
| `data/checkout.db` | profiles, face signatures, visits. Stays on this machine |

## Tests

```bash
uv run pytest
```

`tests/test_engine.py` covers the pick logic on synthetic masks (two shoppers at one shelf, put-backs, bags,
hand-overs, re-entry). `tests/test_crowd.py` runs five shoppers at once and checks that only what was carried out is
charged, once. `tests/test_books.py` covers books: a title is billed only once read in the hands, never twice, an unread one
is not charged, and stray outlines are folded away.

```bash
uv run python -m checkout.cli samples/49350-720.mp4 --fps 3 --verify --pay   # offline, annotated video in out/
uv run tools/live_client.py samples/49350-720.mp4 --name Ada --card pm_card_visa   # real time, against the server
uv run tools/live_client.py samples/IMG_6401.MOV --shop bookshop --width 1600 --dry  # real time, no charge, no email
```

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
- **Stripe sandbox, AE account.** It takes Visa and Mastercard test cards only.
- **GPU cost.** The L40S bills only while it is up: from the first frame until five minutes after the last.
- **Resend's default sender** only delivers to the address that owns the Resend account.
- **Faces are biometric data.** They never leave this machine and "Delete all profiles" removes them, but anyone
  filmed for real needs to have agreed to it.

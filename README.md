# Walk-out checkout on SAM 3.1

A cashierless-shop demo with two stations, each with its own camera.

1. **The gate.** A shopper stands in front of the gate camera. A face the shop knows, with a payment method on file,
   is let in. A new face registers on the spot: name, email and a PayPal payment method. Nobody shops unregistered.
2. **The shop floor.** A second camera watches the shelf. SAM 3.1 outlines every shopper, hand and product, and the
   shopper whose hand takes an item off the shelf gets it on their bill. Put it back and it comes off.
3. **The door.** When a shopper walks out, one model call checks their bill line by line against the pictures of them
   carrying things: took it, put it back, or unclear. Only "took it" is charged, in one PayPal payment. A PayPal
   invoice is written and emailed.

Two shops come with it (`shops/`): a supermarket, where each product has its own phrase, and a bookshop (Folio House,
`shops/bookshop.json`), where every book is just "book" to SAM and the title is read off the book in the shopper's
hands.

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
| The catalogue | http://localhost:8765/catalogue | the shop: the products, their pictures, and adding new ones from a camera or a clip |
| The dashboard | http://localhost:8765/dashboard | the shop: sales, every transaction with the proof behind its bill, the customers, refunds |

On the shop floor pick a shop, then the footage (a camera plugged into this computer, a sample clip or a video
file), and press Start. **Rehearsal** runs everything except the charge, the invoice and the email. The camera model
runs on a GPU that switches off after five idle minutes, and switching it on takes up to a minute. The page asks for
it as it opens and keeps it awake while the page is in use (until 20 minutes after the last click or run), so Start
is immediate during a demo; the GPU bills for that time.

On the picture, whatever a shopper holds carries a small label with its name, in the shopper's colour, with a tick
once it is sure enough to be billed. Titles on their bag whose outline is hidden (the lower books of a stack) are
listed under the same hands. A live camera also shows the date and time of the frame, and the activity list shows
the time of day; the list can be narrowed to one shopper. Choosing "Live camera" lists
every camera this computer can see (the browser asks for permission first), and the gate has its own list, so the two
stations can be tied to different cameras. Products are added on
the **Catalogue** screen (below).

The three staff screens (floor, catalogue, dashboard) share one header: the shop's name, the three screens, "Open
gate" and a gear for **Settings** (what the floor's picture shows, which services are on, and deleting every
shopper). Registered shoppers are listed on the dashboard under Customers. The gate is the shoppers' screen and keeps
its own, larger layout.

**From a phone.** A phone's browser only hands over its camera on an https page, so open the app over https: the
Render address (below), or a tunnel to this computer (`ngrok http 8765`). The pages then use a secure socket
(`wss`). The gate is public; the staff screens need the staff code (next section). A free ngrok plan has a monthly
allowance that the gate's one picture a second and the floor's answers use up within hours.

### Staff sign-in

The gate (`/gate`, where shoppers register and check in, on their own phone if they like) is public. Everything
else is private: the floor, the catalogue, the dashboard, every call behind them and the floor's socket. The guard
in `checkout/server.py` is deny-by-default with a short public list (the gate and its calls, the PayPal return page,
`/signin`, `/healthz`, the static files), and a test walks every route to keep it that way.

Staff type the **staff code** (`STAFF_CODE`) once at `/signin`. The server answers with a signed token (JWT, HS256,
signed with `JWT_SECRET`, good for 12 hours), set as an HttpOnly cookie and also returned in the reply, so scripts can
send it as `Authorization: Bearer …`. Each token is written down by its id with a random salt and the SHA-256 of
salt + token (`staff_tokens` in the database; the token itself is never stored), and is accepted only while its
signature and expiry hold and it is on file, not revoked, with a matching hash. **Sign out** (in Settings) revokes it.
Five wrong codes from one address in ten minutes are turned away. `tools/live_client.py` signs in with
`--staff-code` (or `$STAFF_CODE`).

On this computer, with no `STAFF_CODE` in `.env`, nobody can sign in and the staff screens are open to this computer
only: anything through a tunnel or proxy is refused. A hosted copy does not start without both values.

### Deploying on Render

The repository is a Render Blueprint (`render.yaml`): one Python web service, `uv sync` from the lockfile, Python
3.11 (`.python-version`), a 1 GB persistent disk at `/var/data`, health checks on `/healthz` (which, unlike
`/api/health`, never wakes the GPU).

1. In Render, **New → Blueprint**, and pick this repository.
2. Fill in the values it asks for: `STAFF_CODE` (choose one, at least 8 characters) and the keys from your `.env`
   (`SAM_URL`, `SAM_API_KEY`, `LLM_*`, `PAYPAL_CLIENT_*`, `RESEND_*`, `DATABASE_URL`). `JWT_SECRET` is generated.
   `PAYPAL_ENV` is fixed to `sandbox`.
3. Deploy. Shoppers use `https://<service>.onrender.com/gate`; staff go to the same address and sign in.
4. In the PayPal developer dashboard nothing changes: the return address is taken from each request.

What to know:

- **Its own data.** The database, catalogues and clips live on the disk (`DATA_DIR`). On the first start the
  catalogues and their pictures are copied there from `shops/`; after that the disk's copy is the one the catalogue
  screen edits, and changes to `shops/` in the repository do not reach it. Shoppers register again on Render.
- **Its own face table.** `FACE_TABLE=checkout_faces_render`: the copy on this computer and the one on Render each
  keep the face table matching their own database, and "Delete all shoppers" empties it, so they must not share one.
- **One instance.** Gate tickets, live visits and the sign-in throttle live in the process's memory.
- **Plan.** A disk needs a paid plan. The server peaked at about 250 MB on the supermarket clip and 340 MB on the
  book clip; Starter (512 MB) fits, Standard if it restarts for memory.
- **The face models** (`models/*.onnx`, 37 MB) are in the repository, so the build needs nothing else.

Keys live in `.env`, which git ignores (`.env.example` lists them):

| Variable | For |
|---|---|
| `SAM_URL`, `SAM_API_KEY` | the SAM 3.1 endpoint: the `sam3-supermarket` Modal app (below) |
| `SAM_MAX_PROMPTS`, `SAM_MAX_FPS` | what that endpoint takes: phrases a request, frames a second |
| `SAM_SHARED_URL`, `SAM_SHARED_API_KEY` | the older shared `deck-twin-sam31` app. Copy them over `SAM_URL` and `SAM_API_KEY`, with 8 and 3, to fall back to it |
| `LLM_URL`, `LLM_MODAL_KEY`, `LLM_MODAL_SECRET` | the DeepSeek vision model on Modal: reads titles in the hands, and makes the check before charging |
| `PAYPAL_CLIENT_ID`, `PAYPAL_CLIENT_SECRET`, `PAYPAL_ENV` | PayPal. `PAYPAL_ENV` must be `sandbox`; anything else and payments stay off. The code only ever calls `api-m.sandbox.paypal.com` |
| `DATABASE_URL` | Postgres with pgvector (Neon): the face embeddings the gate searches. Without it faces are matched on this machine |
| `RESEND_API_KEY`, `RESEND_FROM` | invoice emails |
| `DEMO_EMAIL` | prefilled in the gate's registration form on this computer (never on the public gate) |
| `STAFF_CODE`, `JWT_SECRET` | staff sign-in (above) |
| `DATA_DIR`, `FACE_TABLE` | where the data lives, and this copy's own table of faces (above) |

PayPal, the database and email are each optional: without one, that part switches off and the screens say which.

## The gate

`checkout/gate.py`, `web/gate.html`. The gate camera sends a frame about once a second.

- **Known face, method on file:** "Welcome back", the method on file is shown, and a visit opens (a row in
  `admissions`, with the gate camera's picture and the match score, which later go on the transaction as proof).
  The same answer has to come twice running before the gate acts on it. The match percentage shows beside the name.
- **New face:** the form asks for a name, an email address and how to pay:
  - *PayPal account*: a vault setup token with `usage_type: MERCHANT`; the shopper approves it in a PayPal window,
    PayPal sends them back to `/gate/return`, and the approval is exchanged for a payment token.
  - *Card*: a sandbox test card saved the same way (setup token, then payment token). This needs a sandbox app
    whose business account can take cards (a US one does); otherwise PayPal answers
    `NOT_ENABLED_TO_VAULT_PAYMENT_SOURCE` and the gate says so.
  Nothing is saved until the payment method is: no method, no profile, no way in.
- **Known face without a usable method** (for example a profile from the Stripe version): asked to add one.
- **No camera:** "Use a photo" does the same with a picture, which is also how to try the gate on a laptop.

The sign at the top of the gate reads "Gate closed" until the gate has let in the person in front of it, then
"Gate open for" their name. It closes again as soon as they walk on or a different face steps up.

The real-world version of the form is "tap a card or scan a code"; no hardware is wired up here.

### Faces

Faces are found with OpenCV's YuNet and turned into a 128-number embedding with SFace, both on this machine in about
15 ms. Each profile keeps up to six embeddings from different angles and one small face picture.

With `DATABASE_URL` set the embeddings are also stored in Postgres with pgvector (`checkout/vectors.py`, table
`checkout_faces`, HNSW index on cosine distance), and the gate asks that index whose face it sees: one
nearest-neighbour query however many people have registered, about 110 ms to Neon in Frankfurt. Only embeddings and
profile ids go there. Pictures, names, emails and payment links stay in `data/checkout.db`. The shop floor matches
against the copy in memory, so it never waits on the network.

**Who counts as the same person.** SFace gives the same person about 0.6 and up when they face the camera, and two
different people usually under 0.2; two people who look alike have been seen to reach 0.35. So:

- at the gate, where a face is compared with everyone who ever registered, it takes 0.45 (`SURE`) and a lead of 0.1
  over the next profile. A face that only resembles someone is treated as a new face, and registers as itself;
- on the floor, a face is only compared with the people the gate has let in (and this run's own shoppers), and
  needs two looks naming the same person, or one look at 0.45, before anything goes on their bill;
- a profile only learns a new angle from a look it is sure of.

Nobody is turned away for an unclear picture; what the cameras saw is kept instead (below), so a wrong match can be
seen and refunded.

**The same shopper under a new outline.** If a shopper's outline is lost (they turn, are hidden, or step out for
longer than the cutoff) and a new outline turns up with their face, it is folded into the first: one shopper, one
bag. Something the new outline was first seen already holding joins the line they had for it, paid or not, so
nothing carried through the gap is billed twice.

## What happens to a shopper on the floor

1. **Walks in.** The person mask becomes a tracked shopper with a stable id and colour.
2. **Face.** Their face is matched to the profiles. Someone with an open visit shows by name, with their payment
   method. Anyone else, an unknown face or a profile that did not come through the gate this time, is flagged
   **Not registered**, and nothing is ever charged to them.
3. **Picks and returns.** Decided from the masks (below). Each pick is also shown to the DeepSeek model on up to
   three frames; together those looks name the product, or dispute the pick.
4. **Walks out.** A shopper whose outline has been out of the camera's view for three seconds has walked out (so
   has everyone still inside when a clip ends or Stop is pressed). It is the whole person that counts, not the
   face or what is in their hands. The number of seconds is set per shop on the **Catalogue** screen
   (`leave_after_s` in the catalogue). Walking out with something closes the bag and the visit. Leaving the
   picture with empty hands does neither: that shopper is still known when they come back.
5. **The check before charging.** One call to the DeepSeek model with the bill and up to five pictures: for each
   line the picture in which that item was read in their hands, then the latest close view of their hands, then the
   whole scene the last time they were seen. For every line the model reports the picture it sees the item held in
   and, if any, a later picture in which it sees the item at rest again. From that:
   - **kept**: held in a picture, not seen put down afterwards. Charged.
   - **put back**: seen at rest again in a later picture. Not charged.
   - **unclear**: no picture shows it in their hands. Charged only if its title was already read in their hands on
     the floor (two readings) before they left, not first named from the door pictures, and the door look did not
     find it gone: the usual case is a book carried under another
     in a stack, or one of several books with no picture of its own. Otherwise not charged.
   No answer from the model, or no picture kept, means not charged. The bag shows each verdict with its reason
   and the pictures the check was shown.
6. **Charge.** The kept lines, in one Orders v2 payment against the saved method (`vault_id`), shopper not present,
   with a `PayPal-Request-Id` so a retry cannot charge twice.
7. **Invoice.** A PayPal invoice (Invoicing API) with the lines and their category and section, marked as paid. The
   email through Resend says thank you, lists the items and the total, links the PayPal invoice and ends with
   "looking forward to meeting you again". PayPal itself emails nobody.
8. **Visit closed.** The next visit starts at the gate again.
9. **Written down.** The walk-out goes into the ledger with its lines, its steps, its pictures and its numbers.

Two runs of the floor watching the same shopper (the page and a replay, say) charge once: the run that charges
takes the visit first, and the other finds it settled. A registered shopper the floor checks in without the gate
gets a new visit only if their last one was not settled too recently: within 15 minutes for footage, as a replay of
a clip is the same walk-out again. A live camera films the present, so there a visit settled before the run started
is over and they are back for more; only one settled while the run was already watching, or in the last 90 seconds
(the page reloaded straight after the charge, items still in hand), is the same walk-out.

## The catalogue screen

`web/catalogue.html`, `checkout/clips.py`. Two views.

**Products.** One row per product, grouped by category and section: up to four pictures, name, maker, category,
section, the camera phrase, the price. A picture can be added from a file with "+"; clicking one removes it. Below
the list: the shoppers' own things that are never billed, and the seconds out of view that count as walking out.
Changes apply on the next Start.

**From footage.** For working out what a product is called, and which phrase SAM finds it by.

- **Hold to camera.** Hold the product up to the camera and turn it slowly. About every 0.7 s a frame goes to SAM,
  which outlines the item in hand (green once it is sure). After ten clear views, 30 seconds, or Done, the server
  keeps the four views that look most different from each other (picture vectors, below) as the product's
  pictures, asks the vision model what the item is, and tries its phrases on the kept views. The form comes back
  filled in: name, maker, category and section (the shop's own where one fits), what it looks like, the phrase that
  outlined the item best, and the pictures. Only the price is left to type. A picture can be left out by clicking it.
- **A clip.** Import a video (or pick a sample). It is cut into up to 120 frames. Scroll through them; SAM outlines
  what it finds on each (generic phrases, the shop's own, and any typed under "Also look for"). Click an outline,
  or drag a box round something the outlines missed (SAM then outlines the object in that box). Then:
  - *Describe it*: the vision model names the whole object as sold, with its packaging (two tissue rolls wrapped
    together are a pack of two), and suggests three or four phrases.
  - *Test the phrases*: each phrase is asked on that frame and two either side. A phrase counts when its outline
    covers the picked object (overlap of 0.6 or more with its box). The table shows which phrases do, their score,
    on how many frames, and how many other things each also outlines. On the freezer sample, of "frozen food pack",
    "frozen meal", "packaged food" and "food package", only "packaged food" outlined the pack in hand.
  - *Find more like it*: SAM's visual prompt, with the picked box as the example, on that frame.
  - *Add as a new product*, or add the picture to an existing one.

A camera capture is kept as a clip too, so it can be gone through frame by frame afterwards. Clips and the answers
SAM gave for them are kept in `data/clips/` and `data/clip-cache/`, on this computer only. Pictures are kept in
`shops/<shop>/examples/` and listed under each product's `examples` in the shop's file.

### What the pictures are for

SAM 3's visual prompts are boxes drawn *in the image being segmented*: it cannot be handed a photo from elsewhere.
Pasting a product photo beside a frame and boxing it there was tried and found nothing reliable (scores around 0.33,
on the wrong things), so the floor still finds products by phrase. The pictures are used in three other ways:

- **Likeness check** (`checkout/likeness.py`). Every picture is turned into a vector once (DINOv2 on the SAM
  endpoint, `/v1/embed`), and so is every item a shopper picks up, cut out by its box. The cosine says how alike
  they look. At 0.55 or more, and 0.05 ahead of the next product, it helps name the item: once per line, and only
  when the second model's look at the same moment sees the item held. It is never counted as the item being held,
  so it cannot outvote "still on the shelf": a pick that never left the shelf looks just like its product. On the
  freezer sample the pack in hand scored 0.71 against its picture and was named by it. A pick that resembles no
  picture gets no vote. The floor draws a box round each item in hand with the likeness after its name.
- **Reference pictures for the second model.** When it is asked what a picked item is, it is also shown the first
  picture of each candidate product.
- **The catalogue** shows them, so whoever keeps it can see what each phrase is meant to find.

## The dashboard and the ledger

`checkout/ledger.py`, `web/dashboard.html`. For the shop, not for shoppers.

- **Overview:** sales, transactions, items sold, average basket and registered shoppers; sales by day for two
  weeks; the products that sell most; the latest transactions.
- **Transactions:** every walk-out that had something on the bag. Opening one shows:
  - *How this bill was made*: the steps in order with the time of day. The gate opening the visit; on the floor,
    being recognised (with the face match), each item taken, each title read in the hands, anything put back; at
    the door, what counted as walking out, the verdict of the check before charging on every line, the charge, the
    invoice, the email, and any refund.
  - *The pictures behind it*: the gate camera when it let them in, the floor camera when it recognised them, the
    frame in which each item left the shelf (item outlined), and the pictures the check before charging was shown.
  - *The bill*: each line with the evidence it was charged on, or why it was held back.
  - *Under the hood*: frames analysed and the model's time per frame, seconds in view, the face match, how often
    the second model read their hands, the walk-out rule that applied, seconds from walking out to the payment,
    the PayPal capture and order.
- **Charging a held-back line:** when a shopper left with something that was kept off the bill (never identified,
  or not confirmed), the line carries a list of the catalogue and "Charge it". Whoever has looked at the pictures
  says which item it was; it is charged to the shopper's saved method as a transaction of its own, with its own
  invoice and email, and the walk-out links to it. A line can be charged this way once. Not for rehearsals.
- **Refunds:** on a paid transaction, "Refund" on a line gives its price back and "Refund the whole bill" what is
  left, through PayPal (`POST /v2/payments/captures/{id}/refund`, with a `PayPal-Request-Id`). The ledger decides how
  much can still go back, so a line is never refunded twice. A refunded line no longer counts as an item sold.
- **Customers:** each registered shopper with how they pay, what they buy most, what they have spent and when
  they last bought. Their name opens their transactions.
- **Rehearsals** are kept with the same steps and pictures, so a bill can be checked before a real run. They are
  hidden until "Show rehearsals" is ticked and never count as sales.

Sales are counted in the default shop's currency; sales in another currency (the Stripe version charged in AED)
are listed but not added in, and cannot be refunded from here.

Lines that are doubtful are never charged: not in the catalogue, a book whose title could not be read, under 50 %
confidence with no confirmation, disputed by the second model, or not confirmed by the check before charging. They
stay on the bag marked as not charged, with the reason, and the transaction shows as **Needs review** on the
dashboard until someone looks at the pictures and either charges the line (**Charge**) or marks it **Not taken**.
`tools/tidy_ledger_copy.py` rewrites the wording stored in an older `data/checkout.db` in the dashboard's current
terms.

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

Besides phrases, `/v1/segment` takes `exemplars`: boxes round examples in the same picture, SAM 3's visual prompt
(one more detector pass each, about 20 ms). `/v1/embed` turns pictures into DINOv2 vectors for the likeness check. A
request with neither is answered exactly as before: on the same frame and phrases, the new deployment and the old
gave identical detections and scores.

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
     outlines missed that pick, which happens when a book slides straight into a stack). One look is enough when
     a pick of theirs already carries that title from the shelf it was taken from: shelf and hands agree;
   - a title that drops out of **three looks in a row**, with the hands in view, comes off again: it was put down.
     Not while that line's own outline is still in their hand, and not while the look sees as many things in the
     hands that it cannot read as there are titles it did not list: a book turned away or lying under another one
     has not been put down;
   - **one line per title per shopper**. A second line with the same title is the same book seen twice, and is never
     charged.
3. **At the door: kept, or back on the shelf?** A leaving shopper's last six pictures, taken about half a second
   apart, are read again all at once, and settle two things before the check before charging runs.
   - *Which titles.* A pick that still has no confirmed title (taken in their last seconds in view, or never
     legible at the time) gets one on the same two-readings rule, applied to what those pictures show.
   - *Still with them.* A title on the bill that none of those pictures shows in their hands, while at least two of
     them do show their hands and nothing in them is left unread, went back on a shelf at a moment the outlines
     missed: it comes off the charge, marked as put back. Things in their hands that no picture could read get the
     benefit of the doubt, first for picks without a title, then for the titles read most recently.

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

### Which product it is

SAM's phrases are the first guess, and a phrase does not stay on its own product. With a tissue pack in one hand and
a power bank in the other, "wireless power bank" outlined both (0.84 and 0.36) and "tissue pack" scored only 0.30
on its own pack; held edge-on the power bank was "mobile phone" at 0.87. So:

- **A faint outline beside the phrase's own strong one counts for less.** A vote is weighed by how close it comes to
  that phrase's best in the frame. Two of one product score alike and keep their name.
- **The looks can overrule the phrase.** When the second model names another product than the phrase did, and one
  more thing agrees (a second look, or the likeness to the pictures), the line takes that product. The likeness alone
  never renames anything: it compares with billable products only, so it would turn a shopper's phone into the power
  bank it resembles. Books keep their own rules: which title rests on reading it.
- **A name the looks dispute is not billed on the phrase's word.** The line is held for review ("seen in hand as …").
- **The final check counts.** It reports how many of each product it sees in hand at once; a product on the bill
  more often than that is held back for the extra lines. One power bank in hand never pays for two.

### A closer look at far hands

SAM finds an object less well the smaller it is. With the shopper a few steps back, a tissue pack at arm's length
was not outlined at all; the frame cut down to the hands and blown up 2-3 times gave it 0.45-0.59. So while the hands
are small in the frame, each frame also goes to SAM cut down to them, round where they were a moment ago, so both
calls go out together (`checkout/zoom.py`): one cut-out round both hands, or one each when the arms are held out
apart. What it finds joins the frame's own outlines. Not in a bookshop, whose path reads the whole frame.

## Files

| Path | What |
|---|---|
| `checkout/engine.py` | masks to picks, returns and carts |
| `checkout/people.py` | shopper tracking and re-identification by clothing |
| `checkout/gate.py` | the gate: recognise or register, then open a visit |
| `checkout/identity.py` | profiles, faces, gate admissions and visit history in SQLite |
| `checkout/ledger.py` | every walk-out with its steps, pictures and refunds; what the dashboard reads |
| `checkout/vectors.py` | face embeddings in Postgres with pgvector |
| `checkout/payments.py` | PayPal sandbox: save a method in the vault, charge it, write the invoice |
| `checkout/likeness.py` | how alike a picked item looks to the catalogue's pictures of a product |
| `checkout/staff.py` | staff sign-in: the staff code for a token, checked against its salted hash |
| `checkout/paths.py` | where the data lives (`DATA_DIR`) |
| `checkout/zoom.py` | the closer look round a shopper's hands while they are far from the camera |
| `checkout/clips.py` | the catalogue screen's footage: clips cut into frames, camera captures, ranking phrases |
| `checkout/reader.py` | local OCR and matching the text to catalogue titles |
| `checkout/verify.py` | the vision language model: second looks, titles in the hands, the check before charging |
| `checkout/mailer.py` | the invoice email through Resend |
| `modal_app/sam_checkout.py` | the batched SAM 3.1 endpoint on Modal |
| `shops/` | one catalogue per shop |
| `checkout/pipeline.py` | one visit: frame order, plus the jobs that run beside it |
| `checkout/server.py` | the local web server and WebSocket |
| `web/` | the screens: `index.html` + `app.js` (shop floor), `catalogue.html` + `catalogue.js`, `gate.html` + `gate.js` (gate), `dashboard.html` + `dashboard.js`, the staff header and settings in `shell.js`, the shared wording in `words.js`, one `styles.css` |
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
anything but the sandbox, and that a refund goes back once. `tests/test_last_check.py` covers the check before
charging. `tests/test_zoom.py` covers the closer look round far hands. `tests/test_staff.py` covers staff sign-in and walks every route to check that only the gate is public. `tests/test_catalogue.py` covers the catalogue screen: pictures kept through a save, clips and camera captures cut
into frames, phrases ranked by whether they outline the picked object, the most different views kept, and the
likeness check (votes only when clear, off without `/v1/embed`). `tests/test_ledger.py` covers the ledger: the steps, pictures and numbers kept for a walk-out, rehearsals
kept apart from sales, refunds at most once, a held-back line charged by hand once, two runs charging once, and a
shopper recognised again keeping one bag.

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
- **Books lean on the second model.** Titles are billed from what DeepSeek reads in the hands. A book whose title
  never faces the camera, or is too small in the picture to read, is not identified and so not charged; the bag
  and the dashboard say so, with the pictures. A camera that sees the shopper side-on from across a room reads few
  titles: it should face the shopper, as in the sample clip. Two copies of one title in the same hands are billed
  once.
- **One real book clip so far.** One shopper, three books, a moving camera. Several shoppers at one bookshelf, and
  putting a book back, are covered only by synthetic tests.
- **Products are found by phrase.** A vague phrase scores low; the catalogue screen's phrase test shows which phrase
  SAM actually outlines a product by, and the pictures add a likeness check, but SAM itself cannot be shown a
  product photo from elsewhere. A real shop would still give each product its own trained signature.
- **Two items in one hand-full are tested on one picture.** The naming rules and the final check's count were
  worked out on one frame of a tissue pack and a power bank, and with synthetic tests. Products that a phrase cannot
  tell apart and that look alike on camera will still be held for review rather than billed.
- **The closer look costs a SAM call.** While the hands are small, a frame takes two or three calls instead of one.
  On the supermarket sample, with it on, answers came back about 1.2 s after each frame (median).
- **The likeness check is tuned on one clip.** 0.55 and the 0.05 lead were set from a handful of pictures; a
  picture taken from the same clip it is then tested on is the easy case. Products that look alike (two flavours
  of one brand) still need the second model to read them apart.
- **One camera.** An item put in a trolley parked in front of a shelf looks like an item put back on it. And
  "walked out" is only "out of this camera's view for a few seconds": a real shop would use a camera at the door.
- **Refunds stop at the payment.** The PayPal invoice written at the door still reads as paid after a refund, and
  no email is sent about it.
- **One staff code for everyone.** Staff sign in with a single shared code, not personal accounts; there is no
  record of which member of staff did what. Changing `STAFF_CODE` does not end sessions already open; changing
  `JWT_SECRET` does.
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

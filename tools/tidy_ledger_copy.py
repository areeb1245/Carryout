"""One-off: rewrite the wording stored in data/checkout.db (bill notes, picture captions, the timeline kept with each
transaction) in the dashboard's current terms. Only text changes; amounts, statuses and pictures are untouched.
Safe to run twice. Usage: uv run python tools/tidy_ledger_copy.py [path/to/checkout.db]"""
import json
import re
import sqlite3
import sys

RULES = [
    (r'^picture ([A-Z]), read "(.*)" in their hands$', r'In hand in picture \1: "\2"'),
    (r"^picture ([A-Z]), seen in their hands$", r"In hand in picture \1"),
    (r"^in their hands in picture ([A-Z]), put down in picture ([A-Z])$", r"In hand in picture \1, back on the shelf in picture \2"),
    (r"^The check before charging could not confirm it: ", "Not confirmed at checkout: "),
    (r"^The check before charging saw it put back: ", "Returned to the shelf: "),
    (r"^Not identified: its title was never read in their hands$", "Item not identified"),
    (r"^Charged by the shop afterwards, from the pictures of transaction (\d+)$", r"Charged after review of transaction \1"),
    (r"^(Never confirmed in their hands|Not confirmed in hand) \(\d+% sure\)$", "Not confirmed in hand"),
    (r"^Never confirmed in their hands", "Not confirmed in hand"),
    (r"^The second model saw no item in the hand: ", "No item seen in hand: "),
    (r"^It carried this name from the shelf, but the title was never read in their hands$", "Identified on the shelf only, not confirmed in hand"),
    (r"^Only (\d+)% sure$", r"Low confidence (\1%)"),
    (r"^The same title as another line: one item seen twice$", "Duplicate of another line"),
    (r"^Read in their hands as they left, on two looks: ", "Read in hand at exit: "),
    (r"^Read in their hands on two looks: ", "Read in hand twice: "),
    (r"^Named on the shelf, then read in their hands: ", "Identified on the shelf, confirmed in hand: "),
    (r"^not in their hands in any of the last (\d+) pictures of them$", r"Not in hand in any of the last \1 pictures"),
    (r"^the check before charging got no answer$", "The final check got no answer"),
    (r"^no picture of them carrying it was kept$", "No picture of it in hand was kept"),
    (r"^the check gave no answer for this line$", "No answer for this line"),
    (r"^no picture shows it in their hands$", "Not visible in hand in any picture"),
    (r", and the second model agrees$", ", confirmed by the vision model"),
    (r"^Second model read it: ", "Vision model: "),
    # picture captions
    (r"^The gate camera when it let them in \((\d+)% face match\)$", r"Entry gate, \1% face match"),
    (r"^The gate camera when it let them in, as they registered$", "Entry gate, at registration"),
    (r"^The shop floor camera when it recognised them \((\d+)% face match\)$", r"Store camera, shopper recognised, \1% face match"),
    (r"^The shop floor camera when it recognised them$", "Store camera, shopper recognised"),
    (r"^An item not told apart yet", "Unidentified item"),
    (r", leaving the shelf in their hand \(outlined\)$", ", taken from the shelf"),
    (r", first seen in their hand \(outlined\)$", ", first seen in hand"),
    (r", in their hand as they arrived \(outlined\)$", ", in hand on arrival"),
    (r", handed to them \(outlined\)$", ", handed over"),
    (r"^Picture ([A-Z]) the check before charging was shown: ", r"Final check, picture \1: "),
    (r": the look that read (.*) in their hands$", r": \1 in hand"),
    (r": the latest close view of their hands$", ": last close view of hands"),
    (r": the whole scene the last time they were seen$", ": last full view"),
]


def tidy(text: str) -> str:
    for pattern, new in RULES:
        text = re.sub(pattern, new, text)
    return text


def walk(value):
    if isinstance(value, str):
        return tidy(value)
    if isinstance(value, list):
        return [walk(v) for v in value]
    if isinstance(value, dict):
        return {k: walk(v) for k, v in value.items()}
    return value


def main(path: str) -> None:
    db = sqlite3.connect(path)
    changed = 0
    for table, column in (("purchases", "note"), ("proofs", "caption")):
        for rid, text in db.execute(f"SELECT id, {column} FROM {table} WHERE {column} IS NOT NULL").fetchall():
            if tidy(text) != text:
                db.execute(f"UPDATE {table} SET {column} = ? WHERE id = ?", (tidy(text), rid))
                changed += 1
    for vid, detail in db.execute("SELECT id, detail FROM visits WHERE detail IS NOT NULL").fetchall():
        new = json.dumps(walk(json.loads(detail)), separators=(",", ":"))
        if new != detail:
            db.execute("UPDATE visits SET detail = ? WHERE id = ?", (new, vid))
            changed += 1
    db.commit()
    print(f"{changed} rows reworded")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/checkout.db")

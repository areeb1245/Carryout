"""
This month's Modal bill, and a stop when it passes a limit.

    python tools/modal_spend.py                      this month's cost, by app and resource
    python tools/modal_spend.py --stop-above 125     ...and stop the SAM app (sam3-supermarket) when it is over USD 125

The figures come from `modal billing report` (Modal 1.3 or later; run through `uvx` so the project needs no Modal
install). They cover the whole workspace, not only this project. The hard limit belongs in Modal itself: Settings,
Usage & Billing, workspace budget. This is the check you can run yourself, or on a schedule.

A stopped app answers nothing until `modal deploy modal_app/sam_checkout.py` is run again.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys

APP = "sam3-supermarket"
MODAL = ["uvx", "--from", "modal>=1.3", "modal"]


def report() -> list[dict]:
    out = subprocess.run([*MODAL, "billing", "report", "--for", "this month", "--show-resources", "--json"], capture_output=True, text=True)
    if out.returncode != 0:
        sys.exit(f"modal billing report failed: {out.stderr.strip()[-400:]}")
    return json.loads(out.stdout)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stop-above", type=float, help="stop the SAM app when this month's total passes this many USD")
    args = ap.parse_args()

    rows = report()
    total = sum(float(r["cost"]) for r in rows)
    by: dict[tuple[str, str], float] = {}
    for r in rows:
        key = (r.get("description") or r.get("object_id") or "?", r.get("resource") or "")
        by[key] = by.get(key, 0.0) + float(r["cost"])
    print(f"Modal this month: USD {total:.2f} (full days only; today is added tomorrow)")
    for (app, resource), cost in sorted(by.items(), key=lambda x: -x[1]):
        if cost >= 0.01:
            print(f"  {cost:8.2f}  {app}  {resource}")

    if args.stop_above is not None:
        if total > args.stop_above:
            print(f"Over USD {args.stop_above:.2f}: stopping {APP}.")
            subprocess.run([*MODAL, "app", "stop", "--yes", APP], check=False)
        else:
            print(f"Under USD {args.stop_above:.2f}: {APP} keeps running.")


if __name__ == "__main__":
    main()

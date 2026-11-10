"""
Where things are kept on disk.

On this computer everything sits in the project folder: `data/` for what the shop writes (the database, clips,
caches) and `shops/` for the catalogues. A hosted copy (Render) keeps both on a persistent disk instead, named by
`DATA_DIR`: the project folder there is replaced on every deploy. The first start on an empty disk copies the
catalogues and their pictures from the repository; after that the disk's copy is the one that is edited, so later
changes to `shops/` in the repository do not reach it.
"""
from __future__ import annotations

import shutil
from pathlib import Path

from .sam_client import ROOT, load_env

_env = load_env()
_data = _env.get("DATA_DIR", "").strip()

DATA = Path(_data) if _data else ROOT / "data"
SHOPS = DATA / "shops" if _data else ROOT / "shops"
MODELS = Path(_env["MODELS_DIR"]) if _env.get("MODELS_DIR") else ROOT / "models"
DATABASE = DATA / "checkout.db"


def prepare() -> None:
    """Make the folders, and give an empty disk the repository's catalogues."""
    DATA.mkdir(parents=True, exist_ok=True)
    if SHOPS != ROOT / "shops" and not SHOPS.exists():
        shutil.copytree(ROOT / "shops", SHOPS)

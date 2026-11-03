"""Lets the Zillow scripts reuse the King County code, so both datasets go through one copy of every check and model.

Importing this module puts king_county/kc_features and src/ on the import path. Unless MODEL_TABLE_DIR is already
set, it also points the model scripts at the Zillow table, so a Zillow run can never write into the King County
results.
"""
# AI was used to help write this code, but the resulting code was reviewed and edited by a human.

import os
import sys
from pathlib import Path

ZILLOW_DIR = Path(__file__).resolve().parent
PROJECT = ZILLOW_DIR.parent
KING_COUNTY_CODE = PROJECT / "king_county" / "kc_features"
ZILLOW_CODE_TABLES = PROJECT / "src"

os.environ.setdefault("MODEL_TABLE_DIR", str(ZILLOW_DIR / "out" / "model_table"))
TABLE_DIR = Path(os.environ["MODEL_TABLE_DIR"])
CHECKS_DIR = TABLE_DIR.parent / "checks"

for code_folder in (KING_COUNTY_CODE, ZILLOW_CODE_TABLES):
    if str(code_folder) not in sys.path:
        sys.path.insert(0, str(code_folder))

"""Common setup for the Zillow scripts.

Importing this module puts src/ (the Zillow code tables) on the import path. Unless MODEL_TABLE_DIR is already set,
it also points the model scripts at the Zillow table in out/model_table.
"""
# AI was used to help write this code, but the resulting code was reviewed and edited by a human.

import os
import sys
from pathlib import Path

ZILLOW_DIR = Path(__file__).resolve().parent
PROJECT = ZILLOW_DIR.parent
ZILLOW_CODE_TABLES = PROJECT / "src"

os.environ.setdefault("MODEL_TABLE_DIR", str(ZILLOW_DIR / "out" / "model_table"))
TABLE_DIR = Path(os.environ["MODEL_TABLE_DIR"])
CHECKS_DIR = TABLE_DIR.parent / "checks"

if str(ZILLOW_CODE_TABLES) not in sys.path:
    sys.path.insert(0, str(ZILLOW_CODE_TABLES))

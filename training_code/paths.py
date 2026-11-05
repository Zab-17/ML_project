"""Puts zillow/ on the import path, so every model file here can reuse the shared table loaders and scores."""
# AI was used to help write this code, but the resulting code was reviewed and edited by a human.

import sys
from pathlib import Path

TRAINING_CODE = Path(__file__).resolve().parent
ZILLOW_CODE = TRAINING_CODE.parent / "zillow"
RESULTS_DIR = TRAINING_CODE / "results"

if str(ZILLOW_CODE) not in sys.path:
    sys.path.insert(0, str(ZILLOW_CODE))

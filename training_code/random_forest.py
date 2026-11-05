"""Random forest, regression (Lecture 10).

RandomForestRegressor: TREES (50) CART trees on bootstrap samples, at least FOREST_SMALLEST_LEAF (5) homes
per leaf. Searched: share of columns per split in FOREST_COLUMN_SHARES (square root, 10%, 33%).
Defined in zillow/zillow_models.py; run through the four-step runner in zillow/slide_models.py (search on five
folds, winner by lowest validation MSE of the log value, one final fit, one test score).
Results: training_code/results/random_forest.md

Run: python3 training_code/random_forest.py   (Finished steps are skipped, so running this file again does not retrain anything already trained.)
"""
# AI was used to help write this code, but the resulting code was reviewed and edited by a human.

import paths  # noqa: F401  (puts zillow/ on the import path)
from slide_models import run
from zillow_models import MODELS

if __name__ == "__main__":
    run(MODELS, ["random forest"])

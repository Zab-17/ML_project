"""Bagging, regression form (Lecture 13 teaches it as a classifier).

RandomForestRegressor with every column open at every split: TREES (50) trees on bootstrap samples,
averaged. No search.
Defined in zillow/zillow_models.py; run through the four-step runner in zillow/slide_models.py (search on five
folds, winner by lowest validation MSE of the log value, one final fit, one test score).
Results: training_code/results/bagging.md

Run: python3 training_code/bagging.py   (Finished steps are skipped, so running this file again does not retrain anything already trained.)
"""
# AI was used to help write this code, but the resulting code was reviewed and edited by a human.

import paths  # noqa: F401  (puts zillow/ on the import path)
from slide_models import run
from zillow_models import MODELS

if __name__ == "__main__":
    run(MODELS, ["bagging"])

"""Support vector machine, regression form (Lecture 12 teaches the classifier: see linear_svm_classifier.py).

LinearSVR: misses smaller than the margin epsilon cost nothing, larger ones cost C times their size.
Searched: C in MARGIN_PENALTIES (0.01, 1) x epsilon in MARGIN_WIDTHS (0, 0.05).
Defined in zillow/zillow_models.py; run through the four-step runner in zillow/slide_models.py (search on five
folds, winner by lowest validation MSE of the log value, one final fit, one test score).
Results: training_code/results/svm_regression.md

Run: python3 training_code/svm_regression.py   (Finished steps are skipped, so running this file again does not retrain anything already trained.)
"""
# AI was used to help write this code, but the resulting code was reviewed and edited by a human.

import paths  # noqa: F401  (puts zillow/ on the import path)
from slide_models import run
from zillow_models import MODELS

if __name__ == "__main__":
    run(MODELS, ["support vector machine"])

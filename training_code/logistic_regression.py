"""Logistic regression (Lecture 5), on the five price bands.

LogisticRegression: softmax over the five bands, L2 penalty. Searched: C in LOGISTIC_PENALTIES
(0.01, 1, 100), at most LOGISTIC_MAX_PASSES (300) passes of the L-BFGS solver.
Defined in zillow/band_models.py: search on five folds, winner by highest validation accuracy, one final fit,
one test score. Results: training_code/results/logistic_regression.md

Run: python3 training_code/logistic_regression.py   (Finished steps are skipped, so running this file again does not retrain anything already trained.)
"""
# AI was used to help write this code, but the resulting code was reviewed and edited by a human.

import paths  # noqa: F401  (puts zillow/ on the import path)
import band_models

if __name__ == "__main__":
    band_models.main(["logistic regression"])

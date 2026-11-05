"""Naive Bayes (Lecture 11), on the five price bands.

MixedNaiveBayes: a normal distribution per band for each number column and a yes/no probability per band
for each 0/1 column. Searched: Laplace smoothing in SMOOTHING (0.1, 1, 10).
Defined in zillow/band_models.py: search on five folds, winner by highest validation accuracy, one final fit,
one test score. Results: training_code/results/naive_bayes.md

Run: python3 training_code/naive_bayes.py   (Finished steps are skipped, so running this file again does not retrain anything already trained.)
"""
# AI was used to help write this code, but the resulting code was reviewed and edited by a human.

import paths  # noqa: F401  (puts zillow/ on the import path)
import band_models

if __name__ == "__main__":
    band_models.main(["naive Bayes"])

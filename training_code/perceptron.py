"""Perceptron (Lecture 6), on the five price bands.

Perceptron: the perceptron rule, one perceptron per band against the rest, no penalty. Searched: passes
in PERCEPTRON_PASSES (5, 20).
Defined in zillow/band_models.py: search on five folds, winner by highest validation accuracy, one final fit,
one test score. Results: training_code/results/perceptron.md

Run: python3 training_code/perceptron.py   (Finished steps are skipped, so running this file again does not retrain anything already trained.)
"""
# AI was used to help write this code, but the resulting code was reviewed and edited by a human.

import paths  # noqa: F401  (puts zillow/ on the import path)
import band_models

if __name__ == "__main__":
    band_models.main(["perceptron"])

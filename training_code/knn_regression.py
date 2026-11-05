"""K-nearest neighbours, regression form (Lecture 3 teaches the classifier: see knn_classifier.py).

The average log value of the k nearest training homes. Searched: k in NEIGHBOUR_COUNTS (1 to 50) x
Euclidean or cosine distance. Exact search on the graphics processor in blocks (zillow/zillow_neighbours.py).
Defined in zillow/zillow_models.py; run through the four-step runner in zillow/slide_models.py (search on five
folds, winner by lowest validation MSE of the log value, one final fit, one test score).
Results: training_code/results/knn_regression.md

Run: python3 training_code/knn_regression.py   (Finished steps are skipped, so running this file again does not retrain anything already trained.)
"""
# AI was used to help write this code, but the resulting code was reviewed and edited by a human.

import paths  # noqa: F401  (puts zillow/ on the import path)
from slide_models import run
from zillow_models import MODELS

if __name__ == "__main__":
    run(MODELS, ["k-nearest neighbours"])

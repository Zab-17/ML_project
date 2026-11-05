"""Hierarchical clustering (Lecture 14).

Ward agglomerative clustering on HIERARCHY_SAMPLE (10,000) sampled homes; every home joins the nearest
cluster centre and is priced at its cluster's median. Searched: number of clusters in CLUSTER_COUNTS.
Defined in zillow/zillow_models.py; run through the four-step runner in zillow/slide_models.py (search on five
folds, winner by lowest validation MSE of the log value, one final fit, one test score).
Results: training_code/results/hierarchical_clustering.md

Run: python3 training_code/hierarchical_clustering.py   (Finished steps are skipped, so running this file again does not retrain anything already trained.)
"""
# AI was used to help write this code, but the resulting code was reviewed and edited by a human.

import paths  # noqa: F401  (puts zillow/ on the import path)
from slide_models import run
from zillow_models import MODELS

if __name__ == "__main__":
    run(MODELS, ["hierarchical clustering"])

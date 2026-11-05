"""K-means clustering (Lecture 14).

MiniBatchKMeans on the 606 columns, without the label; a home is priced at the median log value of its
cluster. Searched: number of clusters in CLUSTER_COUNTS (50, 200, 1,000).
Defined in zillow/zillow_models.py; run through the four-step runner in zillow/slide_models.py (search on five
folds, winner by lowest validation MSE of the log value, one final fit, one test score).
Results: training_code/results/kmeans_clustering.md

Run: python3 training_code/kmeans_clustering.py   (Finished steps are skipped, so running this file again does not retrain anything already trained.)
"""
# AI was used to help write this code, but the resulting code was reviewed and edited by a human.

import paths  # noqa: F401  (puts zillow/ on the import path)
from slide_models import run
from zillow_models import MODELS

if __name__ == "__main__":
    run(MODELS, ["k-means clustering"])

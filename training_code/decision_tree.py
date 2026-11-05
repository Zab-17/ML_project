"""Decision tree, CART regression (Lecture 10).

DecisionTreeRegressor: binary splits on squared error. Searched: smallest leaf in SMALLEST_LEAVES
(1, 5, 20, 50, 200 homes) instead of pruning.
Defined in zillow/zillow_models.py; run through the four-step runner in zillow/slide_models.py (search on five
folds, winner by lowest validation MSE of the log value, one final fit, one test score).
Results: training_code/results/decision_tree.md

Run: python3 training_code/decision_tree.py   (Finished steps are skipped, so running this file again does not retrain anything already trained.)
"""
# AI was used to help write this code, but the resulting code was reviewed and edited by a human.

import paths  # noqa: F401  (puts zillow/ on the import path)
from slide_models import run
from zillow_models import MODELS

if __name__ == "__main__":
    run(MODELS, ["decision tree"])

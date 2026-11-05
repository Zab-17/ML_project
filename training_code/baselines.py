"""Average and median baselines (no lecture: the floor every model must beat).

DummyRegressor(mean) and DummyRegressor(median): one number for every home. Validated on fold 0 by
zillow/train_models.py and scored on the test homes by zillow/final_training.py.
Results: training_code/results/baselines.md

Everything below the imports is copied unchanged from the zillow/ modules that ran this training on
1-3 November 2026 (each block is marked with its source file); only the table loaders, the score
functions and the project settings are imported.

Run: python3 training_code/baselines.py
"""
# AI was used to help write this code, but the resulting code was reviewed and edited by a human.

from pathlib import Path
import os

from scipy import sparse
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import torch

import paths  # noqa: F401  (puts zillow/ on the import path)
import shared_code  # noqa: F401  (points the loaders at the Zillow table)


# ----- copied from zillow/train_models.py -----


TABLE_DIR = Path(os.environ.get("MODEL_TABLE_DIR", Path(__file__).resolve().parent / "out" / "model_table"))
RESULTS = TABLE_DIR.parent / "model_results.csv"
CLOSE_SHARES = (0.05, 0.10, 0.20)   # a prediction within this share of the sale price counts as close


def comparison_models(training_log_price):
    """The two baselines and linear regression, not yet fitted. The average baseline predicts the average
    training price in dollars, so its constant is the log of that average, not the average log."""
    average_price = np.exp(training_log_price).mean()
    return {"average baseline": DummyRegressor(strategy="constant", constant=np.log(average_price)),
            "median baseline": DummyRegressor(strategy="median"), "linear regression": LinearRegression()}


def scores(log_price, predicted_log_price):
    """Every metric for one set of predictions; the model works in log price, the dollar metrics undo the log."""
    price, predicted = np.exp(log_price), np.exp(predicted_log_price)
    return {**log_scores(log_price, predicted_log_price), **dollar_scores(price, predicted)}


def log_scores(actual, predicted):
    error = predicted - actual
    return {"R2 (log)": r_squared(actual, predicted), "MSE (log)": (error ** 2).mean(),
            "RMSE (log)": np.sqrt((error ** 2).mean()), "MAE (log)": np.abs(error).mean()}


def dollar_scores(price, predicted):
    miss = np.abs(predicted - price)
    share_off = miss / price
    close = {f"within {share:.0%}": (share_off <= share).mean() * 100 for share in CLOSE_SHARES}
    return {"R2 ($)": r_squared(price, predicted), "RMSE ($)": np.sqrt(((predicted - price) ** 2).mean()),
            "MAE ($)": miss.mean(), "median $ off": np.median(miss), "MAPE (%)": share_off.mean() * 100,
            "median % off": np.median(share_off) * 100, **close}


def r_squared(actual, predicted):
    return 1 - ((predicted - actual) ** 2).sum() / ((actual - actual.mean()) ** 2).sum()


# ----- copied from zillow/final_training.py -----


FINAL_DIR = TABLE_DIR.parent / "final_model"


def test_predictions(model, matrix, label, test_matrix):
    """Log-price predictions for the test sales: the network, then linear regression and the baselines fitted on all
    of matrix."""
    predictions = {"neural network": predict(model, test_matrix, np.arange(test_matrix.shape[0]))}
    for name, comparison in comparison_models(label).items():
        predictions[name] = comparison.fit(matrix, label).predict(test_matrix)
    return predictions


# ----- copied from zillow/neural_search.py -----


SCORE_BATCH = 8192
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def predict(model, matrix, rows):
    model.eval()
    with torch.no_grad():
        parts = [model(to_device(matrix[rows[start:start + SCORE_BATCH]])).squeeze(1).cpu().numpy()
                 for start in range(0, len(rows), SCORE_BATCH)]
    return np.concatenate(parts).astype(np.float64)


def to_device(rows):
    return torch.from_numpy(rows.toarray().astype(np.float32)).to(DEVICE)


# ----- the main steps, copied from each module's main() and renamed -----
def run_fold_zero_validation():
    matrix = sparse.load_npz(TABLE_DIR / "train_matrix.npz").tocsr()
    label = np.load(TABLE_DIR / "train_label.npy")
    validation = np.load(TABLE_DIR / "validation_homes.npy")
    models = comparison_models(label[~validation])
    results = []
    for name, model in models.items():
        model.fit(matrix[~validation], label[~validation])
        for part, rows in [("training homes", ~validation), ("validation homes", validation)]:
            results.append({"model": name, "scored on": part, **scores(label[rows], model.predict(matrix[rows]))})
    table = pd.DataFrame(results)
    table.to_csv(RESULTS, index=False)
    print(f"fitted on {(~validation).sum():,} sales, validated on {validation.sum():,} sales of homes never seen in fitting")
    print(table.set_index(["model", "scored on"]).T.round(4).to_string())


TEST_RESULTS = FINAL_DIR / "test_results.csv"


def main():
    """zillow/final_training.py has no skip: running it again would retrain the network and change the reported
    scores. So the steps run only when their test results are missing."""
    if TEST_RESULTS.exists():
        print("Already trained and tested; the scores are in training_code/results/baselines.md. Nothing was rerun.")
        return
    run_fold_zero_validation()
    print("The test score is fitted with the network: run training_code/neural_network.py.")


if __name__ == "__main__":
    main()

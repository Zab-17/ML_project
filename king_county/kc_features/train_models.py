"""First models on the King County table, scored on held-out training homes; the test sales stay sealed.

- Average baseline: every sale is predicted at the average training price (in dollars).
- Median baseline: every sale is predicted at the median training price.
- Linear regression: ordinary least squares on every column.

Run: python3 train_models.py   (after pipeline.py)
"""
# AI was used to help write this code, but the resulting code was reviewed and edited by a human.

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression

TABLE_DIR = Path(__file__).resolve().parent / "out" / "model_table"
RESULTS = TABLE_DIR.parent / "model_results.csv"
CLOSE_SHARES = (0.05, 0.10, 0.20)   # a prediction within this share of the sale price counts as close


def main():
    matrix = sparse.load_npz(TABLE_DIR / "train_matrix.npz").tocsr()
    label = np.load(TABLE_DIR / "train_label.npy")
    validation = np.load(TABLE_DIR / "validation_homes.npy")
    average_price = np.exp(label[~validation]).mean()
    models = {"average baseline": DummyRegressor(strategy="constant", constant=np.log(average_price)),
              "median baseline": DummyRegressor(strategy="median"), "linear regression": LinearRegression()}
    results = []
    for name, model in models.items():
        model.fit(matrix[~validation], label[~validation])
        for part, rows in [("training homes", ~validation), ("validation homes", validation)]:
            results.append({"model": name, "scored on": part, **scores(label[rows], model.predict(matrix[rows]))})
    table = pd.DataFrame(results)
    table.to_csv(RESULTS, index=False)
    print(f"fitted on {(~validation).sum():,} sales, validated on {validation.sum():,} sales of homes never seen in fitting")
    print(table.set_index(["model", "scored on"]).T.round(4).to_string())


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

if __name__ == "__main__":
    main()

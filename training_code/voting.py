"""Voting, regression form (Lecture 13 teaches voting between classifiers).

The average of the log predictions of the three regression models with the lowest average validation MSE:
the neural network, the random forest and k-nearest neighbours, chosen on validation scores only. Nothing is
trained: the vote reads the members' saved test predictions; the same code writes the result tables.
Results: training_code/results/voting.md

Everything below the imports is copied unchanged from the zillow/ modules that ran this training on
1-3 November 2026 (each block is marked with its source file); only the table loaders, the score
functions and the project settings are imported.

Run: python3 training_code/voting.py
"""
# AI was used to help write this code, but the resulting code was reviewed and edited by a human.

import json

import numpy as np
import pandas as pd

import paths  # noqa: F401  (puts zillow/ on the import path)
import shared_code  # noqa: F401  (points the loaders at the Zillow table)
from slide_models import FOLD_RESULTS, PREDICTIONS_DIR as SLIDE_PREDICTIONS
from neural_search import FOLDS, OUT_DIR as SEARCH_DIR
from final_training import FINAL_DIR, PREDICTIONS_DIR as FINAL_PREDICTIONS, load_test_table
from band_models import OUT_DIR as BAND_DIR, PREDICTIONS_DIR as BAND_PREDICTIONS, band_scores, price_bands
from train_models import scores
from shared_code import TABLE_DIR


# ----- copied from zillow/zillow_results.py -----


OUT_DIR = TABLE_DIR.parent / "results"
VOTERS = 3
TOP_SHARE = 0.01
BASELINES = ["average baseline", "median baseline"]


def main():
    OUT_DIR.mkdir(exist_ok=True)
    _, test_label = load_test_table()
    predictions = number_model_predictions()
    voters = validation_ranking().head(VOTERS)
    vote = f"voting (average of {', '.join(voters.index)})"
    predictions[vote] = np.mean([predictions[name] for name in voters.index], axis=0)
    voters.to_csv(OUT_DIR / "validation_ranking_used_for_voting.csv")
    tables = {"all_test_results": all_test_results(predictions, test_label),
              "top_one_percent": top_one_percent(predictions, test_label),
              "band_results": band_results(predictions, test_label)}
    for name, table in tables.items():
        table.to_csv(OUT_DIR / f"{name}.csv", index=False)
        print(f"===== {name}\n{table.round(4).to_string(index=False)}")


def number_model_predictions():
    folders = [FINAL_PREDICTIONS, SLIDE_PREDICTIONS]
    return {path.stem: np.load(path) for folder in folders for path in sorted(folder.glob("*.npy"))}


def validation_ranking():
    """Average validation MSE (log) over the five folds of each number model's winning setting."""
    network = pd.read_csv(FINAL_DIR / "search_ranking.csv")["mean"].iloc[0]
    linear = pd.read_csv(SEARCH_DIR / "linear_regression_folds.csv")["MSE (log)"].mean()
    folds = pd.read_csv(FOLD_RESULTS)
    by_setting = folds.groupby(["model", "setting"])["MSE (log)"].agg(["mean", "count"])
    slide = by_setting[by_setting["count"] == FOLDS]["mean"].groupby("model").min()
    ranking = pd.concat([pd.Series({"neural network": network, "linear regression": linear}), slide])
    return ranking.sort_values().rename("average validation MSE (log)")


def all_test_results(predictions, test_label):
    rows = [{"model": name, **scores(test_label, predicted)} for name, predicted in predictions.items()]
    return pd.DataFrame(rows).sort_values("MSE (log)").reset_index(drop=True)


def top_one_percent(predictions, test_label):
    value = np.exp(test_label)
    top = value >= np.quantile(value, 1 - TOP_SHARE)
    rows = []
    for name, predicted in predictions.items():
        if name in BASELINES:
            continue
        guess = np.exp(predicted)
        squared = (guess - value) ** 2
        rows.append({"model": name, "top 1% homes": int(top.sum()),
                     "median miss on the top 1% (%, minus = too low)": np.median((guess - value)[top] / value[top]) * 100,
                     "share of squared dollar error from the top 1% (%)": squared[top].sum() / squared.sum() * 100,
                     "RMSE ($), top 1%": np.sqrt(squared[top].mean()),
                     "RMSE ($), other 99%": np.sqrt(squared[~top].mean())})
    return pd.DataFrame(rows).sort_values("RMSE ($), other 99%").reset_index(drop=True)


def band_results(predictions, test_label):
    edges = np.array(json.loads((BAND_DIR / "band_edges.json").read_text())["log value edges"])
    true_bands = price_bands(test_label, edges)
    rows = [{"model": name, "kind": "number model, cut into bands", **band_scores(true_bands, price_bands(p, edges))}
            for name, p in predictions.items()]
    rows += [{"model": path.stem, "kind": "band model", **band_scores(true_bands, np.load(path))}
             for path in sorted(BAND_PREDICTIONS.glob("*.npy"))]
    return pd.DataFrame(rows).sort_values("accuracy (%)", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    main()

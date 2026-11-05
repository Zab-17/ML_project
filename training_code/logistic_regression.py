"""Logistic regression (Lecture 5), on the five price bands.

LogisticRegression: softmax over the five bands, L2 penalty. Searched: C in LOGISTIC_PENALTIES
(0.01, 1, 100), at most LOGISTIC_MAX_PASSES (300) passes of the L-BFGS solver.
The four steps (zillow/band_models.py): search on five folds, winner by highest validation accuracy, one final
fit, one test score. Results: training_code/results/logistic_regression.md

Everything below the imports is copied unchanged from the zillow/ modules that ran this training on
1-3 November 2026 (each block is marked with its source file); only the table loaders, the score
functions and the project settings are imported.

Run: python3 training_code/logistic_regression.py   (finished steps are skipped, so nothing already trained reruns)
"""
# AI was used to help write this code, but the resulting code was reviewed and edited by a human.

import json
import sys
import time

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import pandas as pd

import paths  # noqa: F401  (puts zillow/ on the import path)
import shared_code  # noqa: F401  (points the loaders at the Zillow table)
from final_training import load_test_table
from slide_models import append_row, finished, setting_name
from neural_search import FOLDS, load_table, log
from shared_code import TABLE_DIR


# ----- copied from zillow/band_models.py -----


OUT_DIR = TABLE_DIR.parent / "band_models"
LOG = OUT_DIR / "log.txt"
FOLD_RESULTS, TEST_RESULTS = OUT_DIR / "fold_results.csv", OUT_DIR / "test_results.csv"
PREDICTIONS_DIR = OUT_DIR / "test_predictions"
BANDS = 5
LOGISTIC_PENALTIES = [0.01, 1.0, 100.0]    # C: larger means weaker regularization
LOGISTIC_MAX_PASSES = 300


def main(names=None):
    """The named models (every model when none are named), from the command line unless names is given."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(exist_ok=True)
    matrix, label, folds = load_table()
    test_matrix, test_label = load_test_table()
    edges = band_edges(label)
    (OUT_DIR / "band_edges.json").write_text(json.dumps({"log value edges": edges.tolist(),
                                                          "dollar edges": np.exp(edges).round().tolist()}))
    bands, test_bands = price_bands(label, edges), price_bands(test_label, edges)
    for name in (sys.argv[1:] if names is None else names) or list(MODELS):
        search(name, matrix, bands, folds)
        final_test(name, winner(name), matrix, bands, test_matrix, test_bands)


def band_edges(training_log_value):
    """The four inner edges between five equal-sized bands of training homes."""
    return np.quantile(training_log_value, np.arange(1, BANDS) / BANDS)


def price_bands(log_value, edges):
    """Band 1 to 5 for every home."""
    return np.searchsorted(edges, log_value, side="right") + 1


def band_scores(true_bands, predicted_bands):
    return {"accuracy (%)": accuracy_score(true_bands, predicted_bands) * 100,
            "macro F1": f1_score(true_bands, predicted_bands, average="macro"),
            "within one band (%)": (np.abs(true_bands - predicted_bands) <= 1).mean() * 100}


def search(name, matrix, bands, folds):
    settings, build = MODELS[name]
    done = finished(FOLD_RESULTS)
    for fold in range(FOLDS):
        fit_rows, check_rows = folds != fold, folds == fold
        for setting in settings:
            if (name, setting_name(setting), fold) in done:
                continue
            start = time.perf_counter()
            predicted = build(**setting).fit(matrix[fit_rows], bands[fit_rows]).predict(matrix[check_rows])
            row = {"model": name, "setting": setting_name(setting), "fold": fold,
                   "seconds": round(time.perf_counter() - start, 1), **band_scores(bands[check_rows], predicted)}
            append_row(FOLD_RESULTS, row)
            log(f"{name} | {row['setting']} | fold {fold} | validation accuracy {row['accuracy (%)']:.2f}% | "
                f"{row['seconds']:.0f} s", LOG)


def winner(name):
    """The setting with the highest average validation accuracy, among settings scored on all 5 folds."""
    results = pd.read_csv(FOLD_RESULTS)
    runs = results[results["model"] == name]
    complete = runs.groupby("setting").filter(lambda setting_runs: len(setting_runs) == FOLDS)
    ranking = complete.groupby("setting")["accuracy (%)"].mean().sort_values(ascending=False)
    log(f"{name} winner: {ranking.index[0]}, average validation accuracy {ranking.iloc[0]:.2f}%", LOG)
    return json.loads(ranking.index[0])


def final_test(name, setting, matrix, bands, test_matrix, test_bands):
    """Fit the winner on every training home and score it once on the test homes; never twice."""
    if name in {model for model, _, _ in finished(TEST_RESULTS)}:
        log(f"{name} was already scored on the test homes; not scoring it again", LOG)
        return
    _, build = MODELS[name]
    start = time.perf_counter()
    predicted = build(**setting).fit(matrix, bands).predict(test_matrix)
    row = {"model": name, "setting": setting_name(setting), "fold": "test",
           "seconds": round(time.perf_counter() - start, 1), **band_scores(test_bands, predicted)}
    np.save(PREDICTIONS_DIR / f"{name}.npy", predicted)
    append_row(TEST_RESULTS, row)
    log(f"{name} | {row['setting']} | test accuracy {row['accuracy (%)']:.2f}%, macro F1 {row['macro F1']:.3f}", LOG)


# ----- this model, as listed in zillow/band_models.py -----
MODELS = {
    "logistic regression": ([{"penalty_c": c} for c in LOGISTIC_PENALTIES],
                            lambda penalty_c: LogisticRegression(C=penalty_c, max_iter=LOGISTIC_MAX_PASSES)),
}


if __name__ == "__main__":
    main(["logistic regression"])

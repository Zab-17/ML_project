"""Logistic regression (lecture 5), the perceptron (lecture 6) and naive Bayes (lecture 11) on the Zillow homes.

These three models predict a class, not a number, so the log assessed value is cut into five price bands of equal
size at the training quintiles: band 1 is the cheapest fifth of training homes, band 5 the dearest fifth. They go
through the same four steps as every other model: a five-fold search, the winner by the highest average
validation accuracy, one final fit on every training home, and one score on the sealed test homes.
zillow_results.py cuts every number model's test predictions at the same edges, so all models share one table.

Run: python3 band_models.py [model name ...]
Writes out/band_models/: band_edges.json, fold_results.csv, test_results.csv, test_predictions/, log.txt
"""
# AI was used to help write this code, but the resulting code was reviewed and edited by a human.

import json
import sys
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.metrics import accuracy_score, f1_score
from sklearn.naive_bayes import BernoulliNB, GaussianNB

from shared_code import TABLE_DIR
from final_training import load_test_table
from neural_search import FOLDS, SEED, load_table, log
from slide_models import append_row, finished, setting_name

OUT_DIR = TABLE_DIR.parent / "band_models"
LOG = OUT_DIR / "log.txt"
FOLD_RESULTS, TEST_RESULTS = OUT_DIR / "fold_results.csv", OUT_DIR / "test_results.csv"
PREDICTIONS_DIR = OUT_DIR / "test_predictions"
BANDS = 5
LOGISTIC_PENALTIES = [0.01, 1.0, 100.0]    # C: larger means weaker regularization
LOGISTIC_MAX_PASSES = 300
PERCEPTRON_PASSES = [5, 20]
SMOOTHING = [0.1, 1.0, 10.0]               # Laplace smoothing for the 0/1 columns


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(exist_ok=True)
    matrix, label, folds = load_table()
    test_matrix, test_label = load_test_table()
    edges = band_edges(label)
    (OUT_DIR / "band_edges.json").write_text(json.dumps({"log value edges": edges.tolist(),
                                                          "dollar edges": np.exp(edges).round().tolist()}))
    bands, test_bands = price_bands(label, edges), price_bands(test_label, edges)
    for name in sys.argv[1:] or list(MODELS):
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


class MixedNaiveBayes:
    """Naive Bayes for numbers and 0/1 columns together (lecture 11: it 'works with both categorical and numerical
    attributes'): a normal distribution per band for each standardized number, a yes/no probability per band for
    each 0/1 column, all multiplied together with the band's prior. The columns are taken as independent within a
    band, which is the 'naive' part."""

    def __init__(self, number_columns, smoothing):
        self.numbers = number_columns
        self.gaussian, self.bernoulli = GaussianNB(), BernoulliNB(alpha=smoothing, binarize=None)

    def fit(self, matrix, bands):
        self.gaussian.fit(dense_numbers(matrix, self.numbers), bands)
        self.bernoulli.fit(matrix[:, ~self.numbers], bands)
        return self

    def predict(self, matrix):
        joint = (self.gaussian.predict_joint_log_proba(dense_numbers(matrix, self.numbers))
                 + self.bernoulli.predict_joint_log_proba(matrix[:, ~self.numbers])
                 - np.log(self.gaussian.class_prior_))   # the band prior is in both parts; count it once
        return self.gaussian.classes_[joint.argmax(axis=1)]


def dense_numbers(matrix, number_columns):
    return matrix[:, number_columns].toarray()


def number_columns():
    return (pd.read_csv(TABLE_DIR / "columns.csv")["kind"] == "number").to_numpy()


MODELS = {
    "logistic regression": ([{"penalty_c": c} for c in LOGISTIC_PENALTIES],
                            lambda penalty_c: LogisticRegression(C=penalty_c, max_iter=LOGISTIC_MAX_PASSES)),
    "perceptron": ([{"passes": p} for p in PERCEPTRON_PASSES],
                   lambda passes: Perceptron(max_iter=passes, tol=None, random_state=SEED)),
    "naive Bayes": ([{"smoothing": s} for s in SMOOTHING],
                    lambda smoothing: MixedNaiveBayes(number_columns(), smoothing)),
}


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


if __name__ == "__main__":
    main()

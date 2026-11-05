"""AdaBoost.R2, regression form (Lecture 13 teaches it as a classifier).

AdaBoostRegressor: BOOSTING_ROUNDS (50) regression trees, each round resampling the homes by the previous
round's errors, combined by a weighted median. Searched: tree depth in BOOSTED_TREE_DEPTHS (4, 8).
The four steps (zillow/slide_models.py): search on five folds, winner by lowest validation MSE of the log
value, one final fit, one test score. Results: training_code/results/adaboost.md

Everything below the imports is copied unchanged from the zillow/ modules that ran this training on
1-3 November 2026 (each block is marked with its source file); only the table loaders, the score
functions and the project settings are imported.

Run: python3 training_code/adaboost.py   (finished steps are skipped, so nothing already trained reruns)
"""
# AI was used to help write this code, but the resulting code was reviewed and edited by a human.

import json
import time

from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
import numpy as np
import pandas as pd

import paths  # noqa: F401  (puts zillow/ on the import path)
import shared_code  # noqa: F401  (points the loaders at the Zillow table)
from final_training import FINAL_DIR, load_test_table
from neural_search import FOLDS, SEED, TABLE_DIR, load_table, log
from train_models import scores


# ----- copied from zillow/slide_models.py -----


OUT_DIR = TABLE_DIR.parent / "slide_models"
LOG = OUT_DIR / "log.txt"
FOLD_RESULTS = OUT_DIR / "fold_results.csv"
TEST_RESULTS = OUT_DIR / "test_results.csv"
PREDICTIONS_DIR = OUT_DIR / "test_predictions"


def run(models, names):
    """The four steps for every named model (every model when names is empty), then the combined test table."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(exist_ok=True)
    matrix, label, folds = load_table()
    test_matrix, test_label = load_test_table()
    for name in names or list(models):
        model = models[name]
        search(model, matrix, label, folds)
        final_test(model, winner(model), matrix, label, test_matrix, test_label)
    save_all_test_results()


class FittedModel:
    """A scikit-learn model fitted once per setting, on dense rows (trees) or sparse rows (linear models)."""

    def __init__(self, name, settings, build, dense):
        self.name, self.settings, self.build, self.dense = name, settings, build, dense

    def predictions(self, settings, train_matrix, train_label, query_matrix):
        train, query = self.rows(train_matrix), self.rows(query_matrix)
        for setting in settings:
            yield setting, self.build(**setting).fit(train, train_label).predict(query)

    def rows(self, matrix):
        return dense_rows(matrix) if self.dense else matrix


def search(model, matrix, label, folds):
    done = finished(FOLD_RESULTS)
    for fold in range(FOLDS):
        settings = [s for s in model.settings if (model.name, setting_name(s), fold) not in done]
        if not settings:
            continue
        fit_rows, check_rows = folds != fold, folds == fold
        start = time.perf_counter()
        for setting, predicted in model.predictions(settings, matrix[fit_rows], label[fit_rows], matrix[check_rows]):
            row = {"model": model.name, "setting": setting_name(setting), "fold": fold,
                   "seconds": round(time.perf_counter() - start, 1), **scores(label[check_rows], predicted)}
            append_row(FOLD_RESULTS, row)
            log(f"{model.name} | {row['setting']} | fold {fold} | validation MSE (log) {row['MSE (log)']:.4f} | "
                f"{row['seconds']:.0f} s", LOG)
            start = time.perf_counter()


def winner(model):
    """The setting with the lowest average validation MSE (log), among settings scored on all 5 folds."""
    results = pd.read_csv(FOLD_RESULTS)
    runs = results[results["model"] == model.name].dropna(subset=["MSE (log)"])
    complete = runs.groupby("setting").filter(lambda setting_runs: len(setting_runs) == FOLDS)
    ranking = complete.groupby("setting")["MSE (log)"].mean().sort_values()
    log(f"{model.name} winner: {ranking.index[0]}, average validation MSE (log) {ranking.iloc[0]:.4f}", LOG)
    return json.loads(ranking.index[0])


def final_test(model, setting, matrix, label, test_matrix, test_label):
    """Fit the winner on every training sale and score it once on the test sales; never twice."""
    if model.name in {name for name, _, _ in finished(TEST_RESULTS)}:
        log(f"{model.name} was already scored on the test sales; not scoring it again", LOG)
        return
    start = time.perf_counter()
    [(_, predicted)] = model.predictions([setting], matrix, label, test_matrix)
    row = {"model": model.name, "setting": setting_name(setting), "fold": "test",
           "seconds": round(time.perf_counter() - start, 1), **scores(test_label, predicted)}
    np.save(PREDICTIONS_DIR / f"{model.name}.npy", predicted)
    append_row(TEST_RESULTS, row)
    log(f"{model.name} | {row['setting']} | trained on {matrix.shape[0]:,} sales in {row['seconds']:.0f} s | "
        f"test MSE (log) {row['MSE (log)']:.4f}, median % off {row['median % off']:.2f}", LOG)


def save_all_test_results():
    """One table: the slide models next to the network, linear regression and the baselines."""
    earlier = pd.read_csv(FINAL_DIR / "test_results.csv")
    slide = pd.read_csv(TEST_RESULTS).drop(columns=["setting", "fold", "seconds"])
    table = pd.concat([earlier, slide], ignore_index=True).sort_values("MSE (log)")
    table.to_csv(OUT_DIR / "all_test_results.csv", index=False)
    log(table.set_index("model").T.round(4).to_string(), LOG)


def dense_rows(matrix):
    return matrix.astype(np.float32).toarray()


def setting_name(setting):
    return json.dumps(setting, sort_keys=True)


def finished(path):
    if not path.exists():
        return set()
    done = pd.read_csv(path)
    return set(zip(done["model"], done["setting"], done["fold"]))


def append_row(path, row):
    pd.DataFrame([row]).to_csv(path, mode="a", header=not path.exists(), index=False)


# ----- copied from zillow/zillow_models.py -----


BOOSTING_ROUNDS = 50
BOOSTED_TREE_DEPTHS = [4, 8]


def adaboost(tree_depth):
    """AdaBoost for a number (AdaBoost.R2): each round fits a tree of this depth to a sample drawn with the homes the
    earlier rounds missed most drawn more often; the rounds vote by weighted median."""
    return AdaBoostRegressor(DecisionTreeRegressor(max_depth=tree_depth, random_state=SEED),
                             n_estimators=BOOSTING_ROUNDS, random_state=SEED)


# ----- this model, as listed in zillow/zillow_models.py -----
MODELS = {model.name: model for model in [
    FittedModel("AdaBoost", [{"tree_depth": d} for d in BOOSTED_TREE_DEPTHS], adaboost, dense=False),
]}


if __name__ == "__main__":
    run(MODELS, ["AdaBoost"])

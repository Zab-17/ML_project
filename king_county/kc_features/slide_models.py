"""Every course model that predicts a number, put through the same four steps as the neural network.

1. Five-fold search: every setting trains on four fifths of the training homes and is scored on the fifth.
2. Winner: per model, the setting with the lowest average validation MSE (log) over the 5 folds.
3. Final training: the winner is fitted once on all 426,135 training sales.
4. Test: the final model is scored once on the 106,755 sealed test sales.

Models: K-nearest neighbours (lecture 3), CART decision tree with cost-complexity pruning (lecture 10),
bagging and random forest (lecture 10), and a linear support vector machine for regression (lecture 2).
Linear regression and the neural network ran in train_models.py, neural_search.py and final_training.py.
Logistic regression, the perceptron and naive Bayes only predict classes, so they cannot predict a price.

Run: python3 slide_models.py [model name ...]   default: every model, in MODELS order
Writes out/slide_models/: fold_results.csv, test_results.csv, all_test_results.csv, test_predictions/, log.txt
"""
# AI was used to help write this code, but the resulting code was reviewed and edited by a human.

import json
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import LinearSVR
from sklearn.tree import DecisionTreeRegressor

from final_training import FINAL_DIR, load_test_table
from neural_search import DEVICE, FOLDS, SEED, TABLE_DIR, load_table, log
from train_models import scores

OUT_DIR = TABLE_DIR.parent / "slide_models"
LOG = OUT_DIR / "log.txt"
FOLD_RESULTS = OUT_DIR / "fold_results.csv"
TEST_RESULTS = OUT_DIR / "test_results.csv"
PREDICTIONS_DIR = OUT_DIR / "test_predictions"

NEIGHBOUR_COUNTS = [1, 3, 5, 10, 15, 20, 30, 50]
DISTANCES = ["euclidean", "cosine"]
QUERY_BATCH = 1024                        # sales whose neighbours are searched at once on the GPU
PRUNING_STRENGTHS = [0.0, 1e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4]   # ccp_alpha: the error a branch must save to stay
TREES = 50                                # 50, not 100, keeps the whole run near 2.5 hours on the M1 Pro
FOREST_COLUMN_SHARES = ["sqrt", 0.1, 0.33]
MARGIN_PENALTIES = [0.01, 1.0]            # C: how much one sale outside the margin costs
MARGIN_WIDTHS = [0.0, 0.05]               # epsilon in log price: 0.05 means misses under about 5% cost nothing
SVM_MAX_PASSES = 2000


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(exist_ok=True)
    matrix, label, folds = load_table()
    test_matrix, test_label = load_test_table()
    for name in sys.argv[1:] or list(MODELS):
        model = MODELS[name]
        search(model, matrix, label, folds)
        final_test(model, winner(model), matrix, label, test_matrix, test_label)
    save_all_test_results()


class NearestNeighbours:
    """K-nearest neighbours (lecture 3): a sale is priced at the average log price of the k closest training
    sales. Training only stores the rows, so the neighbours are found once per distance for every k."""

    name = "k-nearest neighbours"
    settings = [{"distance": distance, "k": k} for distance in DISTANCES for k in NEIGHBOUR_COUNTS]

    def predictions(self, settings, train_matrix, train_label, query_matrix):
        for distance in dict.fromkeys(setting["distance"] for setting in settings):
            chosen = [setting for setting in settings if setting["distance"] == distance]
            nearest = nearest_rows(train_matrix, query_matrix, max(s["k"] for s in chosen), distance)
            for setting in chosen:
                yield setting, train_label[nearest[:, :setting["k"]]].mean(axis=1)


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


def decision_tree(pruning):
    """CART (lecture 10): binary splits chosen by squared error, grown in full, then pruned by cost complexity."""
    return DecisionTreeRegressor(ccp_alpha=pruning, random_state=SEED)


def random_forest(column_share):
    """Unpruned CART trees on bootstrap samples, each split choosing from a random share of the columns."""
    return RandomForestRegressor(n_estimators=TREES, max_features=column_share, n_jobs=-1, random_state=SEED)


def bagging():
    """Lecture 10's traditional bagging: a forest whose splits may choose from every column."""
    return random_forest(column_share=1.0)


def support_vector_machine(margin_penalty, margin_width):
    return LinearSVR(C=margin_penalty, epsilon=margin_width, dual=True, max_iter=SVM_MAX_PASSES, random_state=SEED)


MODELS = {model.name: model for model in [
    FittedModel("decision tree", [{"pruning": p} for p in PRUNING_STRENGTHS], decision_tree, dense=True),
    FittedModel("random forest", [{"column_share": s} for s in FOREST_COLUMN_SHARES], random_forest, dense=True),
    NearestNeighbours(),
    FittedModel("support vector machine", [{"margin_penalty": c, "margin_width": w}
                                           for c in MARGIN_PENALTIES for w in MARGIN_WIDTHS],
                support_vector_machine, dense=False),
    FittedModel("bagging", [{}], bagging, dense=True),
]}


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


def nearest_rows(train_matrix, query_matrix, k, distance):
    """Row numbers of the k training rows closest to each query row, nearest first, found on the GPU.
    The query row's own squared length is the same for every training row, so it is left out of the ranking."""
    train = on_device(train_matrix, distance)
    train_lengths = (train ** 2).sum(dim=1)
    parts = []
    for start in range(0, query_matrix.shape[0], QUERY_BATCH):
        query = on_device(query_matrix[start:start + QUERY_BATCH], distance)
        ranking = train_lengths - 2 * query @ train.T
        parts.append(torch.topk(ranking, k, dim=1, largest=False).indices.cpu().numpy())
    return np.concatenate(parts)


def on_device(rows, distance):
    """Rows as a GPU tensor; for the cosine distance every row is scaled to length 1, so that the
    Euclidean ranking of the scaled rows is the cosine ranking of the original rows."""
    tensor = torch.from_numpy(dense_rows(rows)).to(DEVICE)
    return torch.nn.functional.normalize(tensor, dim=1) if distance == "cosine" else tensor


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


if __name__ == "__main__":
    main()

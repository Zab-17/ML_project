"""K-nearest neighbours, regression form (Lecture 3 teaches the classifier: see knn_classifier.py).

The average log value of the k nearest training homes. Searched: k in NEIGHBOUR_COUNTS (1 to 50) x
Euclidean or cosine distance. Exact search on the graphics processor in blocks (zillow/zillow_neighbours.py).
The four steps (zillow/slide_models.py): search on five folds, winner by lowest validation MSE of the log
value, one final fit, one test score. Results: training_code/results/knn_regression.md

Everything below the imports is copied unchanged from the zillow/ modules that ran this training on
1-3 November 2026 (each block is marked with its source file); only the table loaders, the score
functions and the project settings are imported.

Run: python3 training_code/knn_regression.py   (finished steps are skipped, so nothing already trained reruns)
"""
# AI was used to help write this code, but the resulting code was reviewed and edited by a human.

import json
import time

import numpy as np
import pandas as pd
import torch

import paths  # noqa: F401  (puts zillow/ on the import path)
import shared_code  # noqa: F401  (points the loaders at the Zillow table)
from final_training import FINAL_DIR, load_test_table
from neural_search import DEVICE, FOLDS, TABLE_DIR, load_table, log
from train_models import scores


# ----- copied from zillow/slide_models.py -----


OUT_DIR = TABLE_DIR.parent / "slide_models"
LOG = OUT_DIR / "log.txt"
FOLD_RESULTS = OUT_DIR / "fold_results.csv"
TEST_RESULTS = OUT_DIR / "test_results.csv"
PREDICTIONS_DIR = OUT_DIR / "test_predictions"
NEIGHBOUR_COUNTS = [1, 3, 5, 10, 15, 20, 30, 50]
DISTANCES = ["euclidean", "cosine"]
DISTANCE_TABLE_BYTES = 1_500_000_000      # GPU memory for one batch of query-to-training distances


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
    batch_size = query_batch_size(train.shape[0])
    ranking = torch.empty((batch_size, train.shape[0]), dtype=train.dtype, device=train.device)
    parts = []
    for start in range(0, query_matrix.shape[0], batch_size):
        query = on_device(query_matrix[start:start + batch_size], distance)
        batch_ranking = ranking[:query.shape[0]]
        torch.addmm(train_lengths, query, train.T, alpha=-2, out=batch_ranking)  # one buffer, no temporaries
        parts.append(torch.topk(batch_ranking, k, dim=1, largest=False).indices.cpu().numpy())
    return np.concatenate(parts)


def query_batch_size(training_rows):
    """Query rows per GPU batch, so that one batch's distance table (4 bytes per pair) fits DISTANCE_TABLE_BYTES.
    About 1,100 rows for King County's 341,000 training rows; about 250 for Zillow's 1.5 million."""
    return max(1, DISTANCE_TABLE_BYTES // (4 * training_rows))


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


# ----- copied from zillow/zillow_neighbours.py -----


TRAINING_BLOCK = 200_000   # training homes on the GPU at once: 200,000 x 606 x 4 bytes = 0.48 GB
QUERY_BATCH = 250          # measured on one fold: 4.4 GB peak (500 reached 5.2 GB) at about the same speed


class BlockSearch:
    """Exact nearest-row search, one block of training rows and one batch of query rows at a time."""

    def __init__(self, training_block, query_batch):
        self.training_block, self.query_batch = training_block, query_batch

    def nearest_rows(self, train_matrix, query_matrix, k, distance):
        """Row numbers of the k training rows closest to each query row, nearest first. The query row's own
        squared length is the same for every training row, so it is left out of the score."""
        nearest = RunningNearest(query_matrix.shape[0], k)
        for block_start in range(0, train_matrix.shape[0], self.training_block):
            block = on_device(train_matrix[block_start:block_start + self.training_block], distance)
            lengths = (block ** 2).sum(dim=1)
            for start in range(0, query_matrix.shape[0], self.query_batch):
                query = on_device(query_matrix[start:start + self.query_batch], distance)
                nearest.merge(start, torch.addmm(lengths, query, block.T, alpha=-2), block_start)
                wait_for_device()   # unfinished GPU work keeps its distance table alive; queued tables pile up
            del block, lengths      # free this block before the next one is moved over
        return nearest.row_numbers.cpu().numpy()


class RunningNearest:
    """For every query row, the k lowest scores seen so far and the training row numbers they belong to."""

    def __init__(self, query_count, k):
        self.k = k
        self.scores = torch.full((query_count, k), float("inf"), device=DEVICE)
        self.row_numbers = torch.zeros((query_count, k), dtype=torch.int64, device=DEVICE)

    def merge(self, first_query, block_scores, first_training_row):
        """Keep the k lowest of the running list and of one block's scores, for the query rows that start at
        first_query. Column c of block_scores is training row first_training_row + c."""
        rows = slice(first_query, first_query + block_scores.shape[0])
        block_best, columns = torch.topk(block_scores, min(self.k, block_scores.shape[1]), dim=1, largest=False)
        scores = torch.cat([self.scores[rows], block_best], dim=1)
        row_numbers = torch.cat([self.row_numbers[rows], columns + first_training_row], dim=1)
        self.scores[rows], kept = torch.topk(scores, self.k, dim=1, largest=False)
        self.row_numbers[rows] = torch.gather(row_numbers, 1, kept)


class BlockNearestNeighbours(NearestNeighbours):
    """K-nearest neighbours as in slide_models (same settings, same averaging), with the search done in blocks.
    The loop repeats NearestNeighbours.predictions because that method calls the one-pass search directly."""

    def __init__(self, search):
        self.search = search

    def predictions(self, settings, train_matrix, train_label, query_matrix):
        for distance in dict.fromkeys(setting["distance"] for setting in settings):
            chosen = [setting for setting in settings if setting["distance"] == distance]
            nearest = self.search.nearest_rows(train_matrix, query_matrix, max(s["k"] for s in chosen), distance)
            for setting in chosen:
                yield setting, train_label[nearest[:, :setting["k"]]].mean(axis=1)


def wait_for_device():
    if DEVICE == "mps":
        torch.mps.synchronize()


# ----- this model, as listed in zillow/zillow_models.py -----
MODELS = {model.name: model for model in [
    BlockNearestNeighbours(BlockSearch(TRAINING_BLOCK, QUERY_BATCH)),
]}


if __name__ == "__main__":
    run(MODELS, ["k-nearest neighbours"])

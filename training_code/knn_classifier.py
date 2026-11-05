"""K-nearest neighbours as Lecture 3 teaches it, on the five price bands.

A home gets the most common band among its k nearest training homes (majority vote). As the slide requires, the
features are MinMax scaled: each of the 7 number columns is mapped onto 0 to 1 with the minimum and maximum of the
homes the model is fitted on; the 599 yes/no and one-hot columns are already 0 or 1. Euclidean and cosine distance
(the slide calls cosine "best suited for one-hot encoding"), and k chosen by five-fold cross-validation.
Trained on all 2,287,442 training homes; tested on all 571,860 test homes. The first run used a random sample of
100,000 training homes; its results stay in results/knn_classifier/ and results/knn_classifier.md, and this run writes to
results/knn_classifier_all_homes/ and results/knn_classifier_all_homes.md, so nothing from the first run is skipped or overwritten.

The four steps are copied from band_runner.py, the runner written for these classifiers.

Run: caffeinate -is python3 -u training_code/knn_classifier.py   (about 6-7 hours on the M1 Pro, on the 67 W charger)
"""
# AI was used to help write this code, but the resulting code was reviewed and edited by a human.

from dataclasses import dataclass
import json
import time

from scipy import sparse
import numpy as np
import pandas as pd

import paths  # noqa: F401  (puts zillow/ on the import path)
import shared_code  # noqa: F401  (points the loaders at the Zillow table)
from band_models import band_edges, band_scores, number_columns, price_bands
from neural_search import FOLDS, SEED, load_table
from slide_models import append_row, finished, query_batch_size, setting_name
from zillow_neighbours import BlockSearch, TRAINING_BLOCK
from final_training import load_test_table


# ----- the four steps, from band_runner.py -----


SCORES = ["accuracy (%)", "macro F1", "within one band (%)"]


@dataclass
class BandTable:
    matrix: sparse.csr_matrix      # training homes, or a sample of them
    bands: np.ndarray              # band 1 to 5 of each of those homes
    folds: np.ndarray              # fold 0 to 4 of each of those homes
    test_matrix: sparse.csr_matrix
    test_bands: np.ndarray
    sample_size: int | None        # None means every training home


def load_band_table(sample_size=None):
    matrix, label, folds = load_table()
    test_matrix, test_label = load_test_table()
    edges = band_edges(label)
    rows = sampled_rows(len(label), sample_size)
    return BandTable(matrix[rows], price_bands(label[rows], edges), folds[rows], test_matrix,
                     price_bands(test_label, edges), sample_size)


def sampled_rows(row_count, sample_size):
    """Every row, or sample_size rows drawn at random with the project seed, kept in their original order."""
    if sample_size is None or sample_size >= row_count:
        return np.arange(row_count)
    return np.sort(np.random.default_rng(SEED).choice(row_count, size=sample_size, replace=False))


@dataclass
class OutputFiles:
    folder: paths.Path

    @classmethod
    def for_model(cls, file_stem):
        folder = paths.RESULTS_DIR / file_stem
        folder.mkdir(parents=True, exist_ok=True)
        return cls(folder)

    @property
    def fold_results(self):
        return self.folder / "fold_results.csv"

    @property
    def test_results(self):
        return self.folder / "test_results.csv"

    @property
    def predictions(self):
        return self.folder / "test_predictions.npy"

    @property
    def log(self):
        return self.folder / "log.txt"

    @property
    def page(self):
        return paths.RESULTS_DIR / f"{self.folder.name}.md"


def run(model, table, file_stem):
    """Search, choose, fit once more and test once; then write the model's results page."""
    files = OutputFiles.for_model(file_stem)
    search(model, table, files)
    final_test(model, winner(model, files), table, files)
    write_results_page(model, table, files)


def search(model, table, files):
    done = finished(files.fold_results)
    for fold in range(FOLDS):
        pending = [s for s in model.settings if (model.name, setting_name(s), fold) not in done]
        if not pending:
            continue
        fit, check = table.folds != fold, table.folds == fold
        start = time.perf_counter()
        for setting, predicted in model.predictions(pending, table.matrix[fit], table.bands[fit], table.matrix[check]):
            seconds, start = time.perf_counter() - start, time.perf_counter()
            row = {"model": model.name, "setting": setting_name(setting), "fold": fold, "seconds": round(seconds, 1),
                   **band_scores(table.bands[check], predicted)}
            append_row(files.fold_results, row)
            log(files, f"{row['setting']} | fold {fold} | accuracy {row['accuracy (%)']:.2f}% | {seconds:.0f} s")


def winner(model, files):
    results = pd.read_csv(files.fold_results)
    complete = results.groupby("setting").filter(lambda runs: len(runs) == FOLDS)
    ranking = complete.groupby("setting")["accuracy (%)"].mean().sort_values(ascending=False)
    log(files, f"winner: {ranking.index[0]}, average validation accuracy {ranking.iloc[0]:.2f}%")
    return json.loads(ranking.index[0])


def final_test(model, setting, table, files):
    if files.test_results.exists():
        log(files, "already scored on the test homes; not scoring again")
        return
    start = time.perf_counter()
    [(_, predicted)] = model.predictions([setting], table.matrix, table.bands, table.test_matrix)
    row = {"model": model.name, "setting": setting_name(setting), "fold": "test",
           "seconds": round(time.perf_counter() - start, 1), **band_scores(table.test_bands, predicted)}
    np.save(files.predictions, predicted)
    append_row(files.test_results, row)
    log(files, f"test | {row['setting']} | accuracy {row['accuracy (%)']:.2f}%, macro F1 {row['macro F1']:.3f}")


def write_results_page(model, table, files):
    folds, test = pd.read_csv(files.fold_results), pd.read_csv(files.test_results)
    trained_on = "all 2,287,442 training homes" if table.sample_size is None else \
        f"a random sample of {table.sample_size:,} training homes (seed {SEED})"
    lines = [f"# {model.name}", "", model.description, "",
             f"Trained on {trained_on}, in five folds. Tested once on all 571,860 test homes. "
             "Five price bands: equal fifths of the training homes. Guessing scores about 20%.", "",
             "## Search: average over the five validation folds", "", search_table(folds), "",
             "## Test", "", test_table(test), ""]
    files.page.write_text("\n".join(lines))


def search_table(folds):
    grouped = folds.groupby("setting", sort=False)
    summary = grouped[SCORES + ["seconds"]].mean()
    summary.insert(0, "accuracy spread", grouped["accuracy (%)"].std())
    summary = summary.sort_values("accuracy (%)", ascending=False)
    header = "| Setting | Accuracy (%) | Spread | Macro F1 | Within one band (%) | Seconds per fold |\n|---|---|---|---|---|---|"
    rows = [f"| {setting} | {r['accuracy (%)']:.2f} | {r['accuracy spread']:.2f} | {r['macro F1']:.4f} | "
            f"{r['within one band (%)']:.2f} | {r['seconds']:.1f} |" for setting, r in summary.iterrows()]
    return "\n".join([header] + rows)


def test_table(test):
    r = test.iloc[0]
    return ("| Winning setting | Accuracy (%) | Macro F1 | Within one band (%) | Seconds (fit and test) |\n|---|---|---|---|---|\n"
            f"| {r['setting']} | {r['accuracy (%)']:.2f} | {r['macro F1']:.4f} | {r['within one band (%)']:.2f} | "
            f"{r['seconds']:.1f} |")


def log(files, message):
    line = f"{time.strftime('%Y-%m-%d %H:%M')} {message}"
    print(line, flush=True)
    with files.log.open("a") as handle:
        handle.write(line + "\n")


# ----- this model -----


SAMPLE_SIZE = None                       # every training home; the first run used 100_000
RESULTS_NAME = "knn_classifier_all_homes"   # a new folder: the first run's results are kept and nothing is skipped
NEIGHBOUR_COUNTS = [1, 3, 5, 10, 15, 20, 30, 50]
DISTANCES = ["euclidean", "cosine"]
BAND_COUNT = 5


class MinMaxColumns:
    """Maps the chosen columns onto 0 to 1 with the minimum and maximum of the rows it was fitted on."""

    def __init__(self, column_mask, fit_matrix):
        self.mask = column_mask
        values = fit_matrix[:, column_mask].toarray()
        self.low, self.span = values.min(axis=0), np.maximum(values.max(axis=0) - values.min(axis=0), 1e-12)

    def scaled(self, matrix):
        """The rows with the chosen columns MinMax scaled. Column order changes, which no distance depends on."""
        scaled_numbers = (matrix[:, self.mask].toarray() - self.low) / self.span
        return sparse.hstack([sparse.csr_matrix(scaled_numbers), matrix[:, ~self.mask]]).tocsr()


def majority_vote(neighbour_bands):
    """The most common band in each row of neighbour bands (nearest first). Each vote also carries a tie-break
    weight that halves with every rank (0.25, 0.125, ...), less than one vote in total, so a strict majority always
    wins and a tie goes to the tied band that holds the nearest neighbour."""
    k = neighbour_bands.shape[1]
    weights = 1 + 0.5 ** (np.arange(k) + 2)
    counts = np.stack([((neighbour_bands == band) * weights).sum(axis=1) for band in range(1, BAND_COUNT + 1)], axis=1)
    return counts.argmax(axis=1) + 1


class MajorityVoteNeighbours:
    name = "k-nearest neighbours (majority vote)"
    description = ("Lecture 3: majority vote of the k nearest training homes, MinMax scaling, Euclidean or cosine "
                   "distance, k chosen by five-fold cross-validation. One neighbour search per distance serves every k, "
                   "so each fold's search time is listed under the first k of that distance.")
    settings = [{"distance": d, "k": k} for d in DISTANCES for k in NEIGHBOUR_COUNTS]

    def __init__(self, number_mask):
        self.number_mask = number_mask

    def predictions(self, settings, fit_matrix, fit_bands, query_matrix):
        scaler = MinMaxColumns(self.number_mask, fit_matrix)
        fit_rows, query_rows = scaler.scaled(fit_matrix), scaler.scaled(query_matrix)
        search = BlockSearch(TRAINING_BLOCK, query_batch_size(fit_rows.shape[0]))
        for distance in dict.fromkeys(setting["distance"] for setting in settings):
            chosen = [setting for setting in settings if setting["distance"] == distance]
            nearest = search.nearest_rows(fit_rows, query_rows, max(s["k"] for s in chosen), distance)
            for setting in chosen:
                yield setting, majority_vote(fit_bands[nearest[:, :setting["k"]]])


def main():
    run(MajorityVoteNeighbours(number_columns()), load_band_table(SAMPLE_SIZE), RESULTS_NAME)


if __name__ == "__main__":
    main()

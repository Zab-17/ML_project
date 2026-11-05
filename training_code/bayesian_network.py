"""The Bayesian network as Lecture 11 teaches it, on the five price bands.

Structure from domain knowledge (causal relationships), as the slide's first route: a home's location, size,
quality and age cause its value, so the price band is a child of those features. The slide says to try several
structures and keep the best performing one, so four candidate parent sets are compared by five-fold
cross-validation. Parameters are learned by counting, as the slide shows: P(band | parents) = (homes with that
band and those parent values + 1) / (homes with those parent values + 5); the +1 (Laplace smoothing) keeps an unseen
band from getting probability 0. Every feature is known when a home is priced, so inference reduces to reading the
band's own table: the edges between features do not change the answer. One addition the slides do not cover: when
a home's parent values never occur in the fitting homes, the last parent is dropped until a seen combination is
found (back-off). Number features are cut at the quintiles of the fitting homes; a one-hot group becomes one value
per home (its baseline when every column of the group is 0). Trained and tested on all homes: it only counts.

The four steps are copied from band_runner.py, the runner written for these classifiers.

Run: python3 training_code/bayesian_network.py
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
from final_training import load_test_table
from neural_search import FOLDS, SEED, load_table
from band_models import band_edges, band_scores, price_bands
from slide_models import append_row, finished, setting_name
from shared_code import TABLE_DIR


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


BAND_COUNT = 5
NUMBER_BINS = 5
VALUE_RADIX = 1000     # every feature has fewer than 1,000 values (ZIP code has 326), so keys never collide
PARENT_SETS = [
    ["ZIP code", "living area (log)"],
    ["ZIP code", "living area (log)", "building quality"],
    ["ZIP code", "living area (log)", "year built"],
    ["ZIP code", "living area (log)", "building quality", "year built"],
]


class FeatureValues:
    """Turns rows of the 606-column table back into one discrete value per feature."""

    def __init__(self, columns):
        self.groups = {group: np.flatnonzero(columns["group"].to_numpy() == group) for group in columns["group"].unique()}
        self.kinds = dict(zip(columns["group"], columns["kind"]))

    def values(self, matrix, group, number_edges):
        block = matrix[:, self.groups[group]]
        if self.kinds[group] == "number":
            return np.searchsorted(number_edges[group], block.toarray().ravel(), side="right")
        if self.kinds[group] == "one-hot":
            dense = block.toarray()
            return np.where(dense.any(axis=1), dense.argmax(axis=1) + 1, 0)
        return block.toarray().ravel().astype(int)

    def number_edges(self, fit_matrix, groups):
        """Inner quintile edges of each number feature among the fitting homes."""
        quantiles = np.arange(1, NUMBER_BINS) / NUMBER_BINS
        return {g: np.unique(np.quantile(fit_matrix[:, self.groups[g]].toarray().ravel(), quantiles))
                for g in groups if self.kinds[g] == "number"}


class BandProbabilityTable:
    """P(band | parent values) learned by counting, for the full parent set and every shorter prefix of it."""

    def __init__(self, parent_values, bands):
        self.counts = [band_counts(combination_keys(parent_values[:size]), bands)
                       for size in range(len(parent_values), 0, -1)]
        self.most_common_band = np.bincount(bands).argmax()

    def predict(self, parent_values):
        predicted = np.zeros(len(parent_values[0]), dtype=int)
        unresolved = np.ones(len(predicted), dtype=bool)
        for size, counts in zip(range(len(parent_values), 0, -1), self.counts):
            keys = combination_keys(parent_values[:size])
            seen = unresolved & np.isin(keys, counts.index)
            predicted[seen] = most_probable_band(counts.reindex(keys[seen]).to_numpy())
            unresolved &= ~seen
        predicted[unresolved] = self.most_common_band   # only when even the first parent's value was never seen
        return predicted


def combination_keys(parent_values):
    """One integer per home for its combination of parent values."""
    key = np.zeros(len(parent_values[0]), dtype=np.int64)
    for values in parent_values:
        key = key * VALUE_RADIX + values
    return key


def band_counts(keys, bands):
    """Rows: parent combinations; columns: bands 1 to 5; cells: how many homes have both."""
    table = pd.crosstab(keys, bands)
    return table.reindex(columns=range(1, BAND_COUNT + 1), fill_value=0)


def most_probable_band(counts):
    """Laplace-smoothed P(band | parents) is (count + 1) / (total + 5): the same order as the counts themselves."""
    probabilities = (counts + 1) / (counts.sum(axis=1, keepdims=True) + BAND_COUNT)
    return probabilities.argmax(axis=1) + 1


class BayesianNetworkBands:
    name = "Bayesian network"
    description = ("Lecture 11: the price band is a child of location, size, quality and age; probability tables "
                   "learned by counting with Laplace smoothing; the best of four structures chosen by cross-validation.")
    settings = [{"parents": parents} for parents in PARENT_SETS]

    def __init__(self, features):
        self.features = features

    def predictions(self, settings, fit_matrix, fit_bands, query_matrix):
        for setting in settings:
            parents = setting["parents"]
            edges = self.features.number_edges(fit_matrix, parents)
            fit_values = [self.features.values(fit_matrix, p, edges) for p in parents]
            query_values = [self.features.values(query_matrix, p, edges) for p in parents]
            yield setting, BandProbabilityTable(fit_values, fit_bands).predict(query_values)


def main():
    features = FeatureValues(pd.read_csv(TABLE_DIR / "columns.csv"))
    run(BayesianNetworkBands(features), load_band_table(), "bayesian_network")


if __name__ == "__main__":
    main()

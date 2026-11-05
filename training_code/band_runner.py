"""The four steps every lecture classifier goes through on the five price bands.

1. Search: each setting is fitted on four folds and scored on the fifth, five times.
2. Winner: the setting with the highest average validation accuracy, among settings scored on all five folds.
3. Final fit on every training home of the table (or of its sample).
4. One score on all 571,860 test homes, never twice.

The bands are the same five as in zillow/band_models.py: equal fifths of all 2,287,442 training homes, so every
classifier is scored on the same task. A model trained on a sample still uses those edges and the full test set.
"""
# AI was used to help write this code, but the resulting code was reviewed and edited by a human.

import json
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse

import paths
import shared_code  # noqa: F401  (points the loaders at the Zillow table before they are imported)
from band_models import band_edges, band_scores, price_bands
from final_training import load_test_table
from neural_search import FOLDS, SEED, load_table
from slide_models import append_row, finished, setting_name

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

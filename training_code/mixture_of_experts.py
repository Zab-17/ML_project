"""The mixture of experts as Lecture 13 teaches it, on the five price bands.

Several expert classifiers are trained at the same time on the same homes, and a gating network with a softmax
gives each home a weight over the experts. Each expert is a linear softmax classifier over the 606 columns; the
probability of a band is the gate-weighted sum of the experts' probabilities. Experts and gate are trained together
by gradient descent on the log-likelihood, so the gate learns to trust, for each part of the data, the expert that
does best there. The number of experts is chosen by five-fold cross-validation. Training stops when the error on
10% of the fitting homes, held out for that purpose, has not improved for 5 passes, so the validation fold is never
used for stopping. Trained on all 2,287,442 training homes; tested on all 571,860 test homes. The first run used a random sample of
100,000 training homes; its results stay in results/mixture_of_experts/ and results/mixture_of_experts.md, and this run writes to
results/mixture_of_experts_all_homes/ and results/mixture_of_experts_all_homes.md, so nothing from the first run is skipped or overwritten.

The four steps are copied from band_runner.py, the runner written for these classifiers.

Run: caffeinate -is python3 -u training_code/mixture_of_experts.py   (about 1.5-3 hours on the M1 Pro, on the 67 W charger)
"""
# AI was used to help write this code, but the resulting code was reviewed and edited by a human.

from dataclasses import dataclass
import json
import time

from scipy import sparse
from torch import nn
import numpy as np
import pandas as pd
import torch

import paths  # noqa: F401  (puts zillow/ on the import path)
import shared_code  # noqa: F401  (points the loaders at the Zillow table)
from neural_search import FOLDS, SEED, load_table
from band_models import band_edges, band_scores, price_bands
from final_training import load_test_table
from slide_models import append_row, finished, setting_name


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
RESULTS_NAME = "mixture_of_experts_all_homes"   # a new folder: the first run's results are kept and nothing is skipped
EXPERT_COUNTS = [2, 4, 8]
BAND_COUNT = 5
BATCH = 1024
LEARNING_RATE = 0.001
MAX_PASSES = 100
PATIENCE = 5
STOPPING_SHARE = 0.1
PREDICTION_BATCH = 50_000


class MixtureNetwork(nn.Module):
    def __init__(self, column_count, expert_count):
        super().__init__()
        self.expert_count = expert_count
        self.experts = nn.Linear(column_count, expert_count * BAND_COUNT)
        self.gate = nn.Linear(column_count, expert_count)

    def forward(self, rows):
        """The log probability of every band for every row."""
        expert_log_probabilities = torch.log_softmax(
            self.experts(rows).view(-1, self.expert_count, BAND_COUNT), dim=2)
        gate_log_weights = torch.log_softmax(self.gate(rows), dim=1)
        return torch.logsumexp(gate_log_weights.unsqueeze(2) + expert_log_probabilities, dim=1)


class MixtureOfExpertsBands:
    name = "mixture of experts"
    description = ("Lecture 13: linear softmax experts trained together with a softmax gating network that weights "
                   "the experts for each home.")
    settings = [{"experts": e} for e in EXPERT_COUNTS]

    def predictions(self, settings, fit_matrix, fit_bands, query_matrix):
        for setting in settings:
            network = self.trained(setting["experts"], fit_matrix, fit_bands)
            yield setting, predicted_bands(network, query_matrix)

    def trained(self, expert_count, fit_matrix, fit_bands):
        torch.manual_seed(SEED)
        learning, stopping = stopping_split(fit_matrix.shape[0])
        rows, targets = fit_matrix[learning], torch.from_numpy(fit_bands[learning] - 1)       # stays sparse
        stop_rows, stop_targets = fit_matrix[stopping], torch.from_numpy(fit_bands[stopping] - 1)
        network = MixtureNetwork(fit_matrix.shape[1], expert_count)
        optimiser = torch.optim.Adam(network.parameters(), lr=LEARNING_RATE)
        best_error, best_state, passes_without_gain = float("inf"), None, 0
        for _ in range(MAX_PASSES):
            train_one_pass(network, optimiser, rows, targets)
            error = log_loss(network, stop_rows, stop_targets)
            if error < best_error:
                best_error, best_state, passes_without_gain = error, clone(network), 0
            else:
                passes_without_gain += 1
            if passes_without_gain >= PATIENCE:
                break
        network.load_state_dict(best_state)
        return network


def stopping_split(row_count):
    """Row numbers that learn and row numbers kept aside to decide when to stop."""
    order = np.random.default_rng(SEED).permutation(row_count)
    held = int(row_count * STOPPING_SHARE)
    return np.sort(order[held:]), np.sort(order[:held])


def train_one_pass(network, optimiser, rows, targets):
    """One pass in shuffled batches. Each batch is made dense on its own: all 1.65 million learning homes as one
    dense block would take about 4 GB."""
    network.train()
    for batch in torch.randperm(rows.shape[0]).split(BATCH):
        optimiser.zero_grad()
        nn.functional.nll_loss(network(dense(rows[batch.numpy()])), targets[batch]).backward()
        optimiser.step()


def log_loss(network, rows, targets):
    """The average negative log-likelihood, computed in parts so the rows are never one dense block."""
    network.eval()
    total = 0.0
    with torch.no_grad():
        for start in range(0, rows.shape[0], PREDICTION_BATCH):
            part = slice(start, start + PREDICTION_BATCH)
            total += nn.functional.nll_loss(network(dense(rows[part])), targets[part], reduction="sum").item()
    return total / rows.shape[0]


def predicted_bands(network, matrix):
    network.eval()
    with torch.no_grad():
        parts = [network(dense(matrix[start:start + PREDICTION_BATCH])).argmax(dim=1).numpy()
                 for start in range(0, matrix.shape[0], PREDICTION_BATCH)]
    return np.concatenate(parts) + 1


def dense(matrix):
    return torch.from_numpy(matrix.astype(np.float32).toarray())


def clone(network):
    return {name: value.clone() for name, value in network.state_dict().items()}


def main():
    run(MixtureOfExpertsBands(), load_band_table(SAMPLE_SIZE), RESULTS_NAME)


if __name__ == "__main__":
    main()

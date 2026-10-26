"""Five-fold search for the best neural network on the King County training sales; the test sales stay sealed.

Each fold holds out one fifth of the training homes (80% train, 20% validation, split by home), so every
training home is validated exactly once. Every setting trains once per fold, from fresh random weights,
until the validation error has not improved for PATIENCE passes (at most MAX_PASSES), and is scored at
its best pass. One pass = every training row of the fold, once, in batches of BATCH rows.

Run: python3 neural_search.py                 full search, every setting on all 5 folds
     python3 neural_search.py --smoke         a few settings, fold 0, SMOKE_PASSES passes: checks the code and timing
"""
# AI was used to help write this code, but the resulting code was reviewed and edited by a human.

import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import sparse
from sklearn.linear_model import LinearRegression

from train_models import scores

TABLE_DIR = Path(__file__).resolve().parent / "out" / "model_table"
OUT_DIR = TABLE_DIR.parent / "neural_search"
FOLDS = 5
BATCH = 1024
SCORE_BATCH = 8192
MAX_PASSES = 100
PATIENCE = 5
SMOKE_PASSES = 2
SEED = 42
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


@dataclass(frozen=True)
class Setting:
    first_layer: int
    second_layer: int
    optimizer: str        # "adam" or "sgd"
    learning_rate: float
    dropout: float

    def name(self):
        return (f"{self.first_layer}-{self.second_layer} {self.optimizer} lr {self.learning_rate:g} "
                f"dropout {self.dropout:g}")


SMOKE_SETTINGS = [Setting(256, 128, "adam", 0.1, 0.0), Setting(256, 128, "sgd", 0.1, 0.0),
                  Setting(256, 128, "adam", 0.001, 0.0)]
SIZES = [(256, 128), (128, 64), (512, 256)]
DROPOUTS = [0.0, 0.2]
# Learning rate 0.1 is Zeyad's choice; with Adam it diverged in the smoke test, with plain SGD it trained well.
SEARCH_SETTINGS = [Setting(first, second, "sgd", 0.1, dropout) for first, second in SIZES for dropout in DROPOUTS]


def main():
    smoke = "--smoke" in sys.argv
    matrix, label, folds = load_table()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    settings, fold_list, max_passes = ((SMOKE_SETTINGS, [0], SMOKE_PASSES) if smoke
                                       else (SEARCH_SETTINGS, range(FOLDS), MAX_PASSES))
    results_path = OUT_DIR / ("smoke_results.csv" if smoke else "fold_results.csv")
    done = finished_runs(results_path)
    for setting in settings:
        for fold in fold_list:
            if (setting.name(), fold) in done:
                continue
            result = train_one(setting, fold, matrix, label, folds, max_passes)
            pd.DataFrame([result]).to_csv(results_path, mode="a", header=not results_path.exists(), index=False)
    if not smoke:
        linear_regression_folds(matrix, label, folds)
        summarize(results_path)


def load_table():
    matrix = sparse.load_npz(TABLE_DIR / "train_matrix.npz").tocsr()
    label = np.load(TABLE_DIR / "train_label.npy")
    return matrix, label, np.load(TABLE_DIR / "train_folds.npy")


def finished_runs(results_path):
    if not results_path.exists():
        return set()
    done = pd.read_csv(results_path)
    return set(zip(done["setting"], done["fold"]))


def train_one(setting, fold, matrix, label, folds, max_passes):
    torch.manual_seed(SEED + fold)
    shuffler = np.random.default_rng(SEED + fold)
    fit_rows, check_rows = np.where(folds != fold)[0], np.where(folds == fold)[0]
    model = network(matrix.shape[1], setting).to(DEVICE)
    optimizer = make_optimizer(model, setting)
    best_error, best_pass, best_predictions = np.inf, 0, None
    start = time.perf_counter()
    for pass_number in range(1, max_passes + 1):
        train_pass(model, optimizer, matrix, label, shuffler.permutation(fit_rows))
        predictions = predict(model, matrix, check_rows)
        error = float(((predictions - label[check_rows]) ** 2).mean())
        log(f"{setting.name()} | fold {fold} | pass {pass_number} | validation MSE (log) {error:.4f} | "
            f"{time.perf_counter() - start:.0f} s")
        if error < best_error:
            best_error, best_pass, best_predictions = error, pass_number, predictions
        elif pass_number - best_pass >= PATIENCE or not np.isfinite(error):
            break
    return {"setting": setting.name(), **asdict(setting), "fold": fold, "best pass": best_pass,
            "passes run": pass_number, "seconds": round(time.perf_counter() - start, 1),
            **(scores(label[check_rows], best_predictions) if best_predictions is not None else {})}


def network(inputs, setting):
    return torch.nn.Sequential(
        torch.nn.Linear(inputs, setting.first_layer), torch.nn.ReLU(), torch.nn.Dropout(setting.dropout),
        torch.nn.Linear(setting.first_layer, setting.second_layer), torch.nn.ReLU(), torch.nn.Dropout(setting.dropout),
        torch.nn.Linear(setting.second_layer, 1))


def make_optimizer(model, setting):
    if setting.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=setting.learning_rate)
    if setting.optimizer == "sgd":
        return torch.optim.SGD(model.parameters(), lr=setting.learning_rate)
    raise ValueError(f"unknown optimizer: {setting.optimizer}")


def train_pass(model, optimizer, matrix, label, rows):
    model.train()
    for start in range(0, len(rows), BATCH):
        batch = rows[start:start + BATCH]
        inputs, targets = to_device(matrix[batch]), torch.from_numpy(label[batch].astype(np.float32)).to(DEVICE)
        loss = torch.nn.functional.mse_loss(model(inputs).squeeze(1), targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def predict(model, matrix, rows):
    model.eval()
    with torch.no_grad():
        parts = [model(to_device(matrix[rows[start:start + SCORE_BATCH]])).squeeze(1).cpu().numpy()
                 for start in range(0, len(rows), SCORE_BATCH)]
    return np.concatenate(parts).astype(np.float64)


def to_device(rows):
    return torch.from_numpy(rows.toarray().astype(np.float32)).to(DEVICE)


def log(line):
    print(line, flush=True)
    with open(OUT_DIR / "log.txt", "a") as file:
        file.write(line + "\n")


def linear_regression_folds(matrix, label, folds):
    """The same 5 folds for linear regression, so the networks are compared fairly."""
    path = OUT_DIR / "linear_regression_folds.csv"
    if path.exists():
        return
    rows = []
    for fold in range(FOLDS):
        fit_rows, check_rows = folds != fold, folds == fold
        model = LinearRegression().fit(matrix[fit_rows], label[fit_rows])
        rows.append({"fold": fold, **scores(label[check_rows], model.predict(matrix[check_rows]))})
    pd.DataFrame(rows).to_csv(path, index=False)


def summarize(results_path):
    results = pd.read_csv(results_path)
    summary = (results.groupby("setting")
               .agg(folds=("fold", "count"), mse_log=("MSE (log)", "mean"), mse_spread=("MSE (log)", "std"),
                    r2_log=("R2 (log)", "mean"), median_percent_off=("median % off", "mean"),
                    within_10=("within 10%", "mean"), best_pass=("best pass", "median"), minutes=("seconds", "sum"))
               .sort_values("mse_log"))
    summary["minutes"] = summary["minutes"] / 60
    summary.to_csv(OUT_DIR / "summary.csv")
    print(summary.round(4).to_string())
    linear = pd.read_csv(OUT_DIR / "linear_regression_folds.csv")
    print(f"linear regression, same 5 folds: MSE (log) {linear['MSE (log)'].mean():.4f}, R2 (log) "
          f"{linear['R2 (log)'].mean():.4f}, median % off {linear['median % off'].mean():.2f}, "
          f"within 10% {linear['within 10%'].mean():.2f}")


if __name__ == "__main__":
    main()

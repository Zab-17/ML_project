"""Neural network (Lecture 7).

Two hidden layers (ReLU), dropout, batches of 1,024, early stopping. Searched in zillow/neural_search.py:
256-128, 128-64 or 512-256 neurons x dropout 0 or 0.2 x SGD at 0.1 or Adam at 0.001 (12 settings, 5 folds).
The winner is retrained on every training home and scored once by zillow/final_training.py.
Results: training_code/results/neural_network.md

Everything below the imports is copied unchanged from the zillow/ modules that ran this training on
1-3 November 2026 (each block is marked with its source file); only the table loaders, the score
functions and the project settings are imported.

Run: python3 training_code/neural_network.py
"""
# AI was used to help write this code, but the resulting code was reviewed and edited by a human.

from dataclasses import asdict, dataclass
import json
import sys
import time

from scipy import sparse
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import torch

import paths  # noqa: F401  (puts zillow/ on the import path)
import shared_code  # noqa: F401  (points the loaders at the Zillow table)
from neural_search import OUT_DIR as SEARCH_DIR
from train_models import TABLE_DIR, comparison_models, scores


# ----- copied from zillow/final_training.py -----


FINAL_DIR = TABLE_DIR.parent / "final_model"
LOG = FINAL_DIR / "log.txt"
PREDICTIONS_DIR = FINAL_DIR / "test_predictions"


def complete_settings_ranking(fold_results):
    """Average validation MSE (log) and its spread per setting, best first, for settings whose 5 runs all scored."""
    scored = fold_results.dropna(subset=["MSE (log)"])
    complete = scored.groupby("setting").filter(lambda runs: len(runs) == FOLDS)
    return complete.groupby("setting")["MSE (log)"].agg(["mean", "std"]).sort_values("mean")


def winner(fold_results, setting_name):
    """The Setting behind a setting name, and the median best pass of its runs."""
    runs = fold_results[fold_results["setting"] == setting_name]
    first = runs.iloc[0]
    setting = Setting(int(first["first_layer"]), int(first["second_layer"]), first["optimizer"],
                      float(first["learning_rate"]), float(first["dropout"]))
    return setting, int(runs["best pass"].median())


def trained_network(setting, passes, matrix, label):
    """A network trained from fresh weights on every row of the matrix for a fixed number of passes."""
    torch.manual_seed(SEED)
    shuffler = np.random.default_rng(SEED)
    rows = np.arange(matrix.shape[0])
    model = network(matrix.shape[1], setting).to(DEVICE)
    optimizer = make_optimizer(model, setting)
    start = time.perf_counter()
    for pass_number in range(1, passes + 1):
        train_pass(model, optimizer, matrix, label, shuffler.permutation(rows))
        error = float(((predict(model, matrix, rows) - label) ** 2).mean())
        log(f"{setting.name()} | pass {pass_number} of {passes} | training MSE (log) {error:.4f} | "
            f"{time.perf_counter() - start:.0f} s", LOG)
        if not np.isfinite(error):
            raise FloatingPointError(f"the final training of {setting.name()} diverged at pass {pass_number}")
    return model


def load_test_table():
    return sparse.load_npz(TABLE_DIR / "test_matrix.npz").tocsr(), np.load(TABLE_DIR / "test_label.npy")


def test_predictions(model, matrix, label, test_matrix):
    """Log-price predictions for the test sales: the network, then linear regression and the baselines fitted on all
    of matrix."""
    predictions = {"neural network": predict(model, test_matrix, np.arange(test_matrix.shape[0]))}
    for name, comparison in comparison_models(label).items():
        predictions[name] = comparison.fit(matrix, label).predict(test_matrix)
    return predictions


def save_predictions(predictions):
    PREDICTIONS_DIR.mkdir(exist_ok=True)
    for name, predicted in predictions.items():
        np.save(PREDICTIONS_DIR / f"{name}.npy", predicted)


def test_results(predictions, test_label):
    return pd.DataFrame([{"model": name, **scores(test_label, predicted)} for name, predicted in predictions.items()])


def save_tables(choice, ranking, results):
    (FINAL_DIR / "choice.json").write_text(json.dumps(choice, indent=2))
    ranking.to_csv(FINAL_DIR / "search_ranking.csv")
    results.to_csv(FINAL_DIR / "test_results.csv", index=False)
    log(f"trained in {choice['training seconds']:.0f} s on {choice['training sales']:,} sales; "
        f"scored once on {choice['test sales']:,} test sales:", LOG)
    log(results.set_index("model").T.round(4).to_string(), LOG)


# ----- copied from zillow/neural_search.py -----


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
OPTIMIZERS = [("sgd", 0.1), ("adam", 0.001)]
SEARCH_SETTINGS = [Setting(first, second, optimizer, learning_rate, dropout)
                   for optimizer, learning_rate in OPTIMIZERS for first, second in SIZES for dropout in DROPOUTS]


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


def log(line, path=OUT_DIR / "log.txt"):
    print(line, flush=True)
    with open(path, "a") as file:
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
               .agg(folds_scored=("MSE (log)", "count"), mse_log=("MSE (log)", "mean"), mse_spread=("MSE (log)", "std"),
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


# ----- the main steps, copied from each module's main() and renamed -----
def run_search():
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


def run_final_training():
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    fold_results = pd.read_csv(SEARCH_DIR / "fold_results.csv")
    ranking = complete_settings_ranking(fold_results)
    setting, passes = winner(fold_results, ranking.index[0])
    log(f"winner: {setting.name()}, trained for {passes} passes (median best pass of its {FOLDS} runs)", LOG)
    matrix, label, _ = load_table()
    start = time.perf_counter()
    model = trained_network(setting, passes, matrix, label)
    choice = {"setting": setting.name(), **asdict(setting), "passes": passes,
              "training sales": int(matrix.shape[0]), "training seconds": round(time.perf_counter() - start, 1)}
    torch.save(model.state_dict(), FINAL_DIR / "network.pt")
    test_matrix, test_label = load_test_table()
    choice["test sales"] = int(test_matrix.shape[0])
    predictions = test_predictions(model, matrix, label, test_matrix)
    save_predictions(predictions)
    save_tables(choice, ranking, test_results(predictions, test_label))


TEST_RESULTS = FINAL_DIR / "test_results.csv"


def main():
    """zillow/final_training.py has no skip: running it again would retrain the network and change the reported
    scores. So the steps run only when their test results are missing."""
    if TEST_RESULTS.exists():
        print("Already trained and tested; the scores are in training_code/results/neural_network.md. Nothing was rerun.")
        return
    run_search()
    run_final_training()


if __name__ == "__main__":
    main()

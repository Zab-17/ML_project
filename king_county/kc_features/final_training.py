"""The final King County network: trained once on every training sale, then scored once on the sealed test sales.

1. Winner: the setting with the lowest average validation MSE (log) in the five-fold search, counting only
   settings whose 5 runs all trained (a run that diverged has no score).
2. Passes: the winner's median best pass. With every training sale in use, no validation fold is left to stop
   on, so the search decides how long to train.
3. Train the winner from fresh weights on all training sales, save its weights, then score it on the test
   sales next to linear regression and the two baselines, each fitted on all training sales.

Run: python3 final_training.py   (after neural_search.py)
Writes out/final_model/: network.pt, choice.json, search_ranking.csv, test_results.csv, log.txt.
"""
# AI was used to help write this code, but the resulting code was reviewed and edited by a human.

import json
import time
from dataclasses import asdict

import numpy as np
import pandas as pd
import torch
from scipy import sparse

from neural_search import (DEVICE, FOLDS, OUT_DIR as SEARCH_DIR, SEED, TABLE_DIR, Setting, load_table, log,
                           make_optimizer, network, predict, train_pass)
from train_models import comparison_models, scores

FINAL_DIR = TABLE_DIR.parent / "final_model"
LOG = FINAL_DIR / "log.txt"


def main():
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
    save_tables(choice, ranking, test_results(model, matrix, label, test_matrix, test_label))


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


def test_results(model, matrix, label, test_matrix, test_label):
    """Every score on the test sales: the network, then linear regression and the baselines fitted on all of matrix."""
    predictions = {"neural network": predict(model, test_matrix, np.arange(test_matrix.shape[0]))}
    for name, comparison in comparison_models(label).items():
        predictions[name] = comparison.fit(matrix, label).predict(test_matrix)
    return pd.DataFrame([{"model": name, **scores(test_label, predicted)} for name, predicted in predictions.items()])


def save_tables(choice, ranking, results):
    (FINAL_DIR / "choice.json").write_text(json.dumps(choice, indent=2))
    ranking.to_csv(FINAL_DIR / "search_ranking.csv")
    results.to_csv(FINAL_DIR / "test_results.csv", index=False)
    log(f"trained in {choice['training seconds']:.0f} s on {choice['training sales']:,} sales; "
        f"scored once on {choice['test sales']:,} test sales:", LOG)
    log(results.set_index("model").T.round(4).to_string(), LOG)


if __name__ == "__main__":
    main()

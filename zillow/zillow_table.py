"""The Zillow model table: the same steps as the King County pipeline, from the raw file to the model matrices.

1. Row rules and the 80/20 split (zillow_rows.py).
2. Features, learned from the training homes only (zillow_features.py).
3. Column and correlation checks; features the rules drop are removed (zillow_checks.py).
4. The King County encoder: numbers standardized with the training average and spread, yes/no columns left as
   0 or 1, categories one-hot with each group's most common value dropped as the baseline.
5. The King County column rules: one column of every pair at 0.85 or more goes (the one weaker with the label),
   then a check that no column is an exact combination of the others.

Run: python3 zillow_table.py [--sample homes]
Writes MODEL_TABLE_DIR (default out/model_table/) in the files the King County model scripts read, and the check
tables and heatmaps into out/checks/. --sample builds a small table from that many homes, for a quick trial run.
"""
# AI was used to help write this code, but the resulting code was reviewed and edited by a human.

import json
import sys

import numpy as np
import pandas as pd
from scipy import sparse

from shared_code import CHECKS_DIR, TABLE_DIR
from features import FeatureTable
from pipeline import encoded_columns, selected_columns
from zillow_checks import run_checks
from zillow_features import build_feature_table
from zillow_rows import split_rows


def main():
    rows = split_rows(sample_homes())
    everything = pd.concat([rows.train, rows.test], ignore_index=True)
    fit = np.arange(len(everything)) < len(rows.train)
    table = build_feature_table(everything, fit)
    dropped = run_checks(rows.train, rows.folds, training_part(table, fit))
    report_drops(dropped)
    table = without_features(table, dropped)
    matrix, columns = encoded_columns(table, fit)
    keep = selected_columns(matrix[fit], columns, table.label[fit].to_numpy())
    save(matrix[:, keep], columns.iloc[keep], table.label.to_numpy(), fit, rows)
    save_row_report(rows, dropped, columns.iloc[keep])


def sample_homes():
    return int(sys.argv[sys.argv.index("--sample") + 1]) if "--sample" in sys.argv else None


def training_part(table, fit):
    return FeatureTable(table.numbers[fit], table.yes_no[fit],
                        {group: labels[fit] for group, labels in table.categories.items()},
                        table.label[fit], table.month_adjusted_label[fit])


def report_drops(dropped):
    print(f"features dropped by the correlation rules: {len(dropped)}")
    for name, reason in dropped.items():
        print(f"  {name}: {reason}")


def without_features(table, dropped):
    return FeatureTable(table.numbers.drop(columns=[d for d in dropped if d in table.numbers]),
                        table.yes_no.drop(columns=[d for d in dropped if d in table.yes_no]),
                        {group: labels for group, labels in table.categories.items() if group not in dropped},
                        table.label, table.month_adjusted_label)


def save(matrix, columns, label, fit, rows):
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    sparse.save_npz(TABLE_DIR / "train_matrix.npz", matrix[fit])
    sparse.save_npz(TABLE_DIR / "test_matrix.npz", matrix[~fit])
    np.save(TABLE_DIR / "train_label.npy", label[fit])
    np.save(TABLE_DIR / "test_label.npy", label[~fit])
    np.save(TABLE_DIR / "train_folds.npy", rows.folds)
    np.save(TABLE_DIR / "validation_homes.npy", rows.folds == 0)
    columns.to_csv(TABLE_DIR / "columns.csv", index=False)
    print(f"saved: {fit.sum():,} training homes, {(~fit).sum():,} test homes, {len(columns):,} columns "
          f"{columns['kind'].value_counts().to_dict()}, from {columns['group'].nunique()} features")


def save_row_report(rows, dropped, columns):
    report = {"row rules (homes removed by each, in order)": rows.dropped_by_rule,
              "training homes": len(rows.train), "test homes": len(rows.test),
              "homes per fold": np.bincount(rows.folds).tolist(),
              "features dropped by the correlation rules": dropped,
              "model columns per feature": columns.groupby("group").size().to_dict()}
    (CHECKS_DIR / "table_report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

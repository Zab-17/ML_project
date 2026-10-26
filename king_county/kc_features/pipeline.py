"""The final King County model table: training and test sales, encoded and scaled from the training sales only.

Steps, in the order FEATURE_DECISIONS.md settles them:
1. build the 60 features for training and test sales together, learning every list from the training sales;
2. drop the 5 features at 0.85 or more with a feature closer to the label (25 October 2026);
3. standardize the numbers with the training average and spread (yes/no columns stay 0 or 1);
4. one-hot the categories and drop each group's most common column, the baseline (D3);
5. drop one column of every remaining pair at 0.85 or more, keeping the one closer to the label;
6. check that no column is a combination of the others.

Run: python3 pipeline.py <king folder>
Writes out/model_table/: the training and test matrices, the labels, the column list and the validation homes.
"""
# AI was used to help write this code, but the resulting code was reviewed and edited by a human.

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from correlations import correlation_matrix
from features import FeatureTable, build_feature_table, one_hot, split_rows, without_sales_missing_rooms
from location_table import HELD_OUT_SHARE

OUT_DIR = Path(__file__).resolve().parent / "out" / "model_table"
SUBDIVISION_MIN_SALES, STREET_MIN_SALES = 100, 200   # D4 and D5
DROPPED_FEATURES = ["appraisal area", "miles north", "miles east", "tidelands or shorelands", "city"]
PAIR_LIMIT = 0.85
ZERO_EIGENVALUE = 1e-8


def main():
    train, test, _ = split_rows()
    train, test = without_sales_missing_rooms(train), without_sales_missing_rooms(test)
    rows = pd.concat([train, test], ignore_index=True)
    fit = np.arange(len(rows)) < len(train)
    table = without_features(build_feature_table(rows, SUBDIVISION_MIN_SALES, STREET_MIN_SALES, fit),
                             DROPPED_FEATURES)
    matrix, columns = encoded_columns(table, fit)
    keep = selected_columns(matrix[fit], columns, table.label[fit].to_numpy())
    save(matrix[:, keep], columns.iloc[keep], table.label.to_numpy(), fit, home_folds(train))


def home_folds(train):
    """Fold 0 to 4 per training sale, from the home's parcel number, so every sale of one home shares a fold.
    Fold 0 is the validation part used so far."""
    return (pd.util.hash_pandas_object(train["parcel"], index=False) % HELD_OUT_SHARE).to_numpy()


def without_features(table, dropped):
    return FeatureTable(table.numbers.drop(columns=[d for d in dropped if d in table.numbers]),
                        table.yes_no.drop(columns=[d for d in dropped if d in table.yes_no]),
                        {group: labels for group, labels in table.categories.items() if group not in dropped},
                        table.label, table.month_adjusted_label)


def encoded_columns(table, fit):
    """Standardized numbers, then yes/no columns, then one-hot columns, as one sparse matrix."""
    average, spread = table.numbers[fit].mean(), table.numbers[fit].std()
    numbers = (table.numbers - average) / spread
    hot, hot_names, hot_groups = one_hot(table.categories)
    columns = pd.DataFrame({
        "column": list(numbers.columns) + list(table.yes_no.columns) + hot_names,
        "group": list(numbers.columns) + list(table.yes_no.columns) + hot_groups,
        "kind": ["number"] * numbers.shape[1] + ["yes/no"] * table.yes_no.shape[1] + ["one-hot"] * len(hot_names),
        "training average": list(average) + [np.nan] * (table.yes_no.shape[1] + len(hot_names)),
        "training spread": list(spread) + [np.nan] * (table.yes_no.shape[1] + len(hot_names)),
    })
    dense = sparse.csc_matrix(np.hstack([numbers.to_numpy(), table.yes_no.to_numpy()]))
    return sparse.hstack([dense, hot]).tocsr(), columns


def selected_columns(fit_matrix, columns, label):
    keep = without_baselines(fit_matrix, columns)
    with_label = sparse.hstack([sparse.csc_matrix(label.reshape(-1, 1)), fit_matrix[:, keep]]).tocsc()
    correlation = correlation_matrix(with_label)
    survivors = without_close_pairs(correlation[1:, 1:], np.abs(correlation[0, 1:]), columns.iloc[keep])
    keep = keep[survivors]
    check_no_combinations(correlation[1:, 1:][np.ix_(survivors, survivors)])
    return keep


def without_baselines(fit_matrix, columns):
    """Positions of all columns except each one-hot group's most common column (D3)."""
    ones = np.asarray(fit_matrix.sum(axis=0)).ravel()
    hot = columns[columns["kind"] == "one-hot"].assign(ones=ones[columns["kind"] == "one-hot"])
    baselines = hot.groupby("group")["ones"].idxmax().to_numpy()
    print(f"baselines dropped ({len(baselines)}): " + "; ".join(columns.loc[baselines, "column"]))
    return np.setdiff1d(np.arange(len(columns)), baselines)


def without_close_pairs(correlation, with_label, columns):
    """Drop one column of every pair at PAIR_LIMIT or more, strongest pair first; the column closer to the label stays."""
    upper = np.abs(np.triu(correlation, k=1))
    rows, cols = np.where(upper >= PAIR_LIMIT)
    dropped = set()
    for row, col in sorted(zip(rows, cols), key=lambda pair: -upper[pair]):
        if row in dropped or col in dropped:
            continue
        weaker = row if with_label[row] < with_label[col] else col
        dropped.add(weaker)
        print(f"  pair {correlation[row, col]:+.3f}: drop {columns['column'].iloc[weaker]} "
              f"(keep {columns['column'].iloc[col if weaker == row else row]})")
    print(f"columns dropped for a pair at {PAIR_LIMIT} or more: {len(dropped)}")
    return np.array([i for i in range(len(correlation)) if i not in dropped])


def check_no_combinations(correlation):
    eigenvalues = np.linalg.eigvalsh(correlation)
    zero = int((eigenvalues < ZERO_EIGENVALUE).sum())
    if zero:
        raise ValueError(f"{zero} columns are still exact combinations of other columns")
    share = 1 - 1 / np.diag(np.linalg.inv(correlation))
    print(f"no column is a combination of others (smallest eigenvalue {eigenvalues[0]:.2e}); "
          f"most predictable column: {share.max():.3f} of it predicted by the others")


def save(matrix, columns, label, fit, folds):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sparse.save_npz(OUT_DIR / "train_matrix.npz", matrix[fit])
    sparse.save_npz(OUT_DIR / "test_matrix.npz", matrix[~fit])
    np.save(OUT_DIR / "train_label.npy", label[fit])
    np.save(OUT_DIR / "test_label.npy", label[~fit])
    np.save(OUT_DIR / "train_folds.npy", folds)
    np.save(OUT_DIR / "validation_homes.npy", folds == 0)
    columns.to_csv(OUT_DIR / "columns.csv", index=False)
    kinds = columns["kind"].value_counts().to_dict()
    print(f"saved: {fit.sum():,} training sales, {(~fit).sum():,} test sales, {len(columns):,} columns {kinds}, "
          f"from {columns['group'].nunique()} features")


if __name__ == "__main__":
    main()

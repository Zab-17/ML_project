"""Correlation of every final column with the label, and of every column with every other column.

Run: python3 correlations.py <king folder>
Writes out/label_correlations.csv and out/feature_pairs.csv, and prints a summary.
"""
# AI was used to help write this code, but the resulting code was reviewed and edited by a human.

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from features import final_training_table, one_hot

OUT_DIR = Path(__file__).resolve().parent / "out"
LABEL_LIMIT = 0.85       # professor's rule 4: a column this close to the label leaks the answer
LISTED_PAIR = 0.7        # feature pairs at or above this are printed
SAVED_PAIR = 0.5         # feature pairs at or above this are saved
TOP_LABEL_ROWS = 25
LABEL_COLUMNS = ["label: log price", "label: month-adjusted"]


def main():
    table = final_training_table()
    matrix, columns = stacked_columns(table)
    correlation = correlation_matrix(matrix)
    label_table = label_correlations(correlation, columns)
    pairs = feature_pairs(correlation, columns)
    OUT_DIR.mkdir(exist_ok=True)
    label_table.to_csv(OUT_DIR / "label_correlations.csv", index=False)
    pairs[pairs["abs r"] >= SAVED_PAIR].to_csv(OUT_DIR / "feature_pairs.csv", index=False)
    print_summary(matrix.shape[0], columns, label_table, pairs)


def stacked_columns(table):
    """Every column side by side: the two labels, the numbers, the yes/no columns, then the one-hot columns."""
    dense = pd.concat([table.label.rename(LABEL_COLUMNS[0]), table.month_adjusted_label.rename(LABEL_COLUMNS[1]),
                       table.numbers, table.yes_no], axis=1)
    dense = (dense - dense.mean()) / dense.std()  # centring keeps the sums precise; a correlation ignores scale
    hot, hot_names, hot_groups = one_hot(table.categories)
    kinds = ["label"] * 2 + ["number"] * table.numbers.shape[1] + ["yes/no"] * table.yes_no.shape[1]
    columns = pd.DataFrame({"column": list(dense.columns) + hot_names,
                            "group": list(dense.columns) + hot_groups,
                            "kind": kinds + ["one-hot"] * len(hot_names)})
    columns["sales with a 1"] = np.nan
    yes_counts = np.concatenate([table.yes_no.sum().to_numpy(), np.asarray(hot.sum(axis=0)).ravel()])
    columns.loc[columns["kind"].isin(["yes/no", "one-hot"]), "sales with a 1"] = yes_counts
    return sparse.hstack([sparse.csc_matrix(dense.to_numpy()), hot]).tocsc(), columns


def correlation_matrix(matrix):
    sales = matrix.shape[0]
    means = np.asarray(matrix.mean(axis=0)).ravel()
    covariance = (matrix.T @ matrix).toarray() / sales - np.outer(means, means)
    spread = np.sqrt(np.clip(np.diag(covariance), 0, None))
    if (spread == 0).any():
        raise ValueError(f"{(spread == 0).sum()} columns never change, so their correlation is undefined")
    return covariance / np.outer(spread, spread)


def label_correlations(correlation, columns):
    features = columns.iloc[len(LABEL_COLUMNS):].copy()
    features["r with label"] = correlation[0, len(LABEL_COLUMNS):]
    features["r with month-adjusted label"] = correlation[1, len(LABEL_COLUMNS):]
    return features.sort_values("r with label", key=np.abs, ascending=False).reset_index(drop=True)


def feature_pairs(correlation, columns):
    first = len(LABEL_COLUMNS)
    rows, cols = np.triu_indices(correlation.shape[0] - first, k=1)
    values = correlation[first:, first:][rows, cols]
    keep = np.abs(values) >= SAVED_PAIR
    features = columns.iloc[first:].reset_index(drop=True)
    left, right = features.iloc[rows[keep]].reset_index(drop=True), features.iloc[cols[keep]].reset_index(drop=True)
    pairs = pd.DataFrame({"column A": left["column"], "sales A": left["sales with a 1"],
                          "column B": right["column"], "sales B": right["sales with a 1"],
                          "same group": left["group"] == right["group"], "r": values[keep]})
    pairs["abs r"] = pairs["r"].abs()
    return pairs.sort_values("abs r", ascending=False).reset_index(drop=True)


def print_summary(sales, columns, label_table, pairs):
    kinds = columns["kind"].value_counts()
    print(f"===== {sales:,} training sales | {len(columns) - len(LABEL_COLUMNS):,} columns: "
          + ", ".join(f"{kinds.get(k, 0)} {k}" for k in ["number", "yes/no", "one-hot"]))
    print_label_side(label_table)
    print_pair_side(pairs)


def print_label_side(label_table):
    over = label_table[label_table["r with label"].abs() >= LABEL_LIMIT]
    print(f"===== columns with |r| >= {LABEL_LIMIT} with the label: {len(over)}")
    print(f"----- top {TOP_LABEL_ROWS} by |r with label|")
    for _, row in label_table.head(TOP_LABEL_ROWS).iterrows():
        print(f"  {row['column'][:60]:60s} {row['r with label']:+.3f}  (month-adjusted {row['r with month-adjusted label']:+.3f})")
    print("----- strongest column per group (|r with label|), one-hot groups")
    hot = label_table[label_table["kind"] == "one-hot"]
    best = hot.loc[hot.groupby("group")["r with label"].apply(lambda r: r.abs().idxmax())]
    for _, row in best.sort_values("r with label", key=np.abs, ascending=False).iterrows():
        print(f"  {row['group']:24s} {row['r with label']:+.3f}  {row['column'][:60]}")


def print_pair_side(pairs):
    listed = pairs[pairs["abs r"] >= LISTED_PAIR]
    for same, title in [(False, "different groups"), (True, "the same one-hot group")]:
        part = listed[listed["same group"] == same]
        print(f"===== feature pairs with |r| >= {LISTED_PAIR}, {title}: {len(part)}")
        for _, row in part.iterrows():
            print(f"  {row['r']:+.3f}  {row['column A'][:55]:55s} {sales_text(row['sales A'])}| "
                  f"{row['column B'][:55]:55s} {sales_text(row['sales B'])}")
    for limit in (0.9, 0.8, 0.5):
        print(f"  pairs with |r| >= {limit}: {(pairs['abs r'] >= limit).sum()}")


def sales_text(count):
    return "" if pd.isna(count) else f"({count:,.0f} sales) "


if __name__ == "__main__":
    main()

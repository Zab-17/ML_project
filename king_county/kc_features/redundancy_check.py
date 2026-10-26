"""Applies D1 to D5 and checks that no final column can be computed from the other columns.

Pairwise correlation only sees two columns at a time. A column that equals a sum of several others
(an area column equals the sum of its sub-area columns) shows up instead as an eigenvalue of 0 in the
correlation matrix, so this script counts those and names the columns behind each one.

Run: python3 redundancy_check.py <king folder>
"""
# AI was used to help write this code, but the resulting code was reviewed and edited by a human.

import numpy as np
import pandas as pd

from correlations import correlation_matrix, feature_pairs, stacked_columns
from features import final_training_table
from final_count import DUPLICATE_LIMIT, STREET_MIN_SALES, SUBDIVISION_MIN_SALES, column_to_drop

ZERO_EIGENVALUE = 1e-8     # below this, one column is (almost exactly) a combination of others
LOADING_SHOWN = 0.1        # columns weighing at least this much in a zero direction are named
DIRECTIONS_SHOWN = 10


def main():
    table = final_training_table(SUBDIVISION_MIN_SALES, STREET_MIN_SALES)
    matrix, columns = stacked_columns(table)
    correlation = correlation_matrix(matrix)
    keep = final_columns(correlation, columns)
    final = correlation[np.ix_(keep, keep)]
    names = columns["column"].to_numpy()[keep]
    print(f"final columns after D1 to D3: {len(names)}")
    print(columns.iloc[keep].groupby("kind").size().to_string())
    report_pairs(final, names)
    report_zero_directions(final, names)


def final_columns(correlation, columns):
    """Positions of the columns left after the duplicate rule (D1, D2) and one baseline per group (D3)."""
    group_of = dict(zip(columns["column"], columns["group"]))
    pairs = feature_pairs(correlation, columns)
    duplicates = pairs[(pairs["abs r"] >= DUPLICATE_LIMIT) & ~pairs["same group"]]
    dropped = {column_to_drop(row, group_of) for _, row in duplicates.iterrows()}
    remaining = columns[(columns["kind"] != "label") & ~columns["column"].isin(dropped)]
    hot = remaining[remaining["kind"] == "one-hot"]
    baselines = set(hot.loc[hot.groupby("group")["sales with a 1"].idxmax(), "column"])
    print("D3 baselines dropped: " + "; ".join(sorted(baselines)))
    return remaining[~remaining["column"].isin(baselines)].index.to_numpy()


def report_pairs(final, names):
    upper = np.abs(np.triu(final, k=1))
    row, col = np.unravel_index(upper.argmax(), upper.shape)
    print(f"strongest remaining pair: {final[row, col]:+.3f}  {names[row]} | {names[col]}")
    for limit in (0.95, 0.9, 0.8, 0.7):
        print(f"  remaining pairs at |r| >= {limit}: {(upper >= limit).sum()}")


def report_zero_directions(final, names):
    values, vectors = np.linalg.eigh(final)
    zero = np.where(values < ZERO_EIGENVALUE)[0]
    print(f"eigenvalues below {ZERO_EIGENVALUE}: {len(zero)} (0 means no column is a combination of others) | "
          f"smallest {values[0]:.2e}, largest {values[-1]:.1f}")
    for direction in zero[:DIRECTIONS_SHOWN]:
        weights = pd.Series(vectors[:, direction], index=names)
        heavy = weights[weights.abs() >= LOADING_SHOWN].sort_values(key=np.abs, ascending=False)
        print(f"  zero direction {direction}: " + "; ".join(f"{name} {weight:+.2f}" for name, weight in heavy.head(8).items()))


if __name__ == "__main__":
    main()

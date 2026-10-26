"""Counts the final columns once every suggestion is applied (D1 to D5).

Run: python3 final_count.py <king folder>
"""
# AI was used to help write this code, but the resulting code was reviewed and edited by a human.

import pandas as pd

from correlations import correlation_matrix, feature_pairs, stacked_columns
from features import final_training_table

SUBDIVISION_MIN_SALES = 100   # D4
STREET_MIN_SALES = 200        # D5
DUPLICATE_LIMIT = 0.95        # D1 and D2: pairs at or above this lose one column
DETAIL_ORDER = ["city", "appraisal area", "area + sub-area", "street name", "subdivision"]  # least to most detailed
WATERFRONT_PAIR = {"waterfront", "waterfront bank"}


def main():
    table = final_training_table(SUBDIVISION_MIN_SALES, STREET_MIN_SALES)
    matrix, columns = stacked_columns(table)
    pairs = feature_pairs(correlation_matrix(matrix), columns)
    group_of = dict(zip(columns["column"], columns["group"]))
    duplicates = pairs[(pairs["abs r"] >= DUPLICATE_LIMIT) & ~pairs["same group"]]
    dropped = {column_to_drop(row, group_of) for _, row in duplicates.iterrows()}
    features = columns[columns["kind"] != "label"]
    final = features[~features["column"].isin(dropped)]
    print(f"before D1/D2: {len(features)} columns | duplicate pairs at {DUPLICATE_LIMIT}+: {len(duplicates)} | dropped {len(dropped)}")
    for column in sorted(dropped):
        print(f"  dropped: {column}")
    print(f"FINAL: {len(final)} columns")
    print(final.groupby("kind").size().to_string())
    print(final[final["kind"] == "one-hot"].groupby("group").size().sort_values(ascending=False).to_string())


def column_to_drop(row, group_of):
    a, b = row["column A"], row["column B"]
    groups = {group_of[a], group_of[b]}
    if groups == WATERFRONT_PAIR:
        return a if group_of[a] == "waterfront bank" else b
    if not groups <= set(DETAIL_ORDER):
        raise ValueError(f"no drop rule for the pair {a} | {b}")
    return a if DETAIL_ORDER.index(group_of[a]) > DETAIL_ORDER.index(group_of[b]) else b


if __name__ == "__main__":
    main()

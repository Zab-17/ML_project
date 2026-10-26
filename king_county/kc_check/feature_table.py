"""Read-only report for a batch of columns: value table, share explained, correlation with the label.

Run: python3 feature_table.py <king folder> <column>:<kind> ...
Kinds: count (each value is a row), decade (years grouped by ten),
renovation (0 = never, else decade), code (category codes, names from the cards page),
yesno (0 = no, any value above 0 = yes).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "kc_cards"))
from measure import training_rows  # noqa: E402

MIN_ROW_SHARE = 0.001  # rows under 0.1% of sales fold into "other"
YES_NO_KINDS = {"renovation", "yesno"}  # correlation is measured on "above 0" for these


def code_names(column):
    cards = json.loads((HERE / "payload.json").read_text())["cards"]
    card = next(c for c in cards if c["name"] == column)
    return {item[0]: item[1] for item in (card.get("bars") or {}).get("items", [])}


def groups_for(values, kind):
    if kind == "yesno":
        return (values > 0).map({True: "yes", False: "no"})
    if kind == "decade":
        return (values // 10 * 10).astype("Int64").astype(str) + "s"
    if kind == "renovation":
        decade = (values // 10 * 10).astype("Int64").astype(str) + "s"
        return decade.where(values > 0, "never renovated")
    return values.astype(str)


def explained_share(groups, target):
    group_mean = target.groupby(groups).transform("mean")
    return 1 - ((target - group_mean) ** 2).sum() / ((target - target.mean()) ** 2).sum()


def print_table(groups, adjusted, names):
    table = adjusted.groupby(groups).agg(["size", "median"])
    table["share"] = table["size"] / len(adjusted) * 100
    rare = table["share"] < MIN_ROW_SHARE * 100
    for value, row in table[~rare].iterrows():
        label = f"{value} {names.get(value, '')}".strip()
        print(f"  {label:45s} {row['share']:6.2f}%  {np.expm1(row['median']) * 100:+6.1f}%")
    if rare.any():
        print(f"  other ({rare.sum()} rare values)               {table.loc[rare, 'share'].sum():6.2f}%")


def print_correlations(numbers, label, adjusted, kind):
    if kind == "code":
        print("  correlation: not meaningful, the codes are names, not amounts")
        return
    if kind in YES_NO_KINDS:
        numbers = (numbers > 0).astype(float)
        print("  (correlation below is for yes/no: above 0)")
    r_label, r_adjusted = numbers.corr(label), numbers.corr(adjusted)
    print(f"  correlation with label {r_label:.3f} | with month-adjusted label {r_adjusted:.3f} "
          f"(straight line explains {r_adjusted ** 2 * 100:.1f}%) | rank correlation {numbers.corr(adjusted, method='spearman'):.3f}")


def report(train, label, adjusted, column, kind):
    numbers = pd.to_numeric(train[column], errors="coerce")
    groups = groups_for(numbers, kind)
    print(f"===== {column} ({kind}) | empty {numbers.isna().mean() * 100:.2f}% | "
          f"as groups explains {explained_share(groups, adjusted) * 100:.1f}% of the month-adjusted label")
    print_table(groups, adjusted, code_names(column) if kind == "code" else {})
    print_correlations(numbers, label, adjusted, kind)


def main():
    train, _ = training_rows()
    label = np.log(train["_price"])
    adjusted = label - label.groupby(train["_date"].dt.to_period("M")).transform("median")
    for spec in sys.argv[2:]:
        column, kind = spec.split(":")
        report(train, label, adjusted, column, kind)


if __name__ == "__main__":
    main()

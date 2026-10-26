"""Read-only report for the location columns: size of each grouping, share explained on held-out homes,
the most common / cheapest / most expensive groups, and how much each column adds on top of appraisal area.

Run: python3 location_table.py <king folder>
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "kc_cards"))
from measure import training_rows  # noqa: E402

HELD_OUT_SHARE = 5          # one home in five is held out to score each grouping
MIN_SALES_FOR_EXTREMES = 500
ROWS_PER_LIST = 5


def text(train, column):
    return train[column].astype(str).str.strip().replace({"": "empty", "nan": "empty"})


def location_groupings(train):
    area = text(train, "Area")
    return {
        "ZIP code": text(train, "ZipCode").str[:5],
        "City": text(train, "DistrictName"),
        "Appraisal area": area,
        "Appraisal area + sub-area": area + "-" + text(train, "SubArea"),
        "Tax district code": text(train, "LevyCode"),
    }


def held_out_mask(train):
    parcel_hash = pd.util.hash_pandas_object(train["parcel"], index=False)
    return (parcel_hash % HELD_OUT_SHARE == 0).to_numpy()


def held_out_explained(groups, target, held_out):
    fit_means = target[~held_out].groupby(groups[~held_out]).mean()
    predicted = groups[held_out].map(fit_means).fillna(target[~held_out].mean())
    actual = target[held_out]
    return 1 - ((actual - predicted) ** 2).sum() / ((actual - actual.mean()) ** 2).sum()


def print_group_lists(groups, adjusted):
    table = adjusted.groupby(groups).agg(["size", "median"])
    table["share"] = table["size"] / len(adjusted) * 100
    table["gap"] = np.expm1(table["median"]) * 100
    big = table[table["size"] >= MIN_SALES_FOR_EXTREMES]
    for title, rows in [("most common", table.nlargest(ROWS_PER_LIST, "size")),
                        ("cheapest", big.nsmallest(ROWS_PER_LIST, "gap")),
                        ("most expensive", big.nlargest(ROWS_PER_LIST, "gap"))]:
        print(f"  {title}: " + " | ".join(f"{name} {row['share']:.1f}% {row['gap']:+.0f}%" for name, row in rows.iterrows()))


def report_single(name, groups, adjusted, held_out):
    explained = held_out_explained(groups, adjusted, held_out)
    sizes = groups.value_counts()
    print(f"===== {name}: {len(sizes)} groups, median {sizes.median():.0f} sales per group, "
          f"empty {(groups == 'empty').mean() * 100:.1f}% | explains {explained * 100:.1f}% on held-out homes "
          f"(correlation ratio {np.sqrt(max(explained, 0)):.2f})")
    print_group_lists(groups, adjusted)


def report_added_to_area(groupings, adjusted, held_out):
    area = groupings["Appraisal area"]
    base = held_out_explained(area, adjusted, held_out)
    print(f"===== added on top of appraisal area alone ({base * 100:.1f}%):")
    for name, groups in groupings.items():
        if name == "Appraisal area":
            continue
        crossed = area + "|" + groups
        print(f"  area x {name}: {held_out_explained(crossed, adjusted, held_out) * 100:.1f}% "
              f"({crossed.nunique()} groups)")


def main():
    train, _ = training_rows()
    label = np.log(train["_price"])
    adjusted = label - label.groupby(train["_date"].dt.to_period("M")).transform("median")
    held_out = held_out_mask(train)
    groupings = location_groupings(train)
    for name, groups in groupings.items():
        report_single(name, groups, adjusted, held_out)
    report_added_to_area(groupings, adjusted, held_out)


if __name__ == "__main__":
    main()

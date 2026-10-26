"""Read-only report: subdivision name and the survey grid (range, township, section, quarter).

The Public Land Survey grid numbers townships north and ranges east in 6-mile steps, and numbers the
36 one-mile sections of a township in a back-and-forth pattern starting at the north-east corner.
That lets each parcel's grid code be turned into an approximate position in miles.

Run: python3 grid_table.py <king folder>
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "kc_cards"))
from measure import training_rows  # noqa: E402
from location_table import held_out_explained, held_out_mask, print_group_lists, text  # noqa: E402

MILES_PER_TOWNSHIP = 6
QUARTER_OFFSET = 0.25


def grid_numbers(train):
    return {name: pd.to_numeric(train[name], errors="coerce") for name in ["Range", "Township", "Section"]}


def section_column_and_row(section):
    """Column counted from the west edge and row counted from the north edge, both 0 to 5."""
    row = (section - 1) // 6
    step = (section - 1) % 6
    column = np.where(row % 2 == 0, 5 - step, step)
    return column, row


def approximate_position(train):
    grid = grid_numbers(train)
    column, row = section_column_and_row(grid["Section"])
    quarter = text(train, "QuarterSection")
    east = (grid["Range"] - 1) * MILES_PER_TOWNSHIP + column + 0.5
    north = (grid["Township"] - 1) * MILES_PER_TOWNSHIP + (5 - row) + 0.5
    east += np.where(quarter.str.endswith("E"), QUARTER_OFFSET, -QUARTER_OFFSET)
    north += np.where(quarter.str.startswith("N"), QUARTER_OFFSET, -QUARTER_OFFSET)
    return east, north


def groupings(train):
    square_mile = text(train, "Township") + "-" + text(train, "Range") + "-" + text(train, "Section")
    return {
        "Subdivision name": text(train, "PlatName"),
        "Range": text(train, "Range"),
        "Township": text(train, "Township"),
        "Section": text(train, "Section"),
        "Quarter section": text(train, "QuarterSection"),
        "Square-mile cell (township-range-section)": square_mile,
        "Quarter-mile cell (+ quarter)": square_mile + "-" + text(train, "QuarterSection"),
    }


def report_groupings(all_groups, adjusted, held_out):
    for name, groups in all_groups.items():
        sizes = groups.value_counts()
        explained = held_out_explained(groups, adjusted, held_out)
        print(f"===== {name}: {len(sizes)} groups, median {sizes.median():.0f} sales per group, "
              f"empty {(groups == 'empty').mean() * 100:.1f}% | explains {explained * 100:.1f}% on held-out homes "
              f"(correlation ratio {np.sqrt(max(explained, 0)):.2f})")
        if name in ("Subdivision name", "Quarter section"):
            print_group_lists(groups, adjusted)


def report_added_to_sub_area(train, all_groups, adjusted, held_out):
    sub_area = text(train, "Area") + "-" + text(train, "SubArea")
    print(f"===== on top of area + sub-area ({held_out_explained(sub_area, adjusted, held_out) * 100:.1f}%):")
    for name in ["Subdivision name", "Square-mile cell (township-range-section)", "Quarter-mile cell (+ quarter)"]:
        crossed = sub_area + "|" + all_groups[name]
        print(f"  + {name}: {held_out_explained(crossed, adjusted, held_out) * 100:.1f}% ({crossed.nunique()} groups)")


def report_position(train, label, adjusted):
    east, north = approximate_position(train)
    for name, miles in [("miles east", east), ("miles north", north)]:
        print(f"  {name}: range {miles.min():.1f} to {miles.max():.1f} | correlation with label {miles.corr(label):.3f}, "
              f"month-adjusted {miles.corr(adjusted):.3f}")


def main():
    train, _ = training_rows()
    label = np.log(train["_price"])
    adjusted = label - label.groupby(train["_date"].dt.to_period("M")).transform("median")
    held_out = held_out_mask(train)
    all_groups = groupings(train)
    report_groupings(all_groups, adjusted, held_out)
    report_added_to_sub_area(train, all_groups, adjusted, held_out)
    print("===== approximate position from the grid:")
    report_position(train, label, adjusted)


if __name__ == "__main__":
    main()

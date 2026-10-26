"""Read-only report on the dropped text columns, so each one gets its own measured reason.

Prints numbers only (never the values), because some columns hold names and addresses.
Also measures one-hot subdivision names kept only when a name has enough sales.

Run: python3 text_columns_table.py <king folder>
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "kc_cards"))
from measure import training_rows  # noqa: E402
from location_table import held_out_explained, held_out_mask, text  # noqa: E402

DROPPED_TEXT_COLUMNS = [
    "SellerName", "BuyerName", "Address", "BuildingNumber", "Fraction", "DirectionPrefix", "StreetName",
    "StreetType", "DirectionSuffix", "PropName", "PlatLot (parcel file)", "PlatBlock (parcel file)",
    "PropType", "AFForestLand", "AFCurrentUseLand", "AFNonProfitUse", "AFHistoricProperty", "OtherNuisances",
]
SUBDIVISION_MIN_SALES = [50, 100, 200, 500]


def report_text_column(name, values, sub_area, adjusted, held_out):
    sizes = values.value_counts()
    alone = held_out_explained(values, adjusted, held_out)
    on_top = held_out_explained(sub_area + "|" + values, adjusted, held_out)
    base = held_out_explained(sub_area, adjusted, held_out)
    print(f"{name:24s} values {len(sizes):7d} | empty {(values == 'empty').mean() * 100:5.1f}% | "
          f"most common value {sizes.iloc[0] / len(values) * 100:5.1f}% | median sales per value {sizes.median():5.0f} | "
          f"explains {alone * 100:5.1f}% | on top of area + sub-area {(on_top - base) * 100:+5.1f} points")


def subdivisions_with_enough_sales(names, held_out, min_sales):
    fit_counts = names[~held_out].value_counts()
    kept = fit_counts[fit_counts >= min_sales].index
    return names.where(names.isin(kept), "other"), len(kept)


def report_subdivision_thresholds(names, sub_area, adjusted, held_out):
    base = held_out_explained(sub_area, adjusted, held_out)
    print(f"===== subdivision name, one-hot only above a sales threshold (area + sub-area alone {base * 100:.1f}%)")
    for min_sales in SUBDIVISION_MIN_SALES:
        grouped, kept = subdivisions_with_enough_sales(names, held_out, min_sales)
        covered = (grouped != "other").mean() * 100
        on_top = held_out_explained(sub_area + "|" + grouped, adjusted, held_out)
        print(f"  at least {min_sales:3d} sales: {kept:5d} names + other = {kept + 1:5d} yes/no columns | "
              f"names cover {covered:4.1f}% of sales | alone {held_out_explained(grouped, adjusted, held_out) * 100:4.1f}% | "
              f"on top of area + sub-area {(on_top - base) * 100:+4.1f} points")


def main():
    train, _ = training_rows()
    label = np.log(train["_price"])
    adjusted = label - label.groupby(train["_date"].dt.to_period("M")).transform("median")
    held_out = held_out_mask(train)
    sub_area = text(train, "Area") + "-" + text(train, "SubArea")
    print("===== dropped text columns (held-out homes)")
    for name in DROPPED_TEXT_COLUMNS:
        report_text_column(name, text(train, name), sub_area, adjusted, held_out)
    report_subdivision_thresholds(text(train, "PlatName"), sub_area, adjusted, held_out)


if __name__ == "__main__":
    main()

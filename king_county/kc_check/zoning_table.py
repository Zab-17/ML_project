"""Read-only report: zoning code, alone and on top of the chosen area + sub-area code.

Run: python3 zoning_table.py <king folder>
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "kc_cards"))
from measure import training_rows  # noqa: E402
from location_table import held_out_explained, held_out_mask, report_single, text  # noqa: E402

MIN_SHARE_FOR_OWN_COLUMN = 0.001  # codes under 0.1% of sales would fold into "other"


def report_code_counts(zoning):
    shares = zoning.value_counts(normalize=True)
    kept = shares[shares >= MIN_SHARE_FOR_OWN_COLUMN]
    print(f"  codes with at least 0.1% of sales: {len(kept)} (covering {kept.sum() * 100:.1f}% of sales)")


def report_added_to_sub_area(train, zoning, adjusted, held_out):
    sub_area = text(train, "Area") + "-" + text(train, "SubArea")
    crossed = sub_area + "|" + zoning
    print(f"===== area + sub-area alone {held_out_explained(sub_area, adjusted, held_out) * 100:.1f}% | "
          f"+ zoning {held_out_explained(crossed, adjusted, held_out) * 100:.1f}% ({crossed.nunique()} groups)")


def main():
    train, _ = training_rows()
    label = np.log(train["_price"])
    adjusted = label - label.groupby(train["_date"].dt.to_period("M")).transform("median")
    held_out = held_out_mask(train)
    zoning = text(train, "CurrentZoning")
    report_single("Zoning code", zoning, adjusted, held_out)
    report_code_counts(zoning)
    report_added_to_sub_area(train, zoning, adjusted, held_out)


if __name__ == "__main__":
    main()

"""Read-only check: the four basement parts on training sales, with the basement garage separated."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "kc_cards"))
from measure import training_rows  # noqa: E402

PARTS = ["above_ground", "finished_basement", "basement_garage", "other_unfinished"]


def numbers(frame, column):
    return pd.to_numeric(frame[column], errors="coerce").fillna(0)


def basement_parts(train):
    finished, garage = numbers(train, "SqFtFinBasement"), numbers(train, "SqFtGarageBasement")
    return pd.DataFrame({
        "above_ground": numbers(train, "SqFtTotLiving") - finished,
        "finished_basement": finished,
        "basement_garage": garage,
        "other_unfinished": (numbers(train, "SqFtTotBasement") - finished - garage).clip(lower=0),
    })


def percent_per_100_sqft(parts, adjusted, areas):
    frame = parts / 100
    frame["y"] = adjusted.values
    demeaned = frame - frame.groupby(areas).transform("mean")
    coef, *_ = np.linalg.lstsq(demeaned[PARTS].to_numpy(), demeaned["y"].to_numpy(), rcond=None)
    return {name: round(float(np.expm1(c)) * 100, 2) for name, c in zip(PARTS, coef)}


def main():
    train, _ = training_rows()
    adjusted = np.log(train["_price"]) - np.log(train["_price"]).groupby(
        train["_date"].dt.to_period("M")).transform("median")
    parts = basement_parts(train)
    for name in PARTS:
        present = parts[name] > 0
        print(f"{name}: present in {present.mean() * 100:.1f}% of sales, median when present "
              f"{parts[name][present].median():.0f} sq ft")
    print("above ground is zero or negative:", int((parts["above_ground"] <= 0).sum()))
    print("% per 100 sq ft, same appraisal area and month:", percent_per_100_sqft(parts, adjusted, train["Area"].values))


if __name__ == "__main__":
    main()

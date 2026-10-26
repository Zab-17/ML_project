"""Read-only check: how total basement, finished basement and living area overlap."""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent / "kc_cards"))
from measure import training_rows  # noqa: E402


def numbers(frame, column):
    return pd.to_numeric(frame[column], errors="coerce").fillna(0)


def basement_kinds(total, finished):
    kinds = np.select(
        [total == 0, finished == 0, finished >= total],
        ["no basement", "basement, nothing finished", "basement, fully finished"],
        "basement, partly finished",
    )
    return pd.Series(kinds).value_counts(normalize=True).mul(100).round(1)


def within_area(frame, columns, groups):
    return frame[columns] - frame[columns].groupby(groups).transform("mean")


def price_per_100_sqft(train, adjusted):
    parts = pd.DataFrame({
        "above_ground": numbers(train, "SqFtTotLiving") - numbers(train, "SqFtFinBasement"),
        "finished_basement": numbers(train, "SqFtFinBasement"),
        "unfinished_basement": (numbers(train, "SqFtTotBasement") - numbers(train, "SqFtFinBasement")).clip(lower=0),
    }) / 100
    parts["y"] = adjusted.values
    demeaned = within_area(parts, list(parts.columns), train["Area"].values)
    x = demeaned[["above_ground", "finished_basement", "unfinished_basement"]].to_numpy()
    coef, *_ = np.linalg.lstsq(x, demeaned["y"].to_numpy(), rcond=None)
    return {name: round(float(np.expm1(c)) * 100, 2) for name, c in zip(
        ["above_ground", "finished_basement", "unfinished_basement"], coef)}


def main():
    train, _ = training_rows()
    adjusted = np.log(train["_price"]) - np.log(train["_price"]).groupby(
        train["_date"].dt.to_period("M")).transform("median")
    total, finished = numbers(train, "SqFtTotBasement"), numbers(train, "SqFtFinBasement")
    print("rows", len(train))
    print("finished bigger than total:", int((finished > total).sum()))
    print(basement_kinds(total, finished).to_string())
    has = total > 0
    print("median unfinished part when a basement exists:", float((total - finished)[has].median()))
    print("finished basement as % of living area, median when finished>0:",
          round(float((finished / numbers(train, 'SqFtTotLiving'))[finished > 0].median() * 100), 1))
    print("% price change per 100 sq ft, same appraisal area, same month:", price_per_100_sqft(train, adjusted))


if __name__ == "__main__":
    main()

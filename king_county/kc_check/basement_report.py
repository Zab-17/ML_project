"""Read-only report: empties, zeros, sizes and price gaps for the basement and garage pieces."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "kc_cards"))
from measure import training_rows  # noqa: E402

RAW = ["SqFtTotBasement", "SqFtFinBasement", "SqFtGarageBasement", "SqFtGarageAttached"]


def numbers(frame, column):
    return pd.to_numeric(frame[column].replace("", np.nan), errors="coerce")


def describe(name, values, adjusted):
    empty, zero = values.isna(), values == 0
    present = ~empty & ~zero
    gap = np.expm1(adjusted[present].median() - adjusted[~present].median()) * 100
    p = values[present]
    print(f"{name:26s} empty {empty.sum():>7,} ({empty.mean()*100:4.1f}%) | zero {zero.sum():>7,} ({zero.mean()*100:4.1f}%) | "
          f"present {present.sum():>7,} ({present.mean()*100:4.1f}%) | when present: median {p.median():,.0f}, "
          f"middle half {p.quantile(.25):,.0f}-{p.quantile(.75):,.0f}, top 1% over {p.quantile(.99):,.0f}, max {p.max():,.0f} | "
          f"price with vs without {gap:+.1f}%")


def main():
    train, _ = training_rows()
    adjusted = np.log(train["_price"]) - np.log(train["_price"]).groupby(
        train["_date"].dt.to_period("M")).transform("median")
    raw = {c: numbers(train, c) for c in RAW}
    total, finished, b_garage, a_garage = (raw[c].fillna(0) for c in RAW)
    other = total - finished - b_garage
    print("rows", len(train))
    for c in RAW:
        describe(c, raw[c], adjusted)
    describe("other unfinished basement", other, adjusted)
    describe("total garage", a_garage + b_garage, adjusted)
    print("checks: finished > total", int((finished > total).sum()), "| garage > total - finished", int((b_garage > total - finished).sum()),
          "| other unfinished negative", int((other < 0).sum()))
    print("houses with both garage kinds", int(((a_garage > 0) & (b_garage > 0)).sum()),
          "| no garage of either kind", int(((a_garage == 0) & (b_garage == 0)).sum()))


if __name__ == "__main__":
    main()

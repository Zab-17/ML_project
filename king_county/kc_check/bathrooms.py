"""Read-only check: the three bathroom columns, their zeros, and what each kind adds to price."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "kc_cards"))
from measure import training_rows  # noqa: E402

BATHS = {"full": "BathFullCount", "three_quarter": "Bath3qtrCount", "half": "BathHalfCount"}
WEIGHTS = {"full": 1.0, "three_quarter": 0.75, "half": 0.5}


def numbers(frame, column):
    return pd.to_numeric(frame[column], errors="coerce").fillna(0)


def month_adjusted_log_price(train):
    log_price = np.log(train["_price"])
    return log_price - log_price.groupby(train["_date"].dt.to_period("M")).transform("median")


def percent_per_extra_bathroom(baths, train, adjusted):
    """Price change per extra bathroom of each kind, same appraisal area, same month, same living-area size."""
    frame = baths.copy()
    frame["log_living_area"] = np.log(numbers(train, "SqFtTotLiving").clip(lower=1))
    inputs = list(frame.columns)
    frame["y"] = adjusted.values
    demeaned = frame - frame.groupby(train["Area"].values).transform("mean")
    coef, *_ = np.linalg.lstsq(demeaned[inputs].to_numpy(), demeaned["y"].to_numpy(), rcond=None)
    return {name: round(float(np.expm1(c)) * 100, 1) for name, c in zip(inputs, coef) if name in BATHS}


def report_zeros(baths):
    no_full = baths["full"] == 0
    no_bath = baths.sum(axis=1) == 0
    print(f"0 full bathrooms: {no_full.sum():,} | of those, with a three-quarter bathroom: "
          f"{(no_full & (baths['three_quarter'] > 0)).sum():,} | no bathroom of any kind: {no_bath.sum():,}")


def report_total(baths, adjusted):
    total = sum(baths[name] * weight for name, weight in WEIGHTS.items())
    print(f"total bathrooms: median {total.median()}, top 1% over {total.quantile(.99)}, max {total.max()} | "
          f"rank correlation with month-adjusted price {total.corr(adjusted, method='spearman'):.3f}")


def main():
    train, _ = training_rows()
    adjusted = month_adjusted_log_price(train)
    baths = pd.DataFrame({name: numbers(train, column) for name, column in BATHS.items()})
    print("rows", len(train))
    report_zeros(baths)
    report_total(baths, adjusted)
    print("% per extra bathroom, same area, month and living area:", percent_per_extra_bathroom(baths, train, adjusted))


if __name__ == "__main__":
    main()

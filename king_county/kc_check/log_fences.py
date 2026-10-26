"""Read-only check: unusual sales at each end once prices are on a log scale."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "kc_cards"))
from measure import training_rows  # noqa: E402


def fences(values):
    q1, q3 = np.percentile(values, [25, 75])
    spread = q3 - q1
    return q1 - 1.5 * spread, q3 + 1.5 * spread


def main():
    train, _ = training_rows()
    price = train["_price"].to_numpy(dtype=float)
    for name, values, back in [("raw dollars", price, lambda v: v), ("log", np.log(price), np.exp)]:
        low, high = fences(values)
        print(f"{name}: below lower fence {(values < low).sum():,} (under ${max(back(low), 0):,.0f}) | "
              f"above upper fence {(values > high).sum():,} (over ${back(high):,.0f})")


if __name__ == "__main__":
    main()

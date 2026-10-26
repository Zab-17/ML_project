"""Read-only check: how much of the squared error the most expensive sales carry, raw dollars versus log."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "kc_cards"))
from measure import training_rows  # noqa: E402


def share_of_squared_error(values, mask):
    squared = (values - values.mean()) ** 2
    return round(float(squared[mask].sum() / squared.sum() * 100), 1)


def main():
    train, _ = training_rows()
    price = train["_price"].to_numpy(dtype=float)
    q1, q3 = np.percentile(price, [25, 75])
    groups = {
        "above upper fence": price > q3 + 1.5 * (q3 - q1),
        "top 1%": price > np.percentile(price, 99),
        "middle half": (price >= q1) & (price <= q3),
    }
    for name, mask in groups.items():
        print(f"{name}: {mask.mean() * 100:.1f}% of sales | share of squared error raw "
              f"{share_of_squared_error(price, mask)}% | log {share_of_squared_error(np.log(price), mask)}%")


if __name__ == "__main__":
    main()

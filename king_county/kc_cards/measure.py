"""Measures every King County column for the feature-card page.

Rows: clean residential sales from 2000 on, joined to parcels with exactly one
house, where the house was not built or renovated after the sale. Train/test
split is by home (parcel), 80/20, seed 42; every number comes from the
training homes only. The label for relationships is the log sale price minus
the median log sale price of the same calendar month (time-adjusted).
"""
import json
import sys

import numpy as np
import pandas as pd

KING = sys.argv[1] if len(sys.argv) > 1 else ""
OUT = sys.argv[2] if len(sys.argv) > 2 else ""

FIRST_YEAR = 2000
MIN_PRICE = 10_000
TEST_SHARE = 0.2
SEED = 42
MIN_GROUP_ROWS = 30
NUMERIC_BINS = 10
HIST_BINS = 40
TOP_CATEGORIES = 12
SAMPLE_ROWS = 200_000
NESTING_PURITY = 0.99

NO_EXAMPLES = {"ExciseTaxNbr", "RecordingNbr", "Major", "Minor", "SellerName", "BuyerName",
               "Address", "BuildingNumber"}
NO_BARS = NO_EXAMPLES
LOOKUP_TYPE = {
    "PropertyType": "1", "PrincipalUse": "2", "PropertyClass": "4", "SaleReason": "5",
    "SaleInstrument": "6", "SaleWarning": "7", "CurrentUseDesignation": "16", "WfntLocation": "50",
    "WfntBank": "52", "TidelandShoreland": "54", "Access": "55", "WaterSystem": "56",
    "SewerSystem": "57", "StreetSurface": "60", "HistoricSite": "67", "BldgGrade": "82",
    "FinBasementGrade": "82", "Condition": "83", "HeatSource": "84", "InadequateParking": "92",
    "TrafficNoise": "95", "PresentUse": "102", "HBUAsIfVacant": "103", "HBUAsImproved": "104",
    "HeatSystem": "108",
}
VIEW_COLUMNS = ["MtRainier", "Olympics", "Cascades", "Territorial", "SeattleSkyline", "PugetSound",
                "LakeWashington", "LakeSammamish", "SmallLakeRiverCreek", "OtherView"]
for view in VIEW_COLUMNS:
    LOOKUP_TYPE[view] = "58"


def read(name):
    frame = pd.read_csv(f"{KING}/{name}.csv", dtype=str, encoding="latin-1", keep_default_na=False)
    frame = frame.apply(lambda column: column.str.strip())
    frame["parcel"] = frame["Major"].str.zfill(6) + "-" + frame["Minor"].str.zfill(4)
    return frame


def split_rows():
    """Training sales, test sales and a summary; the split is by home, so one home never sits on both sides."""
    sales = read("EXTR_RPSale")
    price = pd.to_numeric(sales["SalePrice"], errors="coerce")
    date = pd.to_datetime(sales["DocumentDate"], format="%m/%d/%Y", errors="coerce")
    one_parcel = sales.groupby("ExciseTaxNbr")["parcel"].transform("nunique") == 1
    clean = sales[(sales["SaleReason"] == "1") & sales["SaleInstrument"].isin(["2", "3"])
                  & (sales["SaleWarning"] == "") & one_parcel & (sales["PrincipalUse"] == "6")
                  & (sales["PropertyClass"] == "8") & (price >= MIN_PRICE)
                  & (date.dt.year >= FIRST_YEAR)].copy()
    clean["_price"] = price[clean.index]
    clean["_date"] = date[clean.index]
    buildings = read("EXTR_ResBldg")
    buildings = buildings[buildings.groupby("parcel")["parcel"].transform("size") == 1]
    parcels = read("EXTR_Parcel")
    joined = (clean.merge(buildings.drop(columns=["Major", "Minor"]), on="parcel")
                   .merge(parcels.drop(columns=["Major", "Minor"]), on="parcel",
                          suffixes=("", " (parcel file)")))
    year = joined["_date"].dt.year
    built = pd.to_numeric(joined["YrBuilt"], errors="coerce")
    renovated = pd.to_numeric(joined["YrRenovated"], errors="coerce").fillna(0)
    usable = joined[(built <= year) & (renovated <= year)].reset_index(drop=True)
    homes = usable["parcel"].unique()
    test_homes = set(np.random.default_rng(SEED).choice(homes, int(len(homes) * TEST_SHARE), replace=False))
    is_test = usable["parcel"].isin(test_homes)
    summary = {"clean_sales_2000_on": int(len(clean)), "usable_sales": int(len(usable)),
               "homes": int(len(homes)), "train_sales": int((~is_test).sum()),
               "test_sales": int(is_test.sum()), "train_homes": int(len(homes) - len(test_homes)),
               "first_date": str(usable["_date"].min().date()), "last_date": str(usable["_date"].max().date())}
    return usable[~is_test].reset_index(drop=True), usable[is_test].reset_index(drop=True), summary


def training_rows():
    train, _, summary = split_rows()
    return train, summary


def lookup_names():
    table = pd.read_csv(f"{KING}/EXTR_LookUp.csv", dtype=str, encoding="latin-1", keep_default_na=False)
    table = table.apply(lambda column: column.str.strip())
    names = {}
    for kind, group in table.groupby("LUType"):
        names[kind] = dict(zip(group["LUItem"], group["LUDescription"]))
    return {column: names.get(kind, {}) for column, kind in LOOKUP_TYPE.items()}


def as_percent_vs_month(adjusted_log):
    return round(float(np.expm1(adjusted_log)) * 100, 1)


def explained_share(groups, adjusted):
    is_fit = np.random.default_rng(SEED).random(len(groups)) < 0.5
    stats = adjusted[is_fit].groupby(groups[is_fit]).agg(["mean", "size"])
    means = stats.loc[stats["size"] >= MIN_GROUP_ROWS, "mean"]
    predicted = groups[~is_fit].map(means).astype(float).fillna(adjusted[is_fit].mean())
    actual = adjusted[~is_fit]
    return round(float(1 - ((actual - predicted) ** 2).sum() / ((actual - actual.mean()) ** 2).sum()), 4)


def numeric_values(values):
    number = pd.to_numeric(values.where(values != ""), errors="coerce")
    filled = (values != "").sum()
    return number if filled and number.notna().sum() / filled >= 0.99 else None


def histogram(present, transform):
    data = transform(present)
    low, high = np.percentile(data, [1, 99])
    if high <= low:
        return None
    counts, _ = np.histogram(data[(data >= low) & (data <= high)], bins=HIST_BINS, range=(low, high))
    return {"lo": round(float(low), 4), "hi": round(float(high), 4), "counts": counts.tolist(),
            "skew": round(float(pd.Series(data).skew()), 2)}


def numeric_block(number, adjusted):
    present = number.dropna()
    if present.empty:
        return None
    q = present.quantile([0.01, 0.25, 0.5, 0.75, 0.99])
    fence = 1.5 * (q[0.75] - q[0.25])
    block = {
        "count": int(len(present)), "min": float(present.min()), "max": float(present.max()),
        "mean": round(float(present.mean()), 2), "p1": float(q[0.01]), "p25": float(q[0.25]),
        "p50": float(q[0.5]), "p75": float(q[0.75]), "p99": float(q[0.99]),
        "skew": round(float(present.skew()), 2) if present.nunique() > 1 else 0.0,
        "log_skew": (round(float(np.log1p(present).skew()), 2)
                     if present.min() >= 0 and present.nunique() > 1 else None),
        "zeros": int((present == 0).sum()), "negatives": int((present < 0).sum()),
        "below_fence": int((present < q[0.25] - fence).sum()),
        "above_fence": int((present > q[0.75] + fence).sum()),
        "spearman": (round(float(present.rank().corr(adjusted[present.index].rank())), 3)
                     if present.nunique() > 1 else None),
        "hist": histogram(present.to_numpy(), lambda x: x) if present.nunique() > 20 else None,
        "log_hist": (histogram(present.to_numpy(), np.log1p)
                     if present.nunique() > 20 and present.min() >= 0 else None),
    }
    return block


def numeric_groups(number):
    present = number.dropna()
    if present.nunique() <= 20:
        return number.map(lambda v: "" if pd.isna(v) else f"{v:g}"), "values"
    edges = np.unique(present.quantile(np.linspace(0, 1, NUMERIC_BINS + 1)))
    bins = pd.cut(number, edges, include_lowest=True)
    return bins.astype(str).where(number.notna(), ""), "deciles"


def relation_points(groups, number, adjusted, kind):
    frame = pd.DataFrame({"g": groups, "adj": adjusted})
    if number is not None:
        frame["x"] = number
    frame = frame[frame["g"] != ""]
    agg = frame.groupby("g").agg(adj=("adj", "median"), rows=("adj", "size"),
                                 **({"x": ("x", "median")} if number is not None else {}))
    agg = agg[agg["rows"] >= MIN_GROUP_ROWS]
    if number is not None:
        agg = agg.sort_values("x")
        return [[float(r.x), as_percent_vs_month(r.adj), int(r.rows)] for r in agg.itertuples()]
    return None


def category_bars(values, adjusted, names):
    counts = values[values != ""].value_counts()
    top = counts.index[:TOP_CATEGORIES]
    medians = adjusted.groupby(values).median()
    total = len(values)
    bars = [[code, names.get(code, ""), int(counts[code]), round(counts[code] / total * 100, 2),
             as_percent_vs_month(medians[code]) if counts[code] >= MIN_GROUP_ROWS else None]
            for code in top]
    return {"items": bars, "rest_rows": int(counts.iloc[TOP_CATEGORIES:].sum()),
            "rest_distinct": int(max(len(counts) - TOP_CATEGORIES, 0)),
            "rare_share": round(float(counts[counts < MIN_GROUP_ROWS].sum() / total * 100), 2)}


def nesting_purity(inner, outer):
    pairs = pd.DataFrame({"i": inner, "o": outer})
    pairs = pairs[(pairs["i"] != "") & (pairs["o"] != "")]
    pairs = pairs[pairs.groupby("i")["o"].transform("size") >= MIN_GROUP_ROWS]
    if pairs["i"].nunique() < 2 or pairs["o"].nunique() < 2:
        return 0.0
    largest = pairs.groupby(["i", "o"]).size().groupby(level="i").max()
    return float(largest.sum() / len(pairs))


def containment(sample, columns):
    found = {column: [] for column in columns}
    for inner in columns:
        for outer in columns:
            if inner != outer and nesting_purity(sample[inner], sample[outer]) >= NESTING_PURITY:
                both_ways = nesting_purity(sample[outer], sample[inner]) >= NESTING_PURITY
                found[inner].append([outer, both_ways])
    return found


def main():
    train, summary = training_rows()
    adjusted = np.log(train["_price"]) - np.log(train["_price"]).groupby(train["_date"].dt.to_period("M")).transform("median")
    names = lookup_names()
    columns = [c for c in train.columns if c not in ("parcel", "_price", "_date")]
    sample = train.sample(n=min(SAMPLE_ROWS, len(train)), random_state=SEED)
    sample_adjusted = adjusted[sample.index]

    numbers = {c: numeric_values(train[c]) for c in columns}
    numeric_columns = [c for c in columns if numbers[c] is not None and numbers[c].nunique() > 2]
    rank = pd.DataFrame({c: numbers[c][sample.index] for c in numeric_columns}).corr(method="spearman")
    category_columns = [c for c in columns if c not in NO_BARS and 2 <= train[c].replace("", np.nan).nunique() <= 20_000]
    nests = containment(sample, category_columns)

    cards = []
    for column in columns:
        values = train[column]
        number = numbers[column]
        empty = values == ""
        zero = values.isin(["0", "0.0", "0.00"])
        card = {
            "name": column, "rows": int(len(values)),
            "empty_pct": round(float(empty.mean() * 100), 2), "zero_pct": round(float(zero.mean() * 100), 2),
            "distinct": int(values[~empty].nunique()),
            "examples": [] if column in NO_EXAMPLES else values[~empty & ~zero].drop_duplicates().head(6).tolist(),
            "decode": names.get(column) or None,
            "nested_in": nests.get(column, []),
        }
        absent = empty | zero
        if absent.any() and (~absent).any():
            card["median_when_absent"] = as_percent_vs_month(adjusted[absent].median())
            card["median_when_present"] = as_percent_vs_month(adjusted[~absent].median())
        if number is not None:
            card["numeric"] = numeric_block(number, adjusted)
            groups, kind = numeric_groups(number)
            card["relation"] = {"kind": kind, "points": relation_points(groups, number, adjusted, kind),
                                "explained": explained_share(groups, adjusted)}
            if column in rank.columns:
                others = rank[column].drop(column).dropna()
                card["similar"] = [[name, round(float(others[name]), 3)]
                                   for name in others.abs().sort_values(ascending=False).index[:3]]
        if column not in NO_BARS and (number is None or values[~empty].nunique() <= 20):
            card["bars"] = category_bars(values, adjusted, names.get(column, {}))
            if number is None:
                card["relation"] = {"kind": "categories", "points": None,
                                    "explained": explained_share(values, adjusted)}
        elif number is None:
            card["relation"] = {"kind": "identifier", "points": None,
                                "explained": explained_share(values, adjusted)}
        cards.append(card)

    living = pd.to_numeric(train["SqFtTotLiving"], errors="coerce")
    parts = sum(pd.to_numeric(train[c], errors="coerce").fillna(0)
                for c in ["SqFt1stFloor", "SqFtHalfFloor", "SqFt2ndFloor", "SqFtUpperFloor", "SqFtFinBasement"])
    summary["living_area_identity_pct"] = round(float((living == parts).mean() * 100), 2)
    summary["label"] = numeric_block(train["_price"], adjusted)
    summary["sales_by_year"] = train["_date"].dt.year.value_counts().sort_index().astype(int).to_dict()
    summary["median_price_by_year"] = train.groupby(train["_date"].dt.year)["_price"].median().astype(int).to_dict()
    with open(OUT, "w") as handle:
        json.dump({"summary": summary, "cards": cards}, handle, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    for card in cards:
        rel = card.get("relation", {})
        num = card.get("numeric") or {}
        print(f"{card['name']:26s} empty {card['empty_pct']:6.2f} zero {card['zero_pct']:6.2f} "
              f"distinct {card['distinct']:>7} explained {rel.get('explained')!s:>7} spearman {num.get('spearman')!s:>6} "
              f"skew {num.get('skew')!s:>7} logskew {num.get('log_skew')!s:>6} nested {[n[0] for n in card['nested_in']][:3]}")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("label", "sales_by_year", "median_price_by_year")}))


if __name__ == "__main__":
    main()

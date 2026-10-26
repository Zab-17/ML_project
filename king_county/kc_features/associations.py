"""How strongly every original column and every chosen feature goes with the label and with each other.

The measure depends on the kinds of the two columns:
- number with number: Pearson correlation, signed, from -1 to 1;
- category with number: correlation ratio, from 0 to 1. The category's group averages of the number are
  learned on 4 of every 5 training homes and scored on the 5th; the square root of the share explained;
- category with category: Cramer's V with the small-sample bias correction, from 0 to 1.
Learning on some homes and scoring on others keeps a column with one value per sale (a name, an address)
near 0 against the label, instead of a meaningless near-perfect score.

Run: python3 associations.py <king folder>
Writes the tables and heatmaps into out/.
"""
# AI was used to help write this code, but the resulting code was reviewed and edited by a human.

import json
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from features import (START_OF_2000, DAYS_PER_YEAR, build_feature_table, training_rows,  # noqa: E402
                      without_sales_missing_rooms)
from location_table import held_out_mask, text  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "out"
CARDS = HERE.parent / "kc_check" / "payload.json"
SUBDIVISION_MIN_SALES, STREET_MIN_SALES = 100, 200  # D4 and D5
LABEL = "LABEL: log sale price"
NUMBER_KINDS = {"area (square feet)", "count", "ordered rating (0 to 4)", "percent", "number",
                "label: money (US dollars)", "date", "year", "ordered grade (1 to 13)", "ordered grade (1 to 5)",
                "ordered grade (same scale as BldgGrade)", "money (US dollars)", "length (feet)",
                "ordered rating (0 to 3)", "flag (0/1)"}
RAW_COLUMNS_USED = [
    "DocumentDate", "SqFtTotLiving", "SqFtFinBasement", "SqFtTotBasement", "SqFtGarageBasement", "SqFtGarageAttached",
    "SqFtOpenPorch", "SqFtDeck", "Bedrooms", "BathFullCount", "Bath3qtrCount", "BathHalfCount", "Stories", "YrBuilt",
    "FpSingleStory", "FpMultiStory", "FpFreestanding", "FpAdditional", "Range", "Township", "Section", "QuarterSection",
    "WfntFootage", "SqFtLot", "TrafficNoise", "MtRainier", "Olympics", "Cascades", "Territorial", "SeattleSkyline",
    "PugetSound", "LakeWashington", "LakeSammamish", "SmallLakeRiverCreek", "OtherView", "TidelandShoreland",
    "Topography", "AirportNoise", "PowerLines", "AdjacentGolfFairway", "Easements", "DeedRestrictions",
    "NativeGrowthProtEsmt", "OtherDesignation", "DNRLease", "WaterProblems", "AdjacentGreenbelt", "WaterSystem",
    "SewerSystem", "YrRenovated", "BldgGrade", "FinBasementGrade", "HeatSystem", "HeatSource", "DistrictName", "Area",
    "SubArea", "PlatName", "StreetName", "StreetType", "WfntLocation", "WfntBank", "PresentUse", "Access",
    "StreetSurface", "InadequateParking"]
REVIEW_LABEL_LEVEL = 0.15     # a dropped column this close to the label is worth a second look ...
COVERED_LEVEL = 0.7           # ... unless a kept feature already goes with it this strongly


@dataclass
class Item:
    name: str
    kind: str          # "number" or "category"
    values: np.ndarray  # floats for numbers (NaN allowed), integer codes for categories


def main():
    train = without_sales_missing_rooms(training_rows()[0])
    table = build_feature_table(train, SUBDIVISION_MIN_SALES, STREET_MIN_SALES)
    cards = {card["name"]: card for card in json.loads(CARDS.read_text())["cards"]}
    label = Item(LABEL, "number", table.label.to_numpy())
    raw, features = raw_items(train, cards), feature_items(table)
    matrix = association_matrix([label] + raw + features, held_out_mask(train))
    OUT_DIR.mkdir(exist_ok=True)
    raw_names, feature_names = [LABEL] + [i.name for i in raw], [LABEL] + [i.name for i in features]
    save_table(matrix, raw_names, "original_151_columns", raw_summary(matrix, raw, cards))
    save_table(matrix, feature_names, "chosen_60_features", feature_summary(matrix, features))
    dropped = dropped_check(matrix, raw, features)
    dropped.to_csv(OUT_DIR / "dropped_columns_check.csv", index=False)
    print_findings(matrix, raw, features, dropped)


def raw_items(train, cards):
    items = []
    for column, card in cards.items():
        if column == "DocumentDate":
            values = ((train["_date"] - START_OF_2000).dt.days / DAYS_PER_YEAR).to_numpy()
            items.append(Item(column, "number", values))
        elif card["kind"] in NUMBER_KINDS:
            items.append(Item(column, "number", pd.to_numeric(train[column], errors="coerce").to_numpy(float)))
        else:
            items.append(Item(column, "category", pd.factorize(text(train, column))[0]))
    return items


def feature_items(table):
    numbers = [Item(name, "number", table.numbers[name].to_numpy(float)) for name in table.numbers]
    yes_no = [Item(name, "number", table.yes_no[name].to_numpy(float)) for name in table.yes_no]
    categories = [Item(name, "category", pd.factorize(labels)[0]) for name, labels in table.categories.items()]
    return numbers + yes_no + categories


def association_matrix(items, held_out):
    names = [item.name for item in items]
    matrix = pd.DataFrame(np.nan, index=names, columns=names)
    numbers = [i for i in items if i.kind == "number"]
    categories = [i for i in items if i.kind == "category"]
    frame = pd.DataFrame({item.name: item.values for item in numbers})
    fill_number_pairs(matrix, frame)
    for category in categories:
        fill_category_with_numbers(matrix, category, frame, held_out)
    for first, second in combinations(categories, 2):
        matrix.loc[first.name, second.name] = matrix.loc[second.name, first.name] = cramers_v(first.values, second.values)
    np.fill_diagonal(matrix.values, 1.0)
    return matrix


def fill_number_pairs(matrix, frame):
    pearson = frame.corr()
    matrix.loc[pearson.index, pearson.columns] = pearson


def fill_category_with_numbers(matrix, category, frame, held_out):
    ratios = correlation_ratios(category.values, frame, held_out)
    matrix.loc[category.name, ratios.index] = ratios
    matrix.loc[ratios.index, category.name] = ratios


def correlation_ratios(codes, frame, held_out):
    """Square root of the share of each number's spread that the category's group averages explain on held-out homes."""
    if len(np.unique(codes)) < 2:
        return pd.Series(np.nan, index=frame.columns)
    fit, scored = frame[~held_out], frame[held_out]
    group_means = fit.groupby(codes[~held_out]).mean()
    predicted = group_means.reindex(codes[held_out]).set_axis(scored.index).fillna(fit.mean())
    error = ((scored - predicted) ** 2).sum()
    spread = ((scored - scored.mean()) ** 2).sum()
    explained = (1 - error / spread).where(spread > 0)
    return np.sqrt(explained.clip(lower=0))


def cramers_v(first, second):
    """Cramer's V with the bias correction of Bergsma (2013); NaN when a column never changes."""
    sales = len(first)
    rows, cols = first, second  # factorized codes: every value from 0 to its count minus 1 is present
    r, k = rows.max() + 1, cols.max() + 1
    if r < 2 or k < 2:
        return np.nan
    cells, counts = np.unique(rows.astype(np.int64) * k + cols, return_counts=True)
    row_totals, col_totals = np.bincount(rows), np.bincount(cols)
    chi_square = sales * ((counts ** 2 / (row_totals[cells // k] * col_totals[cells % k].astype(float))).sum() - 1)
    phi_square = max(0.0, chi_square / sales - (k - 1) * (r - 1) / (sales - 1))
    r_fixed, k_fixed = r - (r - 1) ** 2 / (sales - 1), k - (k - 1) ** 2 / (sales - 1)
    denominator = min(r_fixed - 1, k_fixed - 1)
    return np.sqrt(phi_square / denominator) if denominator > 0 else np.nan


def raw_summary(matrix, raw, cards):
    rows = []
    for item in raw:
        card = cards[item.name]
        rows.append({"column": item.name, "plain name": card.get("label", ""), "file": card["file"],
                     "kind": card["kind"], "measure kind": item.kind, "decision": raw_decision(item.name),
                     "with label": matrix.loc[item.name, LABEL],
                     **strongest_partner(matrix, item.name, [i.name for i in raw])})
    return pd.DataFrame(rows)


def raw_decision(column):
    if column == "SalePrice":
        return "label"
    return "used by a chosen feature" if column in RAW_COLUMNS_USED else "dropped"


def feature_summary(matrix, features):
    names = [item.name for item in features]
    return pd.DataFrame([{"feature": item.name, "measure kind": item.kind, "with label": matrix.loc[item.name, LABEL],
                          **strongest_partner(matrix, item.name, names)} for item in features])


def strongest_partner(matrix, name, pool):
    others = matrix.loc[name, [p for p in pool if p != name]].dropna()
    if others.empty:
        return {"strongest partner": "", "strongest partner value": np.nan}
    best = others.abs().idxmax()
    return {"strongest partner": best, "strongest partner value": others[best]}


def dropped_check(matrix, raw, features):
    feature_names = [item.name for item in features]
    rows = []
    for item in raw:
        if raw_decision(item.name) != "dropped":
            continue
        partner = strongest_partner(matrix, item.name, feature_names)
        rows.append({"dropped column": item.name, "with label": matrix.loc[item.name, LABEL],
                     "closest chosen feature": partner["strongest partner"],
                     "closest chosen feature value": partner["strongest partner value"]})
    check = pd.DataFrame(rows)
    check["second look"] = ((check["with label"].abs() >= REVIEW_LABEL_LEVEL)
                            & (check["closest chosen feature value"].abs() < COVERED_LEVEL))
    return check.sort_values("with label", key=np.abs, ascending=False)


def save_table(matrix, names, stem, summary):
    matrix.loc[names, names].round(3).to_csv(OUT_DIR / f"{stem}_matrix.csv")
    summary.sort_values("with label", key=np.abs, ascending=False).to_csv(OUT_DIR / f"{stem}_with_label.csv", index=False)
    draw_heatmap(matrix.loc[names, names], OUT_DIR / f"{stem}_heatmap.png", annotate=len(names) <= 70)


def draw_heatmap(block, path, annotate):
    size = max(10, len(block) * 0.28)
    figure, axes = plt.subplots(figsize=(size, size))
    image = axes.imshow(block.abs().to_numpy(), cmap="Blues", vmin=0, vmax=1)
    axes.set_xticks(range(len(block)), block.columns, rotation=90, fontsize=6)
    axes.set_yticks(range(len(block)), block.index, fontsize=6)
    if annotate:
        for (row, col), value in np.ndenumerate(block.to_numpy()):
            if not np.isnan(value):
                axes.text(col, row, f"{value:.2f}".replace("0.", "."), ha="center", va="center", fontsize=3.5,
                          color="white" if abs(value) > 0.6 else "#333333")
    figure.colorbar(image, ax=axes, fraction=0.03, label="strength (absolute value)")
    axes.set_title(f"{path.stem.replace('_', ' ')}: Pearson for two numbers, correlation ratio for a category "
                   "and a number, Cramer's V for two categories; blank = column never changes", fontsize=9)
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def print_findings(matrix, raw, features, dropped):
    raw_names = [i.name for i in raw]
    constant = [n for n in raw_names if matrix.loc[n, LABEL] != matrix.loc[n, LABEL]]
    print(f"original columns: {len(raw_names)} | never change in training: {len(constant)}: {', '.join(constant)}")
    over = [n for n in raw_names + [i.name for i in features] if abs(matrix.loc[n, LABEL]) >= 0.85 and n != "SalePrice"]
    print(f"columns at 0.85 or more with the label (sale price itself left out): {over or 'none'}")
    print("===== dropped columns, strongest with the label first")
    print(dropped.head(30).round(3).to_string(index=False))
    print("===== chosen features: pairs at 0.7 or more")
    names = [i.name for i in features]
    for first, second in combinations(names, 2):
        value = matrix.loc[first, second]
        if abs(value) >= 0.7:
            print(f"  {value:+.3f}  {first} | {second}")


if __name__ == "__main__":
    main()

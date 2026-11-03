"""The King County column and correlation checks, run on the Zillow training homes.

1. Column check: every raw column's blank share, number of values, strength with the label and strongest
   partner, next to its decision (out/checks/column_check.csv).
2. Association tables and heatmaps for the 58 original columns and for the chosen features. The measure fits
   the pair: Pearson for two numbers, correlation ratio on held-out homes for a category and a number, Cramer's V
   with the bias correction for two categories (the King County functions, unchanged).
3. Zeyad's feature rules, in the King County order:
   - a feature at 0.85 or more with the label leaks it and goes (professor's rule 4);
   - of two features at 0.85 or more with each other, the one weaker with the label goes (strongest pair first);
   - a dropped column comes back if it is between 0.60 and 0.80 with the label and under 0.85 with every chosen
     feature; columns that leak the label never come back.
"""
# AI was used to help write this code, but the resulting code was reviewed and edited by a human.

import numpy as np
import pandas as pd

from shared_code import CHECKS_DIR
from associations import Item, association_matrix, draw_heatmap, strongest_partner
from zillow_columns import CATEGORY, COLUMN_BY_NAME, DROP, LABEL as LABEL_DECISION, LEAK, RAW_COLUMNS
from zillow_features import code_text
from zillow_rows import LABEL

LABEL_ITEM = "LABEL: log assessed value"
LIMIT = 0.85                          # professor's rule 4, and Zeyad's pair rule
COMEBACK_WINDOW = (0.60, 0.80)        # Zeyad's rule 3
REVIEW_LABEL_LEVEL, COVERED_LEVEL = 0.15, 0.7   # King County's second-look test for dropped columns


def run_checks(train, folds, table):
    """Writes every check table and heatmap; returns the features the rules drop, with the reason for each."""
    CHECKS_DIR.mkdir(parents=True, exist_ok=True)
    label = Item(LABEL_ITEM, "number", table.label.to_numpy())
    raw, features = raw_items(train), feature_items(table)
    matrix = association_matrix([label] + raw + features, held_out=folds == 0)
    raw_names, feature_names = [i.name for i in raw], [i.name for i in features]
    save_block(matrix, raw_names, "original_58_columns")
    dropped = dropped_features(matrix, feature_names)
    chosen = [name for name in feature_names if name not in dropped]
    save_block(matrix, feature_names, f"chosen_{len(feature_names)}_features")
    column_check(train, matrix, raw_names, chosen).to_csv(CHECKS_DIR / "column_check.csv", index=False)
    feature_check(matrix, feature_names, dropped).to_csv(CHECKS_DIR / "feature_check.csv", index=False)
    return dropped


def raw_items(train):
    items = []
    for column in RAW_COLUMNS:
        if column.decision == LABEL_DECISION:
            continue
        values = train[column.name]
        if column.measured_as == CATEGORY:
            items.append(Item(column.name, "category", pd.factorize(code_text(values))[0]))
        else:
            items.append(Item(column.name, "number", pd.to_numeric(values, errors="coerce").to_numpy(float)))
    return items


def feature_items(table):
    numbers = [Item(name, "number", table.numbers[name].to_numpy(float)) for name in table.numbers]
    yes_no = [Item(name, "number", table.yes_no[name].to_numpy(float)) for name in table.yes_no]
    categories = [Item(name, "category", pd.factorize(labels)[0]) for name, labels in table.categories.items()]
    return numbers + yes_no + categories


def dropped_features(matrix, feature_names):
    """Rule 4 first (leaks), then the pair rule, strongest pair first; the feature closer to the label stays."""
    with_label = matrix.loc[feature_names, LABEL_ITEM].abs()
    dropped = {name: f"{with_label[name]:.3f} with the label (rule 4)" for name in feature_names
               if with_label[name] >= LIMIT}
    for first, second, value in pairs_at_or_above(matrix, feature_names, LIMIT):
        if first in dropped or second in dropped:
            continue
        weaker, stronger = (first, second) if with_label[first] < with_label[second] else (second, first)
        dropped[weaker] = (f"{value:+.3f} with {stronger}, which is closer to the label "
                           f"({with_label[stronger]:.3f} against {with_label[weaker]:.3f})")
    return dropped


def pairs_at_or_above(matrix, names, limit):
    block = matrix.loc[names, names].to_numpy()
    rows, cols = np.triu_indices(len(names), k=1)
    values = block[rows, cols]
    strong = np.where(np.abs(values) >= limit)[0]
    ordered = strong[np.argsort(-np.abs(values[strong]))]
    return [(names[rows[i]], names[cols[i]], values[i]) for i in ordered]


def column_check(train, matrix, raw_names, chosen):
    rows = []
    for name in raw_names:
        column = COLUMN_BY_NAME[name]
        partner = strongest_partner(matrix, name, raw_names)
        closest = strongest_partner(matrix, name, chosen)
        rows.append({"column": name, "decision": column.decision, "reason": column.reason,
                     "measured as": column.measured_as,
                     "% blank in training homes": round(train[name].isna().mean() * 100, 2),
                     "values": train[name].nunique(), "with label": matrix.loc[name, LABEL_ITEM],
                     "strongest partner (original columns)": partner["strongest partner"],
                     "strongest partner value": partner["strongest partner value"],
                     "closest chosen feature": closest["strongest partner"],
                     "closest chosen feature value": closest["strongest partner value"]})
    check = pd.DataFrame(rows)
    dropped = check["decision"].isin([DROP, LEAK])
    label_strength = check["with label"].abs()
    check["second look"] = (dropped & (label_strength >= REVIEW_LABEL_LEVEL)
                            & (check["closest chosen feature value"].abs() < COVERED_LEVEL))
    check["comes back (rule 3)"] = ((check["decision"] == DROP) & label_strength.between(*COMEBACK_WINDOW)
                                    & (check["closest chosen feature value"].abs() < LIMIT))
    return check.sort_values("with label", key=np.abs, ascending=False)


def feature_check(matrix, feature_names, dropped):
    rows = [{"feature": name, "with label": matrix.loc[name, LABEL_ITEM],
             **strongest_partner(matrix, name, feature_names),
             "dropped by the rules": dropped.get(name, "")} for name in feature_names]
    return pd.DataFrame(rows).sort_values("with label", key=np.abs, ascending=False)


def save_block(matrix, names, stem):
    block = matrix.loc[[LABEL_ITEM] + names, [LABEL_ITEM] + names]
    block.round(3).to_csv(CHECKS_DIR / f"{stem}_matrix.csv")
    draw_heatmap(block, CHECKS_DIR / f"{stem}_heatmap.png", annotate=len(block) <= 70)

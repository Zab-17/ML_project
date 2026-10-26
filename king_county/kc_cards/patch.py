"""Re-measures nominal code columns as categories, fixes containment, adds sales by year."""
import json
import sys

import numpy as np
import pandas as pd

import measure

measure.KING = sys.argv[1]
STATS = sys.argv[2]
NOMINAL = ["ZipCode", "Area", "SubArea", "LevyCode", "Range", "Township", "Section", "PropertyType",
           "PresentUse", "HeatSystem", "HeatSource", "WaterSystem", "SewerSystem", "Access", "StreetSurface",
           "HBUAsIfVacant", "HBUAsImproved", "InadequateParking", "WfntLocation", "WfntBank",
           "WfntRestrictedAccess", "TidelandShoreland", "CurrentUseDesignation", "HistoricSite", "Contamination",
           "SaleInstrument", "QuarterSection", "DistrictName", "CurrentZoning", "PlatName"]
CONTAIN = ["ZipCode", "DistrictName", "Area", "SubArea", "LevyCode", "PlatName", "CurrentZoning", "Range",
           "Township", "Section", "QuarterSection", "PresentUse", "HBUAsIfVacant", "PropertyType", "StreetName",
           "HeatSystem", "HeatSource", "WfntLocation", "WaterSystem", "SewerSystem"]
DOMINANT_LIMIT = 0.95

train, summary = measure.training_rows()
log_price = np.log(train["_price"])
adjusted = log_price - log_price.groupby(train["_date"].dt.to_period("M")).transform("median")
names = measure.lookup_names()
data = json.load(open(STATS))
cards = {c["name"]: c for c in data["cards"]}

for column in NOMINAL:
    values = train[column].str.replace(r"\.0$", "", regex=True)
    card = cards[column]
    card["nominal"] = True
    card["bars"] = measure.category_bars(values, adjusted, names.get(column, {}))
    card["relation"] = {"kind": "categories", "points": None, "explained": measure.explained_share(values, adjusted)}
    if card.get("numeric"):
        card["numeric"]["spearman"] = None
    card.pop("similar", None)

sample = train.sample(n=measure.SAMPLE_ROWS, random_state=measure.SEED)
usable = [c for c in CONTAIN if sample[c][sample[c] != ""].value_counts(normalize=True).iloc[0] < DOMINANT_LIMIT]
for card in cards.values():
    card["nested_in"] = []
for inner in usable:
    for outer in usable:
        if inner != outer and measure.nesting_purity(sample[inner], sample[outer]) >= measure.NESTING_PURITY:
            cards[inner]["nested_in"].append([outer, measure.nesting_purity(sample[outer], sample[inner]) >= measure.NESTING_PURITY])

year = train["_date"].dt.year
by_year = train.groupby(year)["_price"].agg(["size", "median"])
cards["DocumentDate"]["years"] = [[int(y), int(r["size"]), int(r["median"])] for y, r in by_year.iterrows()]
for column in ["Volume", "Page", "PlatNbr", "PlatType", "PlatLot", "PlatBlock"]:
    cards[column]["note_all_years"] = True
json.dump(data, open(STATS, "w"))
for column in NOMINAL:
    c = cards[column]
    print(f"{column:22s} explained {c['relation']['explained']:.4f} distinct {c['distinct']} top {c['bars']['items'][:2]}")
print("containment:", {k: v["nested_in"] for k, v in cards.items() if v["nested_in"]})
print("years:", cards["DocumentDate"]["years"][:3], "...", cards["DocumentDate"]["years"][-2:])

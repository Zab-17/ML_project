"""Final model columns for the King County training sales, built the way FEATURE_DECISIONS.md says.

Three kinds of column come back:
- numbers: one value per sale (standardizing is left to the caller; it does not change a correlation),
- yes/no: 0 or 1 per sale,
- categories: one text label per sale; one_hot() turns every label into its own yes/no column.
"""
# AI was used to help write this code, but the resulting code was reviewed and edited by a human.

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "kc_cards"))
sys.path.insert(0, str(HERE.parent / "kc_check"))
from measure import split_rows, training_rows  # noqa: E402
from feature_table import code_names  # noqa: E402
from grid_table import approximate_position  # noqa: E402
from location_table import text  # noqa: E402

START_OF_2000 = pd.Timestamp("2000-01-01")
DAYS_PER_YEAR = 365.25
MIN_SHARE_FOR_OWN_COLUMN = 0.001
OTHER = "other"
SMALLEST_NAME_THRESHOLD = 50  # subdivision and street name: the larger thresholds keep a subset of these names

BATH_COLUMNS = ["BathFullCount", "Bath3qtrCount", "BathHalfCount"]
FIREPLACE_COLUMNS = ["FpSingleStory", "FpMultiStory", "FpFreestanding", "FpAdditional"]
VIEWS = {"MtRainier": "Mount Rainier", "Olympics": "Olympic Mountains", "Cascades": "Cascade Mountains",
         "Territorial": "territorial", "SeattleSkyline": "Seattle skyline", "PugetSound": "Puget Sound",
         "LakeWashington": "Lake Washington", "LakeSammamish": "Lake Sammamish",
         "SmallLakeRiverCreek": "small lake, river or creek", "OtherView": "other"}
YES_NO_FLAGS = {"PowerLines": "power lines nearby", "AdjacentGolfFairway": "next to a golf course",
                "Easements": "easement", "DeedRestrictions": "deed restrictions",
                "NativeGrowthProtEsmt": "protected native-plants strip", "OtherDesignation": "other legal designation",
                "DNRLease": "land leased from the state", "WaterProblems": "water problems",
                "AdjacentGreenbelt": "next to a green belt"}
HEAT_SYSTEMS = {"5": "forced air", "7": "heat pump / air conditioning", "4": "electric baseboard",
                "1": "floor or wall heater", "6": "hot water", "3": "radiant"}
HEAT_FUELS = {"2": "gas", "3": "electricity", "1": "oil", "5": "gas + solar", "6": "electricity + solar"}
WATERFRONTS = {"0": "not on the water", "6": "Lake Washington", "7": "Lake Sammamish", "3": "Puget Sound",
               "8": "other lake", "9": "river", "1": "river"}  # 1 is the Duwamish River
WATERFRONT_BANKS = {"0": "none", "1": "low", "4": "low", "2": "medium", "3": "high"}  # 4 is "no bank"
NAMED_CODE_COLUMNS = {"PresentUse": "present use", "Access": "road access", "StreetSurface": "street surface",
                      "InadequateParking": "parking"}


@dataclass
class FeatureTable:
    numbers: pd.DataFrame
    yes_no: pd.DataFrame
    categories: dict
    label: pd.Series
    month_adjusted_label: pd.Series


def final_training_table(subdivision_min_sales=SMALLEST_NAME_THRESHOLD, street_min_sales=SMALLEST_NAME_THRESHOLD):
    train, _ = training_rows()
    return build_feature_table(without_sales_missing_rooms(train), subdivision_min_sales, street_min_sales)


def number(train, column):
    return pd.to_numeric(train[column], errors="coerce").fillna(0)


def without_sales_missing_rooms(train):
    bathrooms = sum(number(train, column) for column in BATH_COLUMNS)
    return train[(number(train, "Bedrooms") > 0) & (bathrooms > 0)].reset_index(drop=True)


def build_feature_table(train, subdivision_min_sales, street_min_sales, fit=None):
    """fit marks the training sales: every count, list and median is learned from those rows only."""
    fit = pd.Series(True, index=train.index) if fit is None else pd.Series(fit, index=train.index)
    label = np.log(train["_price"])
    month_median = label.where(fit).groupby(train["_date"].dt.to_period("M")).transform("median")
    return FeatureTable(numeric_inputs(train), yes_no_inputs(train),
                        category_inputs(train, subdivision_min_sales, street_min_sales, fit),
                        label, label - month_median)


def numeric_inputs(train):
    living, finished = number(train, "SqFtTotLiving"), number(train, "SqFtFinBasement")
    basement_garage = number(train, "SqFtGarageBasement")
    east, north = approximate_position(train)
    numbers = pd.DataFrame({
        "years since 2000": (train["_date"] - START_OF_2000).dt.days / DAYS_PER_YEAR,
        "above-ground living area (log)": np.log(positive(living - finished, "above-ground living area")),
        "other unfinished basement": number(train, "SqFtTotBasement") - finished - basement_garage,
        "finished basement": finished,
        "total garage": number(train, "SqFtGarageAttached") + basement_garage,
        "open porch": number(train, "SqFtOpenPorch"),
        "deck": number(train, "SqFtDeck"),
        "bedrooms": number(train, "Bedrooms"),
        "full bathrooms": number(train, "BathFullCount"),
        "three-quarter bathrooms": number(train, "Bath3qtrCount"),
        "half bathrooms": number(train, "BathHalfCount"),
        "stories": number(train, "Stories"),
        "year built": number(train, "YrBuilt"),
        "fireplaces": sum(number(train, column) for column in FIREPLACE_COLUMNS),
        "miles east": east,
        "miles north": north,
        "waterfront length": number(train, "WfntFootage"),
        "lot area (log)": np.log(positive(number(train, "SqFtLot"), "lot area")),
        "traffic noise": number(train, "TrafficNoise"),
    })
    return without_missing_values(numbers)


def positive(values, name):
    if (values <= 0).any():
        raise ValueError(f"{name}: {(values <= 0).sum()} sales are zero or negative, so the log is undefined")
    return values


def without_missing_values(numbers):
    missing = numbers.isna().sum()
    if missing.any():
        raise ValueError(f"missing values in numeric columns: {missing[missing > 0].to_dict()}")
    return numbers


def yes_no_inputs(train):
    columns = {f"view: {name}": number(train, column) > 0 for column, name in VIEWS.items()}
    columns["tidelands or shorelands"] = number(train, "TidelandShoreland") > 0
    columns["hilly lot"] = number(train, "Topography") > 0
    columns["airport noise"] = number(train, "AirportNoise") > 0
    columns.update({name: text(train, column) == "Y" for column, name in YES_NO_FLAGS.items()})
    columns["private water supply"] = text(train, "WaterSystem") == "1"
    columns["private sewer"] = text(train, "SewerSystem") == "1"
    return pd.DataFrame(columns).astype(float)


def category_inputs(train, subdivision_min_sales, street_min_sales, fit):
    area = text(train, "Area")
    categories = {
        "renovated": renovation_decade(train),
        "building grade": building_grade(train),
        "finished basement grade": finished_basement_grade(train),
        "heating system": text(train, "HeatSystem").map(HEAT_SYSTEMS).fillna(OTHER),
        "heating fuel": text(train, "HeatSource").map(HEAT_FUELS).fillna(OTHER),
        "city": text(train, "DistrictName"),
        "appraisal area": area,
        "area + sub-area": area + "-" + text(train, "SubArea"),
        "subdivision": names_with_enough_sales(text(train, "PlatName"), subdivision_min_sales, fit),
        "street name": names_with_enough_sales(text(train, "StreetName"), street_min_sales, fit),
        "street type": fold_rare_values(text(train, "StreetType"), fit),
        "waterfront": text(train, "WfntLocation").map(WATERFRONTS).fillna(OTHER),
        "waterfront bank": text(train, "WfntBank").map(WATERFRONT_BANKS).fillna(OTHER),
    }
    categories.update({name: fold_rare_values(named_codes(train, column), fit)
                       for column, name in NAMED_CODE_COLUMNS.items()})
    return {group: only_training_labels(labels, fit) for group, labels in categories.items()}


def renovation_decade(train):
    year = number(train, "YrRenovated")
    decade = (year // 10 * 10).clip(lower=1960).astype(int).astype(str) + "s"
    return decade.replace("1960s", "1960s or earlier").where(year > 0, "never")


def building_grade(train):
    return number(train, "BldgGrade").clip(5, 12).astype(int).astype(str)


def finished_basement_grade(train):
    grade = number(train, "FinBasementGrade")
    return grade.where(grade == 0, grade.clip(5, 11)).astype(int).astype(str)


def names_with_enough_sales(names, min_sales, fit):
    counts = names[fit].value_counts()
    return names.where(names.map(counts) >= min_sales, OTHER)


def named_codes(train, column):
    names = code_names(column)
    return text(train, column).map(lambda code: names.get(code) or f"code {code}")


def fold_rare_values(labels, fit):
    """Values under 0.1% of training sales join "other"; if "other" itself stays under 0.1%, it joins the most common value."""
    shares = labels[fit].value_counts(normalize=True)
    folded = labels.where(labels.map(shares).fillna(0) >= MIN_SHARE_FOR_OWN_COLUMN, OTHER)
    if (folded[fit] == OTHER).mean() < MIN_SHARE_FOR_OWN_COLUMN:
        folded = folded.replace(OTHER, shares.index[0])
    return folded


def only_training_labels(labels, fit):
    """A label never seen in the training sales becomes the most common training label."""
    seen = labels[fit].value_counts()
    return labels.where(labels.isin(seen.index), seen.index[0])


def one_hot(categories):
    """One sparse yes/no column per label of every category group, plus the column names and their groups."""
    blocks, names, groups = [], [], []
    for group, labels in categories.items():
        codes, uniques = pd.factorize(labels, sort=True)
        rows = np.arange(len(labels))
        blocks.append(sparse.csr_matrix((np.ones(len(labels)), (rows, codes)), shape=(len(labels), len(uniques))))
        names += [f"{group} = {label}" for label in uniques]
        groups += [group] * len(uniques)
    return sparse.hstack(blocks).tocsc(), names, groups

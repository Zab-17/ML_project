"""The Zillow features: numbers, yes/no columns and categories, in the King County FeatureTable shape.

Every list and every fill-in value is learned from the training homes only (the rows where fit is True); the test
homes reuse them. Categories follow the King County rules: a blank is its own value ("not recorded"), a value under
0.1% of training homes joins "other", and a category left with two values becomes one yes/no column.
"""
# AI was used to help write this code, but the resulting code was reviewed and edited by a human.

from functools import lru_cache

import numpy as np
import pandas as pd

import shared_code  # noqa: F401  (puts the King County code on the import path)
from code_tables import decode_tables
from features import FeatureTable, fold_rare_values, only_training_labels
from zillow_rows import LABEL

NOT_RECORDED = "not recorded"
COORDINATE_SCALE = 1_000_000      # Zillow stores degrees times a million
MILES_PER_DEGREE = 69.05          # one degree of latitude; a degree of longitude is this times cos(latitude)
TRACT_DIGITS = 100                # rawcensustractandblock = county + tract "." block: x 100 keeps county + tract

CATEGORY_SOURCES = {
    "building quality": "buildingqualitytypeid",
    "heating": "heatingorsystemtypeid",
    "air conditioning": "airconditioningtypeid",
    "property type": "propertylandusetypeid",
    "county land-use code": "propertycountylandusecode",
    "county": "fips",
    "city": "regionidcity",
    "ZIP code": "regionidzip",
    "neighbourhood": "regionidneighborhood",
    "zoning": "propertyzoningdesc",
}


def build_feature_table(rows, fit):
    label = np.log(rows[LABEL]).rename("log assessed value")
    categories = category_inputs(rows, fit)
    two_valued = {group: labels for group, labels in categories.items() if labels.nunique() == 2}
    yes_no = pd.concat([yes_no_inputs(rows), as_yes_no(two_valued, fit)], axis=1)
    many_valued = {group: labels for group, labels in categories.items() if group not in two_valued}
    return FeatureTable(number_inputs(rows, fit), yes_no, many_valued, label, label)


def number_inputs(rows, fit):
    numbers = pd.DataFrame({
        "bathrooms": rows["bathroomcnt"],
        "bedrooms": rows["bedroomcnt"],
        "year built": rows["yearbuilt"],
        "living area (log)": log_of_positive(rows["calculatedfinishedsquarefeet"]),
        "lot area (log)": log_of_positive(rows["lotsizesquarefeet"]),
        "stories": rows["numberofstories"],
        "garage area": rows["garagetotalsqft"].fillna(0),
        "fireplaces": rows["fireplacecnt"].fillna(0),
        **miles_from_centre(rows, fit),
    })
    return numbers.fillna(numbers[fit].median())


def log_of_positive(values):
    """The log, with 0 and negative areas treated as blank (they would become minus infinity)."""
    return np.log(values.where(values > 0))


def miles_from_centre(rows, fit):
    """Miles north and east of the median training home. Raw degrees are not distances: a degree of longitude
    is 57 miles here, a degree of latitude 69."""
    latitude, longitude = rows["latitude"] / COORDINATE_SCALE, rows["longitude"] / COORDINATE_SCALE
    centre_latitude, centre_longitude = latitude[fit].median(), longitude[fit].median()
    east_miles_per_degree = MILES_PER_DEGREE * np.cos(np.radians(centre_latitude))
    return {"miles north": (latitude - centre_latitude) * MILES_PER_DEGREE,
            "miles east": (longitude - centre_longitude) * east_miles_per_degree}


def yes_no_inputs(rows):
    return pd.DataFrame({"pool": rows["poolcnt"].notna().astype(int),
                         "hot tub or spa": rows["hashottuborspa"].notna().astype(int)})


def category_inputs(rows, fit):
    labels = {group: named_codes(rows[column], column) for group, column in CATEGORY_SOURCES.items()}
    labels["census tract"] = census_tracts(rows["rawcensustractandblock"])
    return {group: only_training_labels(fold_rare_values(values, fit), fit) for group, values in labels.items()}


def named_codes(values, column):
    """Codes as text, swapped for Zillow's own names where its dictionary has them."""
    text = code_text(values)
    names = code_names().get(column, {})
    return text.map(lambda code: names.get(code, code))


@lru_cache(maxsize=1)
def code_names():
    return decode_tables()


def code_text(values):
    """A code as text without a trailing '.0'; a blank becomes 'not recorded'."""
    return values.astype("string").str.replace(r"\.0$", "", regex=True).fillna(NOT_RECORDED).astype(object)


def census_tracts(tract_and_block):
    tracts = np.floor(tract_and_block * TRACT_DIGITS).astype("Int64")
    return code_text(tracts)


def as_yes_no(categories, fit):
    """A category with two values: 1 for the less common training value, 0 for the other."""
    columns = {}
    for group, labels in categories.items():
        rarer = labels[fit].value_counts().index[-1]
        columns[f"{group} = {rarer}"] = (labels == rarer).astype(int)
    return pd.DataFrame(columns, index=next(iter(categories.values())).index) if columns else pd.DataFrame()

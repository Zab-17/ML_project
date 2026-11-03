"""Which Zillow homes are kept, and how they are split into training, test and five folds.

The row rules copy the King County rules as far as the Zillow file allows: one ordinary residential home with a
building on it, a believable value, and one assessment year. Phase 2 kept every row with a label, which let in
shops, parcels with no building, and assessed values of $1.
"""
# AI was used to help write this code, but the resulting code was reviewed and edited by a human.

from dataclasses import dataclass

import numpy as np
import pandas as pd

from shared_code import PROJECT

RAW_FILE = PROJECT / "data" / "raw" / "properties_2017.csv"
LABEL = "taxvaluedollarcnt"
ASSESSMENT_YEAR = 2016
MIN_ASSESSED_VALUE = 10_000      # the King County price floor
DWELLING_TYPES = {261, 266, 246, 247, 248, 269, 263, 265, 260, 267, 275, 264}
# Left out: 31 and 47 (shops and offices with flats), 270 (a residential common area, no home), 279 (one row).
TEST_SHARE = 0.2
SEED = 42
FOLDS = 5


@dataclass
class SplitRows:
    train: pd.DataFrame
    test: pd.DataFrame
    folds: np.ndarray            # fold 0 to 4 for every training home
    dropped_by_rule: dict        # rule -> homes it removed, in the order the rules run


def split_rows(sample_homes=None):
    """Kept homes, split 80/20 at random (each home is one row, so no home can sit on both sides)."""
    homes, dropped = kept_homes(pd.read_csv(RAW_FILE, low_memory=False))
    if sample_homes:
        homes = homes.sample(n=sample_homes, random_state=SEED)
    shuffled = homes.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    test_size = int(len(shuffled) * TEST_SHARE)
    train, test = shuffled.iloc[test_size:].reset_index(drop=True), shuffled.iloc[:test_size].reset_index(drop=True)
    return SplitRows(train, test, home_folds(train), dropped)


def kept_homes(raw):
    rules = {
        "no assessed value": raw[LABEL].isna(),
        f"assessed in a year other than {ASSESSMENT_YEAR}": raw["assessmentyear"] != ASSESSMENT_YEAR,
        f"assessed value under ${MIN_ASSESSED_VALUE:,}": raw[LABEL] < MIN_ASSESSED_VALUE,
        "not a residential home (shop, office, common area or no type)": ~raw["propertylandusetypeid"].isin(DWELLING_TYPES),
        "no living area recorded (no building)": raw["calculatedfinishedsquarefeet"].isna(),
        "0 bedrooms or no bathroom": ~((raw["bedroomcnt"] >= 1) & (raw["bathroomcnt"] > 0)),
    }
    removed, dropped = pd.Series(False, index=raw.index), {"rows in the raw file": len(raw)}
    for rule, failing in rules.items():
        dropped[rule] = int((failing & ~removed).sum())
        removed |= failing
    dropped["homes kept"] = int((~removed).sum())
    return raw[~removed], dropped


def home_folds(train):
    """Fold 0 to 4 per training home, from its parcel number, as in King County. Fold 0 is the validation part."""
    return (pd.util.hash_pandas_object(train["parcelid"], index=False) % FOLDS).to_numpy()

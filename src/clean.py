"""Phase 2 cleaning and pre-processing for the Zillow properties dataset.

Reads data/raw/properties_2017.csv, applies the feature decisions recorded in
changes.md, and writes a model-ready training and test set plus the numbers the
Phase 2 report has to quote.

Everything that learns from data — medians, category lists, frequencies, means
and standard deviations — is fitted on the training rows only, so no test
information reaches the training set.
"""
import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from code_tables import decode_tables

PROJECT = Path(__file__).resolve().parent.parent
RAW_PATH = PROJECT / "data" / "raw" / "properties_2017.csv"
PROCESSED_DIR = PROJECT / "data" / "processed"

LABEL = "taxvaluedollarcnt"
TEST_FRACTION = 0.2
RANDOM_SEED = 42
LEAKAGE_CORRELATION_LIMIT = 0.85  # professor's rule: drop features at or above this
SMALLEST_KEPT_CATEGORY_SHARE = 0.0001  # 0.01% of training rows, about 236 rows
ZONING_CATEGORIES_KEPT = 100

COUNT_FEATURES = ["bathroomcnt", "bedroomcnt", "yearbuilt"]
SKEWED_AREA_FEATURES = ["calculatedfinishedsquarefeet", "lotsizesquarefeet"]
COORDINATE_FEATURES = ["latitude", "longitude"]
COORDINATE_SCALE = 1_000_000
ONE_HOT_FEATURES = [
    "buildingqualitytypeid",
    "fips",
    "heatingorsystemtypeid",
    "propertylandusetypeid",
    "propertycountylandusecode",
]
ZONING_FEATURE = "propertyzoningdesc"
UNKNOWN_CATEGORY = "Unknown"
OTHER_CATEGORY = "Other"

RAW_COLUMNS = (COUNT_FEATURES + SKEWED_AREA_FEATURES + COORDINATE_FEATURES
               + ONE_HOT_FEATURES + [ZONING_FEATURE] + [LABEL])
TEXT_DTYPES = {column: "string" for column in ONE_HOT_FEATURES + [ZONING_FEATURE]}


@dataclass
class FittedParameters:
    """Everything learned from the training rows, reused unchanged on the test rows."""
    medians: dict[str, float] = field(default_factory=dict)
    categories: dict[str, list[str]] = field(default_factory=dict)
    means: dict[str, float] = field(default_factory=dict)
    standard_deviations: dict[str, float] = field(default_factory=dict)


def load_raw() -> pd.DataFrame:
    return pd.read_csv(RAW_PATH, usecols=RAW_COLUMNS, dtype=TEXT_DTYPES)


def drop_rows_without_label(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[frame[LABEL].notna()]


def split_rows(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    shuffled = frame.sample(frac=1.0, random_state=RANDOM_SEED)
    test_size = int(len(shuffled) * TEST_FRACTION)
    return shuffled.iloc[test_size:], shuffled.iloc[:test_size]


def scaled_coordinates(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[COORDINATE_FEATURES] / COORDINATE_SCALE


def logged_areas(frame: pd.DataFrame) -> pd.DataFrame:
    return np.log1p(frame[SKEWED_AREA_FEATURES])


def numeric_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """The numeric features after unit conversion and the log transformation, before imputation."""
    return pd.concat([frame[COUNT_FEATURES], logged_areas(frame), scaled_coordinates(frame)], axis=1)


def frequent_categories(values: pd.Series, limit: int | None = None) -> list[str]:
    counts = values.value_counts()
    frequent = counts[counts >= len(values) * SMALLEST_KEPT_CATEGORY_SHARE]
    kept = frequent if limit is None else frequent.head(limit)
    return sorted(kept.index.astype(str))


@lru_cache(maxsize=1)
def code_names() -> dict:
    """Code-to-name tables for the identifier columns the data dictionary documents."""
    return decode_tables()


def named_categories(values: pd.Series, column: str) -> pd.Series:
    """Replace a numeric type code with its documented name, so the features read as words."""
    table = code_names().get(column)
    if table is None:
        return values
    plain = values.str.replace(r"\.0$", "", regex=True).str.lstrip("0")
    return plain.map(table).fillna(plain.radd(f"{column}_code_"))


def fit(train: pd.DataFrame) -> FittedParameters:
    parameters = FittedParameters()
    numeric = numeric_frame(train)
    parameters.medians = numeric.median().to_dict()
    for column in ONE_HOT_FEATURES:
        parameters.categories[column] = frequent_categories(named_categories(train[column], column))
    parameters.categories[ZONING_FEATURE] = frequent_categories(train[ZONING_FEATURE],
                                                                ZONING_CATEGORIES_KEPT)
    standardised = numeric.fillna(parameters.medians)
    parameters.means = standardised.mean().to_dict()
    parameters.standard_deviations = standardised.std().replace(0, 1).to_dict()
    return parameters


def standardised_numeric(frame: pd.DataFrame, parameters: FittedParameters) -> pd.DataFrame:
    numeric = numeric_frame(frame).fillna(parameters.medians)
    centred = numeric - pd.Series(parameters.means)
    return (centred / pd.Series(parameters.standard_deviations)).astype("float32")


def one_hot(values: pd.Series, column: str, categories: list[str], add_other: bool) -> pd.DataFrame:
    rare = values.notna() & ~values.isin(categories)  # missing must not be mistaken for rare
    known = values.mask(rare, OTHER_CATEGORY if add_other else UNKNOWN_CATEGORY).fillna(UNKNOWN_CATEGORY)
    wanted = categories + ([OTHER_CATEGORY] if add_other else []) + [UNKNOWN_CATEGORY]
    dummies = pd.get_dummies(known, prefix=column, dtype="uint8")
    wanted_columns = [f"{column}_{name}" for name in wanted]
    return dummies.reindex(columns=wanted_columns, fill_value=0).astype("uint8")


def encoded_categories(frame: pd.DataFrame, parameters: FittedParameters) -> list[pd.DataFrame]:
    encoded = [one_hot(named_categories(frame[column], column), column,
                       parameters.categories[column], add_other=True)
               for column in ONE_HOT_FEATURES]
    encoded.append(one_hot(frame[ZONING_FEATURE], ZONING_FEATURE,
                           parameters.categories[ZONING_FEATURE], add_other=True))
    return encoded


def transform(frame: pd.DataFrame, parameters: FittedParameters) -> pd.DataFrame:
    parts = [standardised_numeric(frame, parameters)]
    parts.extend(encoded_categories(frame, parameters))
    return pd.concat(parts, axis=1).reset_index(drop=True)


def label_correlations(features: pd.DataFrame, label: pd.Series) -> pd.Series:
    centred_label = (label - label.mean()).to_numpy(dtype="float32")
    label_spread = np.sqrt(np.square(centred_label).sum())
    correlations = {}
    for column in features.columns:
        values = features[column].to_numpy(dtype="float32")
        centred = values - values.mean()
        spread = np.sqrt(np.square(centred).sum())
        correlations[column] = 0.0 if spread == 0 else float(centred @ centred_label / (spread * label_spread))
    return pd.Series(correlations)


def leaking_features(correlations: pd.Series) -> list[str]:
    return sorted(correlations[correlations.abs() >= LEAKAGE_CORRELATION_LIMIT].index)


def constant_features(features: pd.DataFrame) -> list[str]:
    """Columns that never vary teach a model nothing and break anything that divides by variance."""
    return sorted(features.columns[features.nunique() == 1])


def write_outputs(train: pd.DataFrame, test: pd.DataFrame, report: dict) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train.to_parquet(PROCESSED_DIR / "train.parquet", index=False)
    test.to_parquet(PROCESSED_DIR / "test.parquet", index=False)
    (PROCESSED_DIR / "cleaning_report.json").write_text(json.dumps(report, indent=2))


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    raw = load_raw()
    labelled = drop_rows_without_label(raw)
    train_rows, test_rows = split_rows(labelled)

    parameters = fit(train_rows)
    train_features = transform(train_rows, parameters)
    test_features = transform(test_rows, parameters)

    correlations = label_correlations(train_features, train_rows[LABEL])
    dropped_for_leakage = leaking_features(correlations)
    dropped_for_no_variation = constant_features(train_features.drop(columns=dropped_for_leakage))
    discarded = dropped_for_leakage + dropped_for_no_variation
    train_features = train_features.drop(columns=discarded)
    test_features = test_features.drop(columns=discarded)

    correlations.sort_values(key=abs, ascending=False).to_csv(
        PROCESSED_DIR / "label_correlations.csv", header=["correlation_with_label"])

    report = {
        "rows_in_raw_file": len(raw),
        "rows_dropped_missing_label": len(raw) - len(labelled),
        "rows_kept": len(labelled),
        "training_rows": len(train_rows),
        "test_rows": len(test_rows),
        "final_feature_count": train_features.shape[1],
        "values_imputed_per_numeric_feature": {
            column: int(numeric_frame(labelled)[column].isna().sum()) for column in parameters.medians},
        "medians_used": {k: float(v) for k, v in parameters.medians.items()},
        "categories_kept_per_feature": {k: len(v) for k, v in parameters.categories.items()},
        "features_dropped_for_label_correlation": dropped_for_leakage,
        "features_dropped_for_no_variation": dropped_for_no_variation,
        "highest_label_correlations": correlations.abs().sort_values(ascending=False).head(10).round(4).to_dict(),
    }
    write_outputs(pd.concat([train_features, train_rows[[LABEL]].reset_index(drop=True)], axis=1),
                  pd.concat([test_features, test_rows[[LABEL]].reset_index(drop=True)], axis=1),
                  report)
    print(json.dumps(report, indent=2)[:2000])


if __name__ == "__main__":
    main()

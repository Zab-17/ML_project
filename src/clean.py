"""Phase 2 cleaning and pre-processing for the Zillow properties dataset.

Reads data/raw/properties_2017.csv, applies the feature decisions recorded in
changes.md, and writes a model-ready training and test set plus the numbers the
Phase 2 report has to quote.

Everything that learns from data — medians, category lists, frequencies, means
and standard deviations — is fitted on the training rows only, so no test
information reaches the training set.
"""
# AI was used to help write this code, but the resulting code was reviewed and edited by a human.
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

COUNT_FEATURES = ["bathroomcnt", "bedroomcnt", "yearbuilt"]
SKEWED_AREA_FEATURES = ["calculatedfinishedsquarefeet", "lotsizesquarefeet"]
COORDINATE_FEATURES = ["latitude", "longitude"]
COORDINATE_SCALE = 1_000_000
ONE_HOT_FEATURES = [
    "buildingqualitytypeid",
    "fips",
    "heatingorsystemtypeid",
    "propertylandusetypeid",
]
# 234 county codes would add 234 columns, so this one is encoded as how common each code is.
FREQUENCY_FEATURES = ["propertycountylandusecode"]
# propertyzoningdesc was considered and dropped: 33.59% of rows have no code, and the
# location information it carries is already represented by latitude and longitude.
UNKNOWN_CATEGORY = "Unknown"
OTHER_CATEGORY = "Other"

RAW_COLUMNS = (COUNT_FEATURES + SKEWED_AREA_FEATURES + COORDINATE_FEATURES
               + ONE_HOT_FEATURES + FREQUENCY_FEATURES + [LABEL])
TEXT_DTYPES = {column: "string" for column in ONE_HOT_FEATURES + FREQUENCY_FEATURES}


@dataclass
class FittedParameters:
    """Everything learned from the training rows, reused unchanged on the test rows."""
    medians: dict[str, float] = field(default_factory=dict)
    categories: dict[str, list[str]] = field(default_factory=dict)
    frequencies: dict[str, dict[str, float]] = field(default_factory=dict)
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


def frequency_encoded(frame: pd.DataFrame, parameters: "FittedParameters") -> pd.DataFrame:
    """Each category becomes how large a share of the training rows it holds.

    A code never seen in training, and a missing code, both become 0: no training
    row supports them, so the honest encoded share is zero.
    """
    columns = {f"{column}_frequency": frame[column].map(parameters.frequencies[column]).fillna(0.0)
               for column in FREQUENCY_FEATURES}
    return pd.DataFrame(columns, index=frame.index)


def numeric_frame(frame: pd.DataFrame, parameters: "FittedParameters | None" = None) -> pd.DataFrame:
    """The numeric features after unit conversion and the log transformation, before imputation."""
    parts = [frame[COUNT_FEATURES], logged_areas(frame), scaled_coordinates(frame)]
    if parameters is not None:
        parts.append(frequency_encoded(frame, parameters))
    return pd.concat(parts, axis=1)


def frequent_categories(values: pd.Series) -> list[str]:
    counts = values.value_counts()
    frequent = counts[counts >= len(values) * SMALLEST_KEPT_CATEGORY_SHARE]
    return sorted(frequent.index.astype(str))


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
    for column in FREQUENCY_FEATURES:
        parameters.frequencies[column] = train[column].value_counts(normalize=True).to_dict()
    numeric = numeric_frame(train, parameters)
    parameters.medians = numeric.median().to_dict()
    for column in ONE_HOT_FEATURES:
        parameters.categories[column] = frequent_categories(named_categories(train[column], column))
    standardised = numeric.fillna(parameters.medians)
    parameters.means = standardised.mean().to_dict()
    parameters.standard_deviations = standardised.std().replace(0, 1).to_dict()
    return parameters


def standardised_numeric(frame: pd.DataFrame, parameters: FittedParameters) -> pd.DataFrame:
    numeric = numeric_frame(frame, parameters).fillna(parameters.medians)
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
        correlations[column] = (0.0 if spread == 0
                                else float(centred @ centred_label / (spread * label_spread)))
    return pd.Series(correlations)


def leaking_features(correlations: pd.Series) -> list[str]:
    return sorted(correlations[correlations.abs() >= LEAKAGE_CORRELATION_LIMIT].index)


def constant_features(features: pd.DataFrame) -> list[str]:
    """Columns that never vary teach a model nothing and break anything that divides by variance."""
    return sorted(features.columns[features.nunique() == 1])


def with_label(features: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([features, rows[[LABEL]].reset_index(drop=True)], axis=1)


def save_correlations(correlations: pd.Series) -> None:
    ranked = correlations.sort_values(key=abs, ascending=False)
    ranked.to_csv(PROCESSED_DIR / "label_correlations.csv", header=["correlation_with_label"])


def save_datasets(train: pd.DataFrame, test: pd.DataFrame, report: dict) -> None:
    train.to_parquet(PROCESSED_DIR / "train.parquet", index=False)
    test.to_parquet(PROCESSED_DIR / "test.parquet", index=False)
    (PROCESSED_DIR / "cleaning_report.json").write_text(json.dumps(report, indent=2))


def summarise(raw: pd.DataFrame, labelled: pd.DataFrame, train_rows: pd.DataFrame,
              test_rows: pd.DataFrame, features: pd.DataFrame, parameters: FittedParameters,
              correlations: pd.Series, dropped: dict[str, list[str]]) -> dict:
    """The numbers the Phase 2 report quotes, gathered in one place."""
    missing_per_numeric_feature = numeric_frame(labelled, parameters).isna().sum()
    strongest = correlations.abs().sort_values(ascending=False).head(10)
    return {
        "rows_in_raw_file": len(raw),
        "rows_dropped_missing_label": len(raw) - len(labelled),
        "rows_kept": len(labelled),
        "training_rows": len(train_rows),
        "test_rows": len(test_rows),
        "final_feature_count": features.shape[1],
        "values_imputed_per_numeric_feature": missing_per_numeric_feature.astype(int).to_dict(),
        "medians_used": {name: float(value) for name, value in parameters.medians.items()},
        "categories_kept_per_feature": {name: len(values) for name, values in parameters.categories.items()},
        "categories_frequency_encoded": {name: len(table) for name, table in parameters.frequencies.items()},
        "features_dropped_for_label_correlation": dropped["label_correlation"],
        "features_dropped_for_no_variation": dropped["no_variation"],
        "highest_label_correlations": strongest.round(4).to_dict(),
    }


def unusable_features(features: pd.DataFrame, correlations: pd.Series) -> dict[str, list[str]]:
    """Features a model must not see: those that leak the label, and those that never vary."""
    leaking = leaking_features(correlations)
    return {"label_correlation": leaking, "no_variation": constant_features(features.drop(columns=leaking))}


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    raw = load_raw()
    labelled = drop_rows_without_label(raw)
    train_rows, test_rows = split_rows(labelled)

    parameters = fit(train_rows)
    train_features = transform(train_rows, parameters)
    test_features = transform(test_rows, parameters)

    correlations = label_correlations(train_features, train_rows[LABEL])
    dropped = unusable_features(train_features, correlations)
    discarded = dropped["label_correlation"] + dropped["no_variation"]
    train_features = train_features.drop(columns=discarded)
    test_features = test_features.drop(columns=discarded)

    report = summarise(raw, labelled, train_rows, test_rows, train_features,
                       parameters, correlations, dropped)
    save_correlations(correlations)
    save_datasets(with_label(train_features, train_rows), with_label(test_features, test_rows), report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

"""Measures every column of the Zillow properties file for the column cards.

Every number comes from the training rows of the same 80/20 split that clean.py
makes (seed 42). The test rows are set aside right after the split and never read.
"""
# AI was used to help write this code, but the resulting code was reviewed and edited by a human.
import zipfile
from dataclasses import dataclass

import numpy as np
import pandas as pd

import code_tables
from clean import LABEL, RANDOM_SEED, RAW_PATH, drop_rows_without_label, split_rows

IDENTIFIER_COLUMNS = ["parcelid"]
CATEGORY_COLUMNS = [
    "airconditioningtypeid", "architecturalstyletypeid", "buildingclasstypeid",
    "buildingqualitytypeid", "decktypeid", "fips", "heatingorsystemtypeid",
    "pooltypeid10", "pooltypeid2", "pooltypeid7", "propertycountylandusecode",
    "propertylandusetypeid", "propertyzoningdesc", "rawcensustractandblock",
    "censustractandblock", "regionidcity", "regionidcounty", "regionidneighborhood",
    "regionidzip", "storytypeid", "typeconstructiontypeid", "hashottuborspa",
    "fireplaceflag", "taxdelinquencyflag",
]
LAND_USE = "propertylandusetypeid"
LABEL_PARTS = ["structuretaxvaluedollarcnt", "landtaxvaluedollarcnt"]
EMPTY = "(empty)"

MIN_GROUP_ROWS = 30  # a group's average is trusted only with at least this many rows
NUMERIC_BIN_COUNT = 20  # numeric columns are cut into this many equal-count bins
SAMPLE_ROWS = 200_000  # rows used for the pairwise column comparisons
NESTING_PURITY = 0.99  # share of rows that must agree before one column "sits inside" another
SUM_RULE_TOLERANCE = 1.0  # dollars of rounding allowed in structure + land = total
TOP_LAND_USES = 6
TOP_CATEGORIES = 12
PERCENTILES = [0.01, 0.25, 0.5, 0.75, 0.99]
FENCE_WIDTH = 1.5  # Tukey's fences sit 1.5 interquartile ranges beyond the quartiles
LOW_LABEL_THRESHOLDS = [100, 1_000, 10_000]


@dataclass(frozen=True)
class CommonProfile:
    name: str
    rows: int
    empty: int
    distinct: int
    median_label_when_empty: float
    median_label_when_filled: float
    empty_share_by_land_use: dict[str, float]
    explained_share: float


@dataclass(frozen=True)
class NumericProfile:
    common: CommonProfile
    percentiles: dict[float, float]
    minimum: float
    maximum: float
    mean: float
    skew: float
    log_skew: float
    zeros: int
    negatives: int
    below_fence: int
    above_fence: int
    zero_share_by_land_use: dict[str, float]
    label_rank_correlation: float
    most_similar: tuple[str, float]
    group_medians: pd.DataFrame


@dataclass(frozen=True)
class LabelProfile:
    numeric: NumericProfile
    rows_below: dict[int, int]

    @property
    def common(self) -> CommonProfile:
        return self.numeric.common


@dataclass(frozen=True)
class CategoryProfile:
    common: CommonProfile
    top_categories: pd.DataFrame
    rare_share: float
    containers: list[tuple[str, float, bool]]


@dataclass(frozen=True)
class IdentifierProfile:
    common: CommonProfile
    duplicated: int


@dataclass(frozen=True)
class SharedMeasurements:
    label: pd.Series
    log_label: pd.Series
    land_use_names: pd.Series
    top_land_uses: list[str]
    rank_correlations: pd.DataFrame
    containers: dict[str, list[tuple[str, float, bool]]]
    code_names: dict[str, dict[str, str]]
    leakage_notes: dict[str, str]


def load_training_rows() -> pd.DataFrame:
    text_types = {column: "string[pyarrow]" for column in IDENTIFIER_COLUMNS + CATEGORY_COLUMNS}
    raw = pd.read_csv(RAW_PATH, dtype=text_types, low_memory=False)
    train, _test = split_rows(drop_rows_without_label(raw))
    return train


def measure_shared(train: pd.DataFrame) -> SharedMeasurements:
    names = all_code_names()
    land_use_names = named_codes(normalised_codes(train[LAND_USE]), names.get(LAND_USE, {}))
    sample = train.sample(n=SAMPLE_ROWS, random_state=RANDOM_SEED)
    return SharedMeasurements(
        label=train[LABEL],
        log_label=np.log(train[LABEL]),
        land_use_names=land_use_names,
        top_land_uses=land_use_names.value_counts().index[:TOP_LAND_USES].tolist(),
        rank_correlations=sample[numeric_columns(train)].corr(method="spearman"),
        containers=containment_table(sample),
        code_names=names,
        leakage_notes=sum_rule_notes(train),
    )


def measure(column: str, train: pd.DataFrame, shared: SharedMeasurements):
    values = train[column]
    if column == LABEL:
        return label_profile(values, shared)
    if column in IDENTIFIER_COLUMNS:
        return identifier_profile(column, values, shared)
    if column in CATEGORY_COLUMNS:
        return category_profile(column, values, shared)
    return numeric_profile(column, values, shared)


# ---- shared measurements -------------------------------------------------------------

def numeric_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns
            if column not in IDENTIFIER_COLUMNS and column not in CATEGORY_COLUMNS]


def all_code_names() -> dict[str, dict[str, str]]:
    """{column: {code: name}} for every code sheet in Zillow's data dictionary."""
    tables = {"fips": dict(code_tables.COUNTY_NAMES)}
    with zipfile.ZipFile(code_tables.DICTIONARY_PATH) as workbook:
        strings = code_tables._shared_strings(workbook)
        for sheet, path in code_tables._sheet_paths(workbook).items():
            rows = code_tables._rows(workbook, path, strings)
            tables[sheet.lower()] = {code: name for code, name, *_ in rows[1:] if code}
    return tables


def column_descriptions() -> dict[str, str]:
    with zipfile.ZipFile(code_tables.DICTIONARY_PATH) as workbook:
        strings = code_tables._shared_strings(workbook)
        path = code_tables._sheet_paths(workbook)["Data Dictionary"]
        rows = code_tables._rows(workbook, path, strings)
    return {name.strip().strip("'"): text.strip() for name, text, *_ in rows[1:]}


def normalised_codes(values: pd.Series) -> pd.Series:
    """Removes the '.0' that the CSV adds to whole-number codes, so 261.0 reads as 261."""
    return values.str.replace(r"\.0$", "", regex=True)


def named_codes(codes: pd.Series, names: dict[str, str]) -> pd.Series:
    return codes.map(lambda code: names.get(code, code), na_action="ignore").fillna(EMPTY)


def containment_table(sample: pd.DataFrame) -> dict[str, list[tuple[str, float, bool]]]:
    codes = {column: normalised_codes(sample[column]) for column in CATEGORY_COLUMNS}
    purity = {(inner, outer): nesting_purity(codes[inner], codes[outer])
              for inner in codes for outer in codes if inner != outer}
    return {inner: containers_of(inner, purity, codes) for inner in codes}


def containers_of(inner: str, purity: dict, codes: dict) -> list[tuple[str, float, bool]]:
    found = [(outer, share, purity[(outer, inner)] >= NESTING_PURITY)
             for (candidate, outer), share in purity.items()
             if candidate == inner and share >= NESTING_PURITY]
    return sorted(found, key=lambda item: -codes[item[0]].nunique())


def nesting_purity(inner: pd.Series, outer: pd.Series) -> float:
    """Share of rows whose `outer` value is the most common one within their `inner` group.

    1.0 means every value of `inner` sits inside exactly one value of `outer`, the way a
    zip code sits inside one county. Returns 0.0 when either column has fewer than two
    values, because a constant column "contains" everything and says nothing.
    """
    pairs = pd.DataFrame({"inner": inner, "outer": outer}).dropna()
    pairs = pairs[pairs.groupby("inner")["outer"].transform("size") >= MIN_GROUP_ROWS]
    if pairs["inner"].nunique() < 2 or pairs["outer"].nunique() < 2:
        return 0.0
    largest = pairs.groupby(["inner", "outer"]).size().groupby(level="inner").max()
    return float(largest.sum() / len(pairs))


def sum_rule_notes(train: pd.DataFrame) -> dict[str, str]:
    parts = train[LABEL_PARTS + [LABEL]].dropna()
    gap = (parts[LABEL_PARTS].sum(axis=1) - parts[LABEL]).abs()
    share = float((gap <= SUM_RULE_TOLERANCE).mean())
    note = (f"Structure value + land value equals the total value (within $1) on "
            f"{share:.2%} of the {len(parts):,} rows that have all three.")
    return {column: note for column in LABEL_PARTS + [LABEL]}


# ---- one column ----------------------------------------------------------------------

def common_profile(name: str, values: pd.Series, groups: pd.Series,
                   shared: SharedMeasurements) -> CommonProfile:
    is_empty = values.isna()
    return CommonProfile(
        name=name,
        rows=len(values),
        empty=int(is_empty.sum()),
        distinct=int(values.nunique()),
        median_label_when_empty=float(shared.label[is_empty].median()),
        median_label_when_filled=float(shared.label[~is_empty].median()),
        empty_share_by_land_use=share_by_land_use(is_empty, shared),
        explained_share=explained_share(groups, shared.log_label),
    )


def share_by_land_use(mask: pd.Series, shared: SharedMeasurements) -> dict[str, float]:
    shares = mask.astype(float).groupby(shared.land_use_names).mean()
    return {land_use: float(shares[land_use]) for land_use in shared.top_land_uses}


def explained_share(groups: pd.Series, log_label: pd.Series) -> float:
    """Share of the variation in log value that the column explains on its own.

    Group averages are learned on one random half of the training rows and scored on
    the other half. A column with a unique value per row (an identifier) therefore
    scores zero instead of a perfect, meaningless one.
    """
    is_fit = np.random.default_rng(RANDOM_SEED).random(len(groups)) < 0.5
    averages = trusted_group_means(groups[is_fit], log_label[is_fit])
    fallback = log_label[is_fit].mean()
    predictions = groups[~is_fit].map(averages).astype(float).fillna(fallback)
    return r_squared(log_label[~is_fit], predictions)


def trusted_group_means(groups: pd.Series, values: pd.Series) -> pd.Series:
    stats = values.groupby(groups).agg(["mean", "size"])
    return stats.loc[stats["size"] >= MIN_GROUP_ROWS, "mean"]


def r_squared(actual: pd.Series, predicted: pd.Series) -> float:
    residual = ((actual - predicted) ** 2).sum()
    total = ((actual - actual.mean()) ** 2).sum()
    return float(1 - residual / total)


def numeric_groups(values: pd.Series) -> pd.Series:
    """Equal-count bins, or the exact values when the column has only a few of them."""
    present = values.dropna()
    if present.nunique() <= NUMERIC_BIN_COUNT:
        return values.astype("string").fillna(EMPTY)
    edges = np.unique(present.quantile(np.linspace(0, 1, NUMERIC_BIN_COUNT + 1)))
    return pd.cut(values, edges, include_lowest=True).astype("string").fillna(EMPTY)


def numeric_profile(name: str, values: pd.Series, shared: SharedMeasurements) -> NumericProfile:
    groups = numeric_groups(values)
    present = values.dropna()
    quantiles = present.quantile(PERCENTILES)
    fence = FENCE_WIDTH * (quantiles[0.75] - quantiles[0.25])
    return NumericProfile(
        common=common_profile(name, values, groups, shared),
        percentiles={share: float(value) for share, value in quantiles.items()},
        minimum=float(present.min()),
        maximum=float(present.max()),
        mean=float(present.mean()),
        skew=float(present.skew()),
        log_skew=float(np.log1p(present).skew()) if present.min() >= 0 else float("nan"),
        zeros=int((present == 0).sum()),
        negatives=int((present < 0).sum()),
        below_fence=int((present < quantiles[0.25] - fence).sum()),
        above_fence=int((present > quantiles[0.75] + fence).sum()),
        zero_share_by_land_use=share_by_land_use(values == 0, shared),
        label_rank_correlation=float(shared.rank_correlations.loc[name, LABEL]),
        most_similar=most_similar_column(name, shared.rank_correlations),
        group_medians=group_medians(values, groups, shared.label),
    )


def most_similar_column(name: str, correlations: pd.DataFrame) -> tuple[str, float]:
    others = correlations[name].drop([name, LABEL]).dropna()
    if others.empty:
        return ("none measurable", float("nan"))
    closest = others.abs().idxmax()
    return (closest, float(others[closest]))


def group_medians(values: pd.Series, groups: pd.Series, label: pd.Series) -> pd.DataFrame:
    frame = pd.DataFrame({"group": groups, "value": values, "label": label})
    frame = frame[frame["group"] != EMPTY]
    medians = frame.groupby("group").agg(value=("value", "median"), label=("label", "median"),
                                         rows=("label", "size"))
    return medians[medians["rows"] >= MIN_GROUP_ROWS].sort_values("value")


def label_profile(values: pd.Series, shared: SharedMeasurements) -> LabelProfile:
    return LabelProfile(
        numeric=numeric_profile(LABEL, values, shared),
        rows_below={limit: int((values < limit).sum()) for limit in LOW_LABEL_THRESHOLDS},
    )


def category_profile(name: str, values: pd.Series, shared: SharedMeasurements) -> CategoryProfile:
    codes = normalised_codes(values)
    groups = codes.fillna(EMPTY)
    counts = groups.value_counts()
    return CategoryProfile(
        common=common_profile(name, codes, groups, shared),
        top_categories=top_categories(name, groups, counts, shared),
        rare_share=float(counts[counts < MIN_GROUP_ROWS].sum() / len(groups)),
        containers=shared.containers.get(name, []),
    )


def top_categories(name: str, groups: pd.Series, counts: pd.Series,
                   shared: SharedMeasurements) -> pd.DataFrame:
    top = counts.index[:TOP_CATEGORIES]
    medians = shared.label.groupby(groups).median()
    names = shared.code_names.get(name, {})
    return pd.DataFrame({
        "code": top,
        "name": [names.get(code, "") for code in top],
        "rows": counts[top].to_numpy(),
        "share": (counts[top] / len(groups)).to_numpy(),
        "median_label": medians[top].to_numpy(),
    })


def identifier_profile(name: str, values: pd.Series, shared: SharedMeasurements) -> IdentifierProfile:
    return IdentifierProfile(
        common=common_profile(name, values, values.fillna(EMPTY), shared),
        duplicated=int(values.duplicated().sum()),
    )

"""Figures for the Phase 2 report: missingness, distributions and correlations.

Every figure answers one question and is saved to report/figures/ as a PNG.
Run after src/clean.py, because two of the figures read the processed training set.
"""
# AI was used to help write this code, but the resulting code was reviewed and edited by a human.
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be chosen first)

from clean import (COUNT_FEATURES, FREQUENCY_FEATURES, LABEL, ONE_HOT_FEATURES,  # noqa: E402
                   PROCESSED_DIR, RAW_PATH, SKEWED_AREA_FEATURES,
                   SMALLEST_KEPT_CATEGORY_SHARE, named_categories)

FIGURE_DIR = Path(__file__).resolve().parent.parent / "report" / "figures"
KEPT_COLOUR = "#3b6fd4"
DROPPED_COLOUR = "#d4713b"
SECOND_COLOUR = "#7a51b8"
GRID_COLOUR = "#d9d9d6"
SURFACE = "#fcfcfb"
HIGH_MISSINGNESS_LIMIT = 70  # percent; the line above which a column was dropped
LEAKAGE_LIMIT = 0.85
SAMPLE_ROWS = 400_000  # histograms only; keeps every figure under a few seconds
RANDOM_SEED = 42
UPPER_DISPLAY_QUANTILE = 0.99  # heavy tails would otherwise squeeze every histogram into one bar
COUNT_DISPLAY_QUANTILES = (0.001, 0.999)

KEPT_RAW_FEATURES = COUNT_FEATURES + SKEWED_AREA_FEATURES + ["latitude", "longitude"] \
                    + ONE_HOT_FEATURES + FREQUENCY_FEATURES


def style_axes(axes: plt.Axes) -> None:
    axes.set_facecolor(SURFACE)
    axes.grid(True, color=GRID_COLOUR, linewidth=0.6, alpha=0.8)
    axes.set_axisbelow(True)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(GRID_COLOUR)


def save(figure: plt.Figure, name: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    figure.patch.set_facecolor(SURFACE)
    figure.savefig(FIGURE_DIR / name, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"wrote {name}")


def missingness_figure(raw_sample: pd.DataFrame) -> None:
    """Why 44 of 57 candidate columns were dropped: most of them are mostly empty."""
    missing = (raw_sample.isna().mean() * 100).drop(LABEL).sort_values()
    colours = [KEPT_COLOUR if column in KEPT_RAW_FEATURES else DROPPED_COLOUR for column in missing.index]
    figure, axes = plt.subplots(figsize=(9, 12))
    axes.barh(missing.index, missing.values, color=colours, height=0.72)
    axes.axvline(HIGH_MISSINGNESS_LIMIT, color="#6b6b68", linewidth=1.4, linestyle="--")
    axes.text(HIGH_MISSINGNESS_LIMIT + 1, 0.5, f"{HIGH_MISSINGNESS_LIMIT}% missing",
              color="#6b6b68", fontsize=9)
    axes.set_xlabel("missing values (% of rows)")
    axes.set_title("Missingness per column, kept features in blue, dropped columns in orange", fontsize=11)
    axes.tick_params(labelsize=8)
    style_axes(axes)
    save(figure, "fig1_missingness.png")


def label_distribution_figure(label: pd.Series) -> None:
    """Why the label is heavy-tailed, and what a logarithm does to it."""
    positive = label[label > 0]
    figure, (left, right) = plt.subplots(1, 2, figsize=(11, 4))
    left.hist(positive.clip(upper=positive.quantile(UPPER_DISPLAY_QUANTILE)), bins=80, color=KEPT_COLOUR)
    left.set_title("Assessed value, clipped at the 99th percentile")
    left.set_xlabel("US dollars")
    right.hist(np.log1p(positive), bins=80, color=SECOND_COLOUR)
    right.set_title("Natural logarithm of assessed value")
    right.set_xlabel("log(1 + dollars)")
    for axes in (left, right):
        axes.set_ylabel("properties")
        style_axes(axes)
    figure.suptitle("The label is strongly right-skewed; the logarithm makes it roughly symmetric",
                    fontsize=11)
    save(figure, "fig2_label_distribution.png")


def area_transformation_figure(raw_sample: pd.DataFrame) -> None:
    """Why the two area features are log-transformed before scaling."""
    figure, panels = plt.subplots(2, 2, figsize=(11, 7))
    for row, column in enumerate(SKEWED_AREA_FEATURES):
        values = raw_sample[column].dropna()
        values = values[values > 0]
        panels[row][0].hist(values.clip(upper=values.quantile(UPPER_DISPLAY_QUANTILE)), bins=80,
                            color=DROPPED_COLOUR)
        panels[row][0].set_title(f"{column}: raw (skew {values.skew():.1f})")
        panels[row][1].hist(np.log1p(values), bins=80, color=KEPT_COLOUR)
        panels[row][1].set_title(f"{column}: log (skew {np.log1p(values).skew():.1f})")
        for axes in panels[row]:
            axes.set_ylabel("properties")
            style_axes(axes)
    figure.suptitle("Log transformation of the two area features", fontsize=11)
    figure.tight_layout()
    save(figure, "fig3_area_transformations.png")


def count_distribution_figure(raw_sample: pd.DataFrame) -> None:
    """What the three count features look like, and why they are only median-imputed."""
    figure, panels = plt.subplots(1, 3, figsize=(12, 3.6))
    for axes, column in zip(panels, COUNT_FEATURES):
        values = raw_sample[column].dropna()
        lower, upper = (values.quantile(share) for share in COUNT_DISPLAY_QUANTILES)
        axes.hist(values.clip(lower=lower, upper=upper), bins=40, color=KEPT_COLOUR)
        axes.set_title(f"{column}\nmissing {raw_sample[column].isna().mean() * 100:.2f}%, "
                       f"median {values.median():.0f}", fontsize=10)
        axes.set_ylabel("properties")
        style_axes(axes)
    figure.suptitle("Count features, trimmed to the 0.1st-99.9th percentile for display", fontsize=11)
    figure.tight_layout()
    save(figure, "fig4_count_distributions.png")


def label_correlation_figure(correlations: pd.Series) -> None:
    """Evidence for the professor's 0.85 rule: nothing comes close to the limit."""
    strongest = correlations.reindex(correlations.abs().sort_values(ascending=False).index).head(20)[::-1]
    figure, axes = plt.subplots(figsize=(9, 7))
    axes.barh(strongest.index, strongest.values,
              color=[KEPT_COLOUR if value >= 0 else DROPPED_COLOUR for value in strongest.values], height=0.7)
    axes.axvline(LEAKAGE_LIMIT, color="#6b6b68", linewidth=1.4, linestyle="--")
    axes.text(LEAKAGE_LIMIT, -1.2, "leakage limit 0.85", color="#6b6b68", fontsize=9, ha="center")
    axes.set_xlim(-0.2, 1.0)
    axes.set_xlabel("Pearson correlation with the label")
    axes.set_title("The 20 features most correlated with the label (training rows)", fontsize=11)
    axes.tick_params(labelsize=8)
    style_axes(axes)
    save(figure, "fig5_label_correlation.png")


def cross_correlation_figure(train: pd.DataFrame) -> None:
    """Cross-correlation among the numeric features, to show none of them duplicates another."""
    numeric = COUNT_FEATURES + SKEWED_AREA_FEATURES + ["latitude", "longitude"]
    matrix = train[numeric + [LABEL]].corr()
    figure, axes = plt.subplots(figsize=(7.5, 6.5))
    image = axes.imshow(matrix, cmap="PuOr_r", vmin=-1, vmax=1)
    axes.set_xticks(range(len(matrix)), matrix.columns, rotation=45, ha="right", fontsize=8)
    axes.set_yticks(range(len(matrix)), matrix.columns, fontsize=8)
    for row in range(len(matrix)):
        for column in range(len(matrix)):
            value = matrix.iat[row, column]
            axes.text(column, row, f"{value:.2f}", ha="center", va="center", fontsize=7.5,
                      color="#ffffff" if abs(value) > 0.6 else "#2b2b29")
    axes.set_title("Cross-correlation of the numeric features and the label", fontsize=11)
    figure.colorbar(image, ax=axes, shrink=0.8, label="Pearson correlation")
    save(figure, "fig6_cross_correlation.png")


def category_coverage_figure(raw_sample: pd.DataFrame) -> None:
    """Why rare categories go into an 'Other' bucket instead of their own columns."""
    figure, axes = plt.subplots(figsize=(8, 4.5))
    for column, colour in [("propertycountylandusecode", KEPT_COLOUR),
                           ("buildingqualitytypeid", SECOND_COLOUR)]:
        counts = named_categories(raw_sample[column].astype("string"), column).value_counts()
        coverage = counts.cumsum() / counts.sum() * 100
        axes.plot(range(1, len(coverage) + 1), coverage.values, color=colour, linewidth=2, label=column)
        common = (counts >= len(raw_sample) * SMALLEST_KEPT_CATEGORY_SHARE).sum()
        axes.scatter([common], [coverage.values[common - 1]], color=colour, s=45, zorder=3)
        axes.annotate(f"{common} categories hold 0.01% of rows or more "
                      f"({coverage.values[common - 1]:.1f}% of rows)",
                      (common, coverage.values[common - 1]), textcoords="offset points",
                      xytext=(12, -16), fontsize=9, color=colour)
    axes.set_xscale("log")
    axes.set_xlabel("number of categories kept, ordered by frequency (log scale)")
    axes.set_ylabel("share of rows covered (%)")
    axes.set_title("How few categories cover almost every row", fontsize=11)
    axes.legend(frameon=False, fontsize=9, loc="lower right")
    style_axes(axes)
    save(figure, "fig7_category_coverage.png")


def main() -> None:
    raw_sample = pd.read_csv(RAW_PATH, dtype={"propertycountylandusecode": "string",
                                              "propertyzoningdesc": "string"}).sample(
        SAMPLE_ROWS, random_state=RANDOM_SEED)
    missingness_figure(raw_sample)
    label_distribution_figure(raw_sample[LABEL].dropna())
    area_transformation_figure(raw_sample)
    count_distribution_figure(raw_sample)
    category_coverage_figure(raw_sample)

    train = pd.read_parquet(PROCESSED_DIR / "train.parquet")
    correlations = pd.read_csv(PROCESSED_DIR / "label_correlations.csv", index_col=0).iloc[:, 0]
    label_correlation_figure(correlations)
    cross_correlation_figure(train)


if __name__ == "__main__":
    main()

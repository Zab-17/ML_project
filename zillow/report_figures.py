"""Report figures, drawn from the training rows of a model table: the Zillow Phase III report or the King County
experimental report.

1. distributions.png: the label and the living area, in their raw units and after the natural log, with the skew
   of each. This answers the Phase 2 comments "no distribution analysis" and "log with no analysis".
2. correlation_heatmap.png: the check heatmap with its title strip removed (the report caption replaces it).
The skew numbers also go to distribution_stats.json, so the report quotes the same values the figure shows.

Run: python3 report_figures.py [zillow | king-county]   (default zillow; after the dataset's table is built)
"""
# AI was used to help write this code, but the resulting code was reviewed and edited by a human.

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter
from PIL import Image
from scipy import sparse
from scipy.stats import skew

from shared_code import CHECKS_DIR, PROJECT

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (the backend must be chosen before pyplot loads)

BAR_COLOR = "#2a78d6"
TEXT_COLOR = "#52514e"
BINS = 80
RAW_AXIS_CUT = 0.99             # raw panels stop at the 99th percentile, or the long tail squeezes every bar into one
LOG_AXIS_CUTS = (0.001, 0.999)  # log panels drop the outer 0.1% on each side (Zillow has living areas of 1 sq ft)


@dataclass(frozen=True)
class ReportData:
    table_dir: Path
    label_name: str           # what the label is, in the panel titles
    living_area_column: str   # the log living-area column in columns.csv
    row_name: str             # what one training row is, on the y axis
    heatmap: Path
    heatmap_title_pixels: int  # height of the title strip cut from the top of the heatmap
    figure_dir: Path


KING_COUNTY_OUT = PROJECT / "king_county" / "kc_features" / "out"
DATASETS = {
    "zillow": ReportData(CHECKS_DIR.parent / "model_table", "assessed value", "living area (log)", "training homes",
                         CHECKS_DIR / "chosen_23_features_heatmap.png", 180,
                         PROJECT / "report" / "phase3_report" / "figures"),
    "king-county": ReportData(KING_COUNTY_OUT / "model_table", "sale price", "above-ground living area (log)",
                              "training sales", KING_COUNTY_OUT / "chosen_60_features_heatmap.png", 240,
                              PROJECT / "report" / "experimental_report" / "figures"),
}


def main():
    data = DATASETS[sys.argv[1] if len(sys.argv) > 1 else "zillow"]
    data.figure_dir.mkdir(parents=True, exist_ok=True)
    log_value, log_area = np.load(data.table_dir / "train_label.npy"), training_log_living_area(data)
    stats = {data.label_name: distribution_stats(log_value), "living area": distribution_stats(log_area)}
    draw_distributions(data, log_value, log_area, stats)
    (data.figure_dir / "distribution_stats.json").write_text(json.dumps(stats, indent=2))
    crop_heatmap(data)
    print(json.dumps(stats, indent=2))


def training_log_living_area(data):
    """The log living area of every training row, with the standardization undone."""
    columns = pd.read_csv(data.table_dir / "columns.csv")
    row = columns.index[columns["column"] == data.living_area_column][0]
    standardized = sparse.load_npz(data.table_dir / "train_matrix.npz").tocsc()[:, row].toarray().ravel()
    return standardized * columns.at[row, "training spread"] + columns.at[row, "training average"]


def distribution_stats(log_values):
    raw = np.exp(log_values)
    return {"skew (raw)": float(skew(raw)), "skew (log)": float(skew(log_values)),
            "median (raw)": float(np.median(raw)), "smallest (raw)": float(raw.min()),
            "largest (raw)": float(raw.max()), f"{RAW_AXIS_CUT:.0%} point (raw)": float(np.quantile(raw, RAW_AXIS_CUT))}


def draw_distributions(data, log_value, log_area, stats):
    figure, axes = plt.subplots(2, 2, figsize=(7.0, 4.6))
    panels = [(log_value, data.label_name, "US dollars", millions), (log_area, "living area", "square feet", commas)]
    for (log_values, name, unit, raw_format), (raw_axis, log_axis) in zip(panels, axes):
        raw = np.exp(log_values)
        histogram(raw_axis, raw[raw <= np.quantile(raw, RAW_AXIS_CUT)],
                  f"{name}, raw (skew {stats[name]['skew (raw)']:.1f})", unit, data.row_name)
        raw_axis.xaxis.set_major_formatter(FuncFormatter(raw_format))
        low, high = np.quantile(log_values, LOG_AXIS_CUTS)
        histogram(log_axis, log_values[(log_values >= low) & (log_values <= high)],
                  f"{name}, natural log (skew {stats[name]['skew (log)']:.2f})", f"log of {unit}", data.row_name)
    figure.tight_layout()
    figure.savefig(data.figure_dir / "distributions.png", dpi=200)
    plt.close(figure)


def millions(value, _position):
    return f"${value / 1e6:.1f}M"


def commas(value, _position):
    return f"{value:,.0f}"


def histogram(axis, values, title, unit, row_name):
    axis.hist(values, bins=BINS, color=BAR_COLOR, edgecolor="white", linewidth=0.3)
    axis.set_title(title, fontsize=9, color="black")
    axis.set_xlabel(unit, fontsize=8, color=TEXT_COLOR)
    axis.set_ylabel(row_name, fontsize=8, color=TEXT_COLOR)
    axis.tick_params(labelsize=7, colors=TEXT_COLOR)
    axis.yaxis.set_major_formatter(FuncFormatter(commas))
    axis.grid(axis="y", color="#e6e5e1", linewidth=0.5)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)


def crop_heatmap(data):
    image = Image.open(data.heatmap)
    image.crop((0, data.heatmap_title_pixels, image.width, image.height)).save(
        data.figure_dir / "correlation_heatmap.png")


if __name__ == "__main__":
    main()

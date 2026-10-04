# Machine Learning class project — CSCE 3602 / DSCI 3415

**Predicting the assessed value of a property** from the Zillow properties file:
2,985,217 rows and 58 columns covering Los Angeles, Orange and Ventura counties, California.
Label: `taxvaluedollarcnt`, the total value the county assessor places on the parcel.

Zeyad Khaled and Mohamed Elsayed.

---

## Where everything is

| Path | What it is | Who wrote it |
|---|---|---|
| `changes.md`, `changes.txt` | The feature analysis: every column, kept or dropped, with the reason | Mohamed |
| `test.py` | The scratch script used for that analysis, one column at a time | Mohamed |
| `src/clean.py` | **The cleaning pipeline.** Raw file in, model-ready table out | Zeyad |
| `src/code_tables.py` | Turns Zillow's numeric codes into names, read from Zillow's own data dictionary | Zeyad |
| `src/figures.py` | The seven figures the report needs | Zeyad |
| `data/processed/` | **The cleaned output** — see the next section | produced by `src/clean.py` |
| `report/figures/` | `fig1` to `fig7`, ready to drop into the report | produced by `src/figures.py` |
| `report/phase1_report/` | The Phase 1 report that was rejected, kept for reference | both |
| `REFERENCE.md` | **Every decision explained with its number.** Written to be read, then rewritten in our own words for the report. Not for submission | Zeyad |
| `HANDOFF.md` | Project status, decisions already taken, what is left | Zeyad |
| `archive/egypt-phase1/` | **Ignore this for the current project.** The Egyptian developer tables from the rejected first version | Zeyad |

## The cleaned output, in `data/processed/`

| File | What it holds |
|---|---|
| `train.parquet` | 2,360,761 rows × 48 columns — 47 features plus the label |
| `test.parquet` | 590,190 rows × 48 columns, the same columns in the same order |
| `cleaned_sample_10k.csv` | The first 10,000 training rows as a spreadsheet, for looking at |
| `cleaning_report.json` | Every number the report quotes: rows dropped, values imputed, medians, categories kept |
| `label_correlations.csv` | Every feature ranked by correlation with the label |

Reading the full table needs pandas:

```python
import pandas as pd
train = pd.read_parquet("data/processed/train.parquet")
```

## What the pipeline did

| | Before | After |
|---|---|---|
| Rows | 2,985,217 | **2,950,951** (34,266 dropped for having no label) |
| Columns | 58 | **48** (47 features + 1 label) |
| Raw features used | — | **12**, which become 47 after encoding |
| Missing values | millions | **zero** |

In order: drop rows with no label → split 80/20 **before** fitting anything → convert the
coordinates and take logarithms of the two area features → fill gaps with training medians →
one-hot encode the four small categorical columns with separate `Other` and `Unknown`
columns, and frequency-encode the 234-category county land-use code → z-score every numeric
feature using training statistics → drop anything correlating 0.85 or more with the
label (nothing qualified; the highest is 0.369) and anything that never varies.

`REFERENCE.md` explains why each of those steps exists.

## Running it yourself

The raw file is not in this repository: it is 650 megabytes and its licence forbids
redistribution. Download `properties_2017.csv` into `data/raw/`, then:

```bash
python3 src/clean.py      # about 20 seconds
python3 src/figures.py    # about 13 seconds
```

Needs pandas, numpy, pyarrow and matplotlib. Nothing else.

## Two things to settle before submission

1. **The report itself** is the deliverable and is not written yet.
2. **The licence:** this data may be used *"solely for the purpose and duration of the
   Competition"*, which closed on 2018-01-10. The professor should hear that from us.

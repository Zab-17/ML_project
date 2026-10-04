# Phase 2 reference — cleaning and pre-processing

**This file is a study reference for Zeyad and Mohamed, not a submission.** It explains every
decision, the reason behind it and the number that supports it, so the two of you can write
the report in your own words.

Everything below was produced by `src/clean.py` and `src/figures.py` on the real file,
`data/raw/properties_2017.csv`, 650,093,621 bytes. Both scripts run in under half a minute.
Last run: 4 October 2026.

---

## 1. The picture before the detail

The dataset is one snapshot of **2,985,217 properties** in three Californian counties — Los
Angeles, Orange and Ventura — described by 58 columns. The label we predict is
`taxvaluedollarcnt`, the **total value the county assessor places on the parcel**.

Phase 2 asks for one thing: turn that raw file into a table a model can learn from, and
explain every choice. Mohamed analysed the features column by column and wrote `changes.md`
and `changes.txt`. This half executes his decisions and adds what the professor's four extra
rules require.

Two honest limitations belong at the top of the report, not buried at the end:

- **The label is an assessed value, not a sale price.** The file contains no sale price
  anywhere. An assessor's valuation follows the market but is not the market.
- **The data is licensed for a competition that closed on 2018-01-10.** The rules say
  participants may use it *"solely for the purpose and duration of the Competition"*.

---

## 2. The professor's requirements, and the evidence for each

| Requirement | What we did | Evidence |
|---|---|---|
| **At least 300,000 rows** | 2,950,951 rows kept, nearly ten times the floor | `cleaning_report.json` |
| **Named columns, not bare identifiers** | All 58 column names are English words, and the *values* of three identifier columns were decoded into names from Zillow's own dictionary | `src/code_tables.py`; features read `fips_Los Angeles`, `heatingorsystemtypeid_Central`, `propertylandusetypeid_Single Family Residential` |
| **Rule 1: all scaling finished in Phase 2** | Every numeric feature z-scored, with mean and standard deviation from the training rows only | training mean 0 and standard deviation 1; the test set reads 0.0018, which is itself the proof it was not standardised separately |
| **Rule 2: distributions, correlations and diagrams in the report** | Seven figures in `report/figures/`, described in section 6 | the figure files |
| **Rule 3: text features one-hot encoded or embedded, never dropped unjustified** | The dataset has five non-numeric columns. One is frequency encoded with the professor's approval; four are dropped and each needs its written justification (section 5) | measured on the raw file |
| **Rule 4: drop any feature correlating 0.85 or more with the label** | Computed for every encoded column on the training rows. Nothing reaches the limit; the highest is **0.369** | `label_correlations.csv`, figure 5 |

---

## 3. The pipeline, step by step, with the reason for each step

### Step 1 — read only the columns we need
Twelve features plus the label, out of 58. **Why:** the file is 650 megabytes; reading a
fifth of it keeps the whole run near twenty seconds.

### Step 2 — drop rows with no label
**34,266 rows, 1.15%.** **Why:** a row with no label can neither train a model nor test one,
and imputing a label would mean inventing the answer and then grading ourselves on it.
**2,950,951 rows remain.**

### Step 3 — split into training and test rows *before* anything is measured
80% training (2,360,761 rows) and 20% test (590,190 rows), shuffled with seed 42 so the split
is identical on every run.

**Why this order matters, and it is the single most important idea in the pipeline:** a
median, a category list, a frequency, a mean and a standard deviation are all *learned from
data*. Computed over the whole dataset and then applied, they carry a trace of the test rows
into training, the test score is flattered, and the model looks better than it is. That is
data leakage. Mohamed's specification demanded it for the frequency encoding; the same rule
applies to every learned quantity, so the split happens first and `fit()` only ever sees
training rows.

### Step 4 — convert units and tame the skew
- **Latitude and longitude divided by 1,000,000.** Zillow stores them multiplied by a
  million, so `34144442` is really 34.144442 degrees.
- **Natural logarithm of finished square feet and lot size.** Both are extremely
  right-skewed. Figure 3 shows skew falling from 7.2 to −1.4 for finished area and from 23.3
  to 0.2 for lot size. **Why it matters:** models minimise squared error, so with a raw heavy
  tail a handful of mansions contribute as much error as hundreds of ordinary houses.
  `log1p` is used rather than `log` because it is defined at zero.

### Step 5 — fill the gaps that remain
Median imputation, with the median taken from the training rows: bathrooms 2, bedrooms 3,
year built 1963, plus the medians of the two logged areas and the two coordinates.

Values filled: lot size **238,965**, year built **43,671**, finished area **40,306**,
bathrooms **25**, bedrooms **13**, coordinates **0**.

**Why the median and not the mean:** the mean of a skewed column is dragged up by its tail,
so filling with it would quietly tell the model that every unknown property is larger than
typical. **Say in the report that 8% of the lot-size column is our guess, not Zillow's data.**

### Step 6 — encode the categories, two different ways

**One-hot, for the four columns with few categories.** Each category becomes its own 0/1
column, with an **Other** column (value present but rare) and an **Unknown** column (value
missing) kept separate. Those are different facts and a model can use each differently. *(An
early version merged them, which hid a third of a column inside "Other". The bug was caught
by checking that each Unknown column's share equals the column's measured missingness, and
it now does: 34.23% against 34.22% for building quality, 36.65% against 36.67% for heating.)*

The rarity rule: a category is kept if it holds at least **0.01% of training rows**, about
236 properties. Below that, a one-hot column is almost all zeros and its weight is estimated
from a handful of properties, which is noise with a name.

**Frequency encoding, for `propertycountylandusecode`.** This column has **234 codes**, so
one-hot would add 234 mostly empty columns. Instead each code becomes the share of training
rows holding it: code `0100` covers 39% of rows and becomes 0.391. A code never seen in
training, and a missing code, both become 0, because no training row supports them. 220 codes
appear in the training rows.

- **The cost, which belongs in the report:** frequency encoding says two categories are
  similar whenever they are equally common, even if they mean unrelated things.
- **Why we use it anyway:** it is Mohamed's original choice and the professor approved it.

### Step 7 — standardise the numeric features
Each numeric value becomes the **z-score**, `(value − training mean) ÷ training standard
deviation`, the standard formula from statistics and what `StandardScaler` computes.

**Why:** raw features live on incompatible scales — bathrooms around 2, longitude around
−118. Any model that measures distance (k-nearest neighbours), penalises coefficient size
(ridge regression) or descends a gradient would otherwise be dominated by the biggest
numbers. Worked example: bathrooms have training mean 2.24 and standard deviation 1.06, so
3 bathrooms becomes +0.72 and 1 bathroom becomes −1.17.

The frequency column is standardised too, because it is a number. The one-hot columns stay
0 and 1, since they are already comparable.

### Step 8 — check for leakage, and for columns that say nothing
**Leakage:** correlation of every encoded column with the label, on the training rows.
**Nothing was dropped, because nothing came close** — the strongest is finished square feet
at 0.369, then bathrooms at 0.312. The three genuinely leaking columns were already removed
in Mohamed's analysis: `structuretaxvaluedollarcnt` and `landtaxvaluedollarcnt` **sum exactly
to the label**, and `taxamount` correlates 0.979 because the tax is computed from the value.

**No variation:** four one-hot columns were entirely zero, such as `fips_Unknown`, since no
property is missing its county. A column that never changes teaches nothing and breaks
anything that divides by variance.

### Step 9 — write the outputs
`data/processed/train.parquet` (2,360,761 × 48), `test.parquet` (590,190 × 48),
`cleaning_report.json` and `label_correlations.csv`. Parquet is used because it stores column
types, so the one-hot columns stay single bytes instead of becoming text.

---

## 4. The result in numbers

| | Before | After |
|---|---|---|
| Rows | 2,985,217 | **2,950,951** |
| Columns | 58 (1 label + 57 candidates) | **48 (47 features + 1 label)** |
| Raw features used | — | **12** |
| Raw columns dropped | — | **45** |
| Missing values | millions | **zero** |

The 47 features are:
- **8 numeric, z-scored:** bathrooms, bedrooms, year built, log finished area, log lot size,
  latitude, longitude, and the frequency-encoded county land-use code
- **39 one-hot columns:** property land-use type 14, building quality 13, heating system 9,
  county 3

---

## 5. Every column, and why it was kept or dropped

### Kept (12)

| Feature | Type | Why | Treatment |
|---|---|---|---|
| `bathroomcnt` | count | 0.1% missing, correlation 0.312 with the label | median fill, z-score |
| `bedroomcnt` | count | 0.1% missing, correlation 0.141 | median fill, z-score |
| `yearbuilt` | number | 1.6% missing; age is meaningful and ordered | median fill, z-score |
| `calculatedfinishedsquarefeet` | number | 1.5% missing, the strongest single feature at 0.369 | logarithm, median fill, z-score |
| `lotsizesquarefeet` | number | land size, separate from built area | logarithm, median fill, z-score |
| `latitude`, `longitude` | number | location finer than the county | divide by one million, median fill, z-score |
| `buildingqualitytypeid` | category | quality classes show clearly different value distributions | one-hot, 12 categories + Unknown |
| `fips` | category | the three counties differ | one-hot, decoded to county names |
| `heatingorsystemtypeid` | category | heating classes differ in value | one-hot, decoded to names, 7 kept |
| `propertylandusetypeid` | category | a house, a condominium and a duplex are different products | one-hot, decoded to names, 13 kept |
| `propertycountylandusecode` | **text** | finer land use than the previous column | **frequency encoded**, 220 codes seen in training |

### Dropped (45), in four groups

1. **Leakage, 3 columns:** `structuretaxvaluedollarcnt`, `landtaxvaluedollarcnt`, `taxamount`.
2. **Too empty to use, about 24 columns:** `basementsqft` (99.95% missing), every pool column
   (~99%), `fireplacecnt` (89.5%), the garage columns (70%), `storytypeid`,
   `typeconstructiontypeid`, `unitcnt`, `numberofstories`, the yard-area columns,
   `taxdelinquencyflag` and `taxdelinquencyyear` (98.1%), `hashottuborspa` (98.3%),
   `fireplaceflag` (99.8%), `airconditioningtypeid`, `architecturalstyletypeid` and
   `buildingclasstypeid` (effectively 100% empty). **The deeper reason is not the percentage
   but the ambiguity:** a blank in `fireplacecnt` may mean no fireplace or an unrecorded one.
3. **Says the same thing as a kept column, about 15 columns:** `calculatedbathnbr` (agrees
   with `bathroomcnt` in 100% of the 2,868,061 rows where both exist), `fullbathcnt`, the six
   other finished-area measurements, the five region identifiers whose location is already
   carried by county plus coordinates — and whose codes have **no published lookup table** —
   and `propertyzoningdesc` (see below).
4. **No usable variation, 3 columns:** `parcelid`, `assessmentyear` (2016 in 99.9% of rows),
   `roomcnt` (77.6% of properties record zero rooms, which is plainly wrong).

### The five text columns, treated one by one (rule 3)

| Column | Missing | Distinct | What we did, and the justification the report must carry |
|---|---|---|---|
| `propertycountylandusecode` | 0.10% | 234 | **Frequency encoded**, as Mohamed planned and the professor approved. 234 one-hot columns would be mostly empty. |
| `propertyzoningdesc` | 33.59% | 5,651 | **Dropped.** A third of the rows have no code, and the location information it carries is already represented by latitude and longitude. |
| `taxdelinquencyflag` | 98.11% | 1 (`"Y"`) | **Dropped.** Only one value is ever recorded and 98% of rows are blank, so there is no second category for an encoding to contrast against. |
| `fireplaceflag` | 99.83% | 1 (`True`) | **Dropped**, same reasoning. |
| `hashottuborspa` | 98.32% | 1 (`True`) | **Dropped**, same reasoning. |

**Two things the report cannot omit**, because rule 3 penalises dropping text without
justification: that the professor approved frequency encoding, and the reason above for each
dropped text column. Also state why word2vec and doc2vec were not used: those methods learn
meaning from the words surrounding a word, and these are administrative codes such as `LAR1`
with no surrounding text, so there would be no context to learn from.

### The identifier problem, which the professor raised directly
Of the 58 columns, 21 hold numeric codes and only **7 have a decode table** in Zillow's
dictionary. We decoded the three that matter (county, heating system, land-use type). One
kept column cannot be decoded: **`buildingqualitytypeid`** has no published table, which is
exactly why it is treated as an unordered category — we must not assume a higher number means
better quality.

`propertycountylandusecode` is also undecodable, and it is three local vocabularies in one
column: measured, **233 of its 234 codes appear in exactly one county** (186 Los Angeles, 19
Orange, 30 Ventura). It is text that looks numeric — `0100`, `010C`, `122`, `1` — so reading
it as a number would turn `0100` into `100` and collide two counties' codes. It is read as
text throughout.

---

## 6. The seven figures and what each one is for

| File | What it shows | The sentence it supports |
|---|---|---|
| `fig1_missingness.png` | Missingness of all 57 candidate columns, kept features in blue and dropped ones in orange, with the 70% line | "Most of the dataset is empty: the dropped columns are chosen by evidence, not taste." |
| `fig2_label_distribution.png` | The label, raw and logged | "Assessed value is strongly right-skewed, so error should be judged in relative terms." |
| `fig3_area_transformations.png` | The two area features before and after the logarithm, with the skew printed | "The logarithm moves skew from 7.2 to −1.4 and from 23.3 to 0.2." |
| `fig4_count_distributions.png` | Bathrooms, bedrooms and year built, with missingness and median | "These three are nearly complete, so a median fill touches almost nothing." |
| `fig5_label_correlation.png` | The strongest correlations with the label, against the 0.85 line | "Rule 4 is satisfied with room to spare: the strongest feature is 0.369." |
| `fig6_cross_correlation.png` | Correlation among the numeric features and the label | "Bathrooms and finished area correlate 0.69, so they share information but neither is redundant." |
| `fig7_category_coverage.png` | How many categories are needed to cover the rows | "A handful of categories cover almost every row, which is why rare ones are pooled." |

Figures 1 to 4 and 7 use a random sample of 400,000 rows for speed; figures 5 and 6 use the
full training set. Say so in a caption — presenting a sampled figure as the whole dataset is
a small dishonesty a careful reader will catch.

---

## 7. Where this differs from Mohamed's analysis

The feature list now matches his `changes.md` exactly: his twelve features, no more and no
less. Two things were added where his specification was silent:

| | What was added | Why |
|---|---|---|
| Scaling | z-scores on all numeric features, fitted on training rows | rule 1 |
| Rare categories | keep categories holding at least 0.01% of rows, pool the rest into Other | otherwise the sparse tail becomes near-empty columns |

One contradiction inside his own files had to be resolved: `changes.txt` keeps `fullbathcnt`
while `changes.md` drops it. We followed `changes.md`, whose counts add up to 58.

---

## 8. How to reproduce everything

```bash
cd ~/Desktop/ML_project
python3 src/clean.py        # about 20 seconds, writes data/processed/
python3 src/figures.py      # about 13 seconds, writes report/figures/
```

`data/raw/properties_2017.csv` is not in the repository: it is 650 megabytes and the licence
forbids redistribution. Download it into `data/raw/` first.

## 9. What is still open

1. The report itself, which the two of you write from this document.
2. Telling the professor about the competition licence.

# Phase 2 reference — cleaning and pre-processing

**This file is a study reference for Zeyad and Mohamed, not a submission.** It explains
every decision, the reason behind it and the number that supports it, so the two of you
can write the report in your own words. Nothing here is written to be handed in.

Everything was produced by `src/clean.py` and `src/figures.py` on the real file,
`data/raw/properties_2017.csv`, 650,093,621 bytes. Both scripts run in under half a minute.

---

## 1. The picture before the detail

The dataset is one snapshot of **2,985,217 properties** in three Californian counties —
Los Angeles, Orange and Ventura — described by 58 columns. The label we predict is
`taxvaluedollarcnt`, the **total value the county assessor places on the parcel**.

Phase 2 asks for one thing: turn that raw file into a table a model can learn from, and
explain every choice. The work splits in two. Mohamed analysed the features, column by
column, and wrote `changes.md` and `changes.txt`. This half takes his decisions, applies
the professor's four extra rules, and executes them.

Two honest limitations belong at the top of the report, not buried at the end:

- **The label is an assessed value, not a sale price.** The file contains no sale price
  anywhere. An assessor's valuation follows the market but is not the market. Any claim
  about "predicting house prices" should be worded as predicting assessed value.
- **The data is licensed for a competition that closed in January 2018.** The rules say
  participants may use it *"solely for the purpose and duration of the Competition"*.
  Raise this with the professor rather than let him find it.

---

## 2. The professor's requirements, and the evidence for each

| Requirement | What we did | Evidence |
|---|---|---|
| **At least 300,000 rows** | Kept 2,950,951 rows, nearly ten times the floor | `cleaning_report.json` |
| **Named columns, not just identifiers** | All 58 column names are English words. We went further and decoded the *values* of three identifier columns into names, from Zillow's own dictionary | `src/code_tables.py`; features now read `fips_Los Angeles`, `heatingorsystemtypeid_Central`, `propertylandusetypeid_Single Family Residential` |
| **Rule 1: all scaling finished in Phase 2** | Every numeric feature is z-scored. Mean and standard deviation come from the training rows only | `standardised_numeric()`; verified mean 0 and standard deviation 1 on the training set |
| **Rule 2: distributions, correlations and diagrams in the report** | Seven figures in `report/figures/`, described in section 6 | the figure files |
| **Rule 3: text features one-hot encoded or embedded, never dropped unjustified** | The dataset has exactly five non-numeric columns. None is dropped silently; see section 5 | measured with `dtype == object` on the raw file |
| **Rule 4: drop any feature correlating 0.85 or more with the label** | Computed for all 205 encoded columns on the training rows. Nothing reaches the limit; the highest is 0.369 | `label_correlations.csv`, figure 5 |

---

## 3. The pipeline, step by step, with the reason for each step

### Step 1 — read only the columns we need
Thirteen features plus the label, out of 58. **Why:** the file is 650 megabytes; reading a
fifth of it keeps the whole run near twenty seconds and leaves room in memory for the
encoding step.

### Step 2 — drop rows with no label
**34,266 rows, 1.15%.** **Why:** a row with no label cannot train a model and cannot test
one. Imputing a label would mean inventing the answer and then scoring ourselves on it.
**2,950,951 rows remain.**

### Step 3 — split into training and test rows *before* anything is measured
80% training (2,360,761 rows) and 20% test (590,190 rows), shuffled with a fixed seed of 42
so the split is identical on every run.

**Why this order matters, and it is the single most important idea in the whole pipeline:**
a median, a category list, a mean and a standard deviation are all *learned from data*. If
they are computed over the whole dataset and then applied, the training rows carry a trace
of the test rows, the test score is flattered, and the model looks better than it is. This
is called data leakage. Mohamed's specification already demanded it for the frequency
encoding; the same rule applies to every learned quantity, so the split happens first and
`fit()` only ever sees training rows.

### Step 4 — convert units and tame the skew
- **Latitude and longitude divided by 1,000,000.** Zillow stores them multiplied by one
  million, so `34144442` is really 34.144442 degrees. Left as they were, the model would
  treat them as enormous numbers and the scaling step would be meaningless.
- **Natural logarithm of finished square feet and lot size.** Both are extremely
  right-skewed: most homes are modest and a few are vast. Figure 3 shows the skew falling
  from 7.2 to −1.4 for finished area and from 23.3 to 0.2 for lot size. **Why it matters:**
  with a raw heavy tail, a handful of mansions dominate the squared error and a linear
  model spends its capacity on them. `log1p` is used rather than `log` because it is defined
  at zero.

### Step 5 — fill the gaps that remain
Median imputation, with the median taken from the training rows:
`bathroomcnt` 2, `bedroomcnt` 3, `yearbuilt` 1963, and the medians of the logged areas and
of the coordinates. Counts filled: year built 43,671, finished area 40,306, lot size 238,965,
bathrooms 25, bedrooms 13, coordinates 0.
**Why the median and not the mean:** the mean of a skewed column is dragged by the tail, so
filling with it would push thousands of ordinary homes towards the luxury end.

### Step 6 — encode the categories
Every categorical and text column becomes one-hot columns, each with its own **Other** and
**Unknown** column.
- **Unknown** means the value was missing.
- **Other** means the value was present but too rare to keep.

Keeping those two apart is deliberate: "we do not know the heating system" and "the heating
system is unusual" are different facts, and a model can use each differently. *(An early
version of the script merged them, which hid 32% of the zoning column inside "Other". The
bug was found by checking that each Unknown column's share equals the column's measured
missingness, and it now does: 34.23% against 34.22% for building quality, 36.65% against
36.67% for heating, 32.85% against 32.84% for zoning.)*

**The rarity rule:** a category is kept if it holds at least 0.01% of the training rows,
which is about 236 rows. **Why:** below that, a one-hot column is almost all zeros and its
coefficient is estimated from a handful of properties, which is noise with a name. Figure 7
shows what the rule buys: for the county land-use code, **51 categories cover 99.8% of all
rows**, while the remaining 183 would add 183 near-empty columns.

### Step 7 — standardise the numeric features
Each numeric feature becomes `(value − training mean) ÷ training standard deviation`.
**Why:** the raw features live on wildly different scales — a bathroom count of 2 beside a
longitude of −118.17. Any model that measures distance or penalises coefficient size (the
k-nearest-neighbours method, ridge regression, support vector machines, gradient descent)
would otherwise let the biggest-numbered column dominate. The one-hot columns stay as 0 and
1, because they are already on a common scale and scaling them would destroy their meaning.

### Step 8 — check for leakage and drop anything at 0.85 or above
Correlation of every one of the 205 encoded columns with the label, computed on the training rows.
**Nothing was dropped, because nothing came close:** the strongest is finished square feet at
0.369, then bathrooms at 0.312. The three genuinely leaking columns were already removed in
Mohamed's analysis: `structuretaxvaluedollarcnt` and `landtaxvaluedollarcnt` **sum exactly to
the label**, and `taxamount` correlates 0.979 with it because the tax is computed from the
assessed value.

### Step 9 — write the outputs
Four one-hot columns are also dropped here because they never vary: a category that appears
in no training row (`fips_Unknown`, since every row has a county) teaches a model nothing and
breaks anything that divides by variance.

`data/processed/train.parquet` (2,360,761 × 202), `test.parquet` (590,190 × 202),
`cleaning_report.json` and `label_correlations.csv`. The parquet format is used because it
stores column types, so the 198 one-hot columns stay as single bytes instead of becoming
text, and the training file is 59 megabytes rather than several hundred.

---

## 4. The result in numbers

| | Before | After |
|---|---|---|
| Rows | 2,985,217 | **2,950,951** |
| Columns | 58 (1 label + 57 candidates) | **202 (201 features + 1 label)** |
| Raw features used | — | **13** |
| Raw columns dropped | — | **44** |
| Missing values | millions | **zero** |

The 13 features become 201 columns: 7 numeric, and 194 one-hot columns — zoning 102,
county land-use code 53, land-use type 14, building quality 13, heating system 9, county 3.
(Six more were built and then dropped: two that merged into a category nobody uses, and four
that never vary.)

---

## 5. Every column, and why it was kept or dropped

### Kept (13)

| Feature | Type | Why | Treatment |
|---|---|---|---|
| `bathroomcnt` | count | 0.1% missing, correlation 0.312 with the label | median fill, z-score |
| `bedroomcnt` | count | 0.1% missing, correlation 0.141 | median fill, z-score |
| `yearbuilt` | number | 1.6% missing; age is meaningful and ordered | median fill, z-score |
| `calculatedfinishedsquarefeet` | number | 1.5% missing, the strongest single feature at 0.369 | logarithm, median fill, z-score |
| `lotsizesquarefeet` | number | land size, separate from built area | logarithm, median fill, z-score |
| `latitude`, `longitude` | number | location finer than the county | divide by one million, median fill, z-score |
| `buildingqualitytypeid` | category | the quality classes show clearly different value distributions | one-hot, 12 categories + Other + Unknown |
| `fips` | category | the three counties differ | one-hot, decoded to county names |
| `heatingorsystemtypeid` | category | heating classes differ in value | one-hot, decoded to names, 7 kept |
| `propertylandusetypeid` | category | a house, a condominium and a duplex are different products | one-hot, decoded to names, 13 kept |
| `propertycountylandusecode` | **text** | finer land use than the previous column | one-hot, 51 categories covering 99.8% |
| `propertyzoningdesc` | **text** | the only free-text column; rule 3 | one-hot, top 100 codes + Other + Unknown |

### Dropped (44), in four groups

1. **Leakage, 3 columns:** `structuretaxvaluedollarcnt`, `landtaxvaluedollarcnt`, `taxamount`.
   The first two add up to the label; the third is computed from it.
2. **Too empty to use, about 24 columns:** `basementsqft` (99.95% missing), every pool
   column (~99%), `fireplacecnt` (89.5%), the garage columns (70%), `storytypeid`,
   `typeconstructiontypeid`, `unitcnt`, `numberofstories`, the yard-area columns,
   `taxdelinquencyflag` and `taxdelinquencyyear` (98.1%), `hashottuborspa` (98.3%),
   `fireplaceflag` (99.8%), `airconditioningtypeid`, `architecturalstyletypeid` and
   `buildingclasstypeid` (effectively 100% empty). **The deeper reason is not the
   percentage but the ambiguity:** a blank in `fireplacecnt` may mean no fireplace or an
   unrecorded one, and nothing in the data distinguishes them.
3. **Says the same thing as a kept column, about 14 columns:** `calculatedbathnbr` (agrees
   with `bathroomcnt` in 100% of the 2,868,061 rows where both exist), `fullbathcnt`, the
   six other finished-area measurements, and the five region identifiers
   (`regionidcity`, `regionidzip`, `regionidneighborhood`, `regionidcounty`,
   `rawcensustractandblock`) whose location information is already carried by county plus
   coordinates — and whose codes have **no published lookup table**, so they cannot even
   be named.
4. **No usable variation, 3 columns:** `parcelid` (a unique identifier, nothing to learn),
   `assessmentyear` (2016 in 99.9% of rows), `roomcnt` (77.6% of properties record zero
   rooms, which is plainly wrong).

### The five text columns, treated one by one (rule 3)

| Column | Missing | Distinct | What we did and why |
|---|---|---|---|
| `propertycountylandusecode` | 0.10% | 234 | **One-hot.** Mohamed's plan was frequency encoding, but rule 3 permits only one-hot or an embedding. |
| `propertyzoningdesc` | 33.59% | 5,651 | **Kept and one-hot encoded.** Mohamed dropped it for missingness. We argue for keeping it, because rule 3 singles out text features, and because he kept `heatingorsystemtypeid` at 36.7% missing, which is emptier. A word2vec or doc2vec embedding is the wrong instrument here: these are administrative zoning codes such as `LAR1` and `SCUR2`, not sentences, so there is no word context for an embedding to learn. |
| `taxdelinquencyflag` | 98.11% | 1 (`"Y"`) | **Dropped**, matching Mohamed. Arguably it is a presence marker where a blank means "not delinquent", but Zeyad chose to follow the 98% missingness line. |
| `fireplaceflag` | 99.83% | 1 (`True`) | **Dropped**, same reasoning. |
| `hashottuborspa` | 98.32% | 1 (`True`) | **Dropped**, same reasoning. |

### The identifier problem, which the professor raised directly
Of the 58 columns, 21 hold numeric codes and only **7 have a decode table** in Zillow's
dictionary. We decoded the three that matter to us (county, heating system, land-use type).
Two kept columns cannot be decoded and the report should say so plainly:
- **`buildingqualitytypeid`** has no published table. This is exactly why it is treated as
  an unordered category: we must not assume a higher number means better quality.
- **`propertycountylandusecode`** is each county assessor's own code. Measured: **233 of the
  234 codes appear in exactly one county** (186 Los Angeles, 19 Orange, 30 Ventura), so the
  column is really three local vocabularies stacked together. It is also text that looks
  numeric — `0100`, `010C`, `122`, `1` — so reading it as a number would turn `0100` into
  `100` and collide two different counties' codes. It is read as text throughout.

---

## 6. The seven figures and what each one is for

| File | What it shows | The sentence it supports in the report |
|---|---|---|
| `fig1_missingness.png` | Missingness of all 57 candidate columns, kept features in blue and dropped ones in orange, with the 70% line | "Most of the dataset is empty: the columns we drop are not chosen by taste but by how much data they contain." |
| `fig2_label_distribution.png` | The label, raw and logged | "Assessed value is strongly right-skewed, which is why error should be judged in relative rather than absolute terms." |
| `fig3_area_transformations.png` | The two area features before and after the logarithm, with the skew printed on each panel | "The logarithm is not decoration: it moves skew from 7.2 to −1.4 and from 23.3 to 0.2." |
| `fig4_count_distributions.png` | Bathrooms, bedrooms and year built, with missingness and median | "These three are nearly complete, so a median fill touches almost nothing." |
| `fig5_label_correlation.png` | The 20 strongest correlations with the label, against the 0.85 line | "Rule 4 is satisfied with room to spare: the strongest feature is 0.369." |
| `fig6_cross_correlation.png` | Correlation among the seven numeric features and the label | "Bathrooms and finished area correlate 0.69 with each other, so they share information but neither is redundant." |
| `fig7_category_coverage.png` | How many categories are needed to cover the rows, for the two text features | "51 of 234 county codes cover 99.8% of rows; the other 183 would be near-empty columns." |

Figures 1 to 4 and 7 use a random sample of 400,000 rows for speed; figures 5 and 6 use the
full training set. Say so in a caption — a sampled figure that is presented as the whole
dataset is a small dishonesty that a careful reader will catch.

---

## 7. Where this deviates from Mohamed's analysis

| | His specification | What the code does | The reason |
|---|---|---|---|
| `propertyzoningdesc` | dropped | kept, one-hot | rule 3 — **settled on 3 October: Zeyad put the argument to Mohamed and the two agreed to keep the column** |
| `propertycountylandusecode` | frequency encoding | one-hot | rule 3 permits only one-hot or embedding |
| scaling | not mentioned | z-scores on all numeric features | rule 1 |
| rare categories | not mentioned | 0.01% threshold, rest into Other | otherwise 183 near-empty columns |
| `fullbathcnt` | `changes.txt` keeps it, `changes.md` drops it | dropped | followed `changes.md`, whose counts add to 58 |
| identifier columns | heating to be mapped to names | all three decodable columns mapped | his own instruction, plus the professor's point about identifiers |

Write these up as coordination, not contradiction: his analysis predates the professor's
four rules, and each change names the rule that forced it.

---

## 8. How to reproduce everything

```bash
cd ~/Desktop/ML_project
python3 src/clean.py        # about 20 seconds, writes data/processed/
python3 src/figures.py      # about 13 seconds, writes report/figures/
```

`data/raw/properties_2017.csv` is not in the repository, because it is 650 megabytes and the
licence forbids redistribution. Download it yourself and put it in `data/raw/`.

## 9. What is still open

1. The report itself, which the two of you write from this document.
2. Telling the professor about the competition licence.

Settled on 3 October: `propertyzoningdesc` is kept, with its missing third held in an explicit
`Unknown` column rather than filled in. Inventing a zoning code for a property that has none
would be fabricating data; an `Unknown` column is honest and still usable by a model.

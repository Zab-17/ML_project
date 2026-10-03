# Handoff — Machine Learning class project (CSCE 3602 / DSCI 3415)

Last updated 2026-10-03. Read this first when a session opens in `~/Desktop/ML_project`.
Zeyad Khaled, partner **Mohamed Elsayed**. Zeyad is the only decision maker: report
findings, propose, then wait for his explicit go-ahead before any action.

---

## 1. What the project is now

**Predicting the assessed value of a property** (`taxvaluedollarcnt`) from the Zillow
properties file: 2,985,217 rows, 58 named columns, covering Los Angeles, Orange and Ventura
counties in California, as of 2016 and 2017.

The project moved to this dataset after the professor rejected Phase 1. The two reasons he
gave, in his own words, were that *"your chosen dataset's size is very small … You need to
find a larger dataset or change the regional scoping"* and that the report was *"not
discussing the datasets in depth or the limitations of each, but rather iterating the phases
that I already shared on the course outlines"*. He then set a floor of **300,000 rows** with
**real named columns rather than bare identifiers**.

**The earlier Egyptian project is parked, not abandoned.** Four research agents established
with measured evidence that Egypt cannot meet the floor: no Egyptian dataset exceeds 64,106
usable rows, the entire live inventory of Egypt's largest portal is about 279,500 listings,
Egypt's national open-data portal does not resolve in the domain name system, and no public
transaction register exists. That work — 225 developers, 1,090 projects, a compound lookup —
is preserved in `data/developers/` and would matter again only if the project returns to
Egypt or becomes a product.

### Two limitations that must appear in the report
- **The label is an assessed value, not a sale price.** The file contains no sale price at
  all. The competition's own target was `logerror = log(Zestimate) − log(SalePrice)`, and
  both of those quantities were withheld.
- **The licence permits use only during a competition that closed on 2018-01-10.**
  Quoted from the rules: participants may use the data *"solely for the purpose and duration
  of the Competition"*. **Not yet raised with the professor. It should be.**

---

## 2. Phase 2 (due 2026-10-04): where it stands

Phase 2 asks for a report on the chosen dataset, the cleaning and pre-processing, and the
generated features with the reasoning for each. The professor added four rules:
scaling finished in this phase; the analysis with diagrams in the report; text features
one-hot encoded or embedded, never dropped unjustified; and any feature correlating 0.85 or
more with the label dropped.

| Part | Owner | State |
|---|---|---|
| Feature analysis (`changes.md`, `changes.txt`) | Mohamed | written as prose, **no diagrams of his own** |
| Cleaning and pre-processing pipeline | Zeyad, built here | **done and verified** |
| Figures | built here | **seven figures in `report/figures/`** |
| **The report document** | both | **not written — this is the deliverable** |

**Result of the pipeline:** 2,950,951 rows kept, 13 raw features becoming **201 encoded
columns**, no missing values anywhere, numeric features standardised on training statistics
only, nothing correlating at or above 0.85 with the label (the highest is 0.369).

**`REFERENCE.md` is the study document for writing the report.** It explains every decision,
the reason behind it and the supporting number, and it is explicitly not for submission.

---

## 3. What is in this repository

```
src/clean.py              the pipeline; about 20 seconds on the full file
src/code_tables.py        decodes county, heating system and land-use type from Zillow's
                          own data dictionary, so features read as names not codes
src/figures.py            the seven figures; about 13 seconds
REFERENCE.md              every decision explained, for writing the report
changes.md, changes.txt   Mohamed's feature analysis
test.py                   Mohamed's scratch analysis script
report/figures/           fig1 missingness … fig7 category coverage
report/Phase1_Report*/    the rejected Phase 1 report and its Overleaf copies
data/developers/          the parked Egyptian developer tables and their build scripts
```

`data/raw/` and `data/processed/` are not in the repository: the raw file is 650 megabytes
and the licence forbids redistribution. Download `properties_2017.csv` yourself, put it in
`data/raw/`, then run `python3 src/clean.py` followed by `python3 src/figures.py`.

Git identity for every commit: `Zab-17 <128490152+Zab-17@users.noreply.github.com>`, never
any other address. Remote: `https://github.com/Zab-17/ML_project.git`.

---

## 4. Decisions already taken (do not re-litigate)

- **Label:** `taxvaluedollarcnt`. The three columns that leak it are removed:
  `structuretaxvaluedollarcnt` and `landtaxvaluedollarcnt` sum exactly to the label, and
  `taxamount` correlates 0.979 with it.
- **Rows with no label are dropped**, 34,266 of them, rather than having a label invented.
- **The split happens before anything is fitted**, 80/20 with seed 42, so medians, category
  lists, means and standard deviations are learned from training rows only.
- **`propertyzoningdesc` is kept** (3 October, agreed with Mohamed), with its missing third
  in an explicit `Unknown` column rather than filled in.
- **The three columns that are 98% or more missing are dropped** — `taxdelinquencyflag`,
  `fireplaceflag`, `hashottuborspa` — matching Mohamed's analysis.
- **`propertycountylandusecode` is one-hot encoded**, not frequency encoded, because the
  professor's rule allows only one-hot or an embedding.
- **`fullbathcnt` is dropped**, following `changes.md`, whose counts add up to 58.
- **Rare categories** (under 0.01% of training rows) go into an `Other` column, kept separate
  from `Unknown`, which means the value was missing.

## 5. What is left

1. **Write the Phase 2 report** from `REFERENCE.md`, in your own words, with the seven figures.
2. **Tell the professor about the competition licence.**
3. After Phase 2: the models themselves, starting from a baseline that predicts the median.

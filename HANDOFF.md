# Handoff — Machine Learning class project (CSCE 3602 / DSCI 3415)

Written 2026-10-02. Read this first when a session opens in `~/Desktop/ML_project`.
Zeyad Khaled, partner **Mohamed Akram**. Zeyad is the only decision maker: report findings,
propose, then wait for his explicit go-ahead before any action.

---

## 1. Where the project stands

**Phase 1 (due 2026-09-20) was submitted and the professor rejected it on two counts, in his words:**
1. *"your chosen dataset's size is very small and doesn't match the requirements discussed in class.
   You need to find a larger dataset or change the regional scoping"*
2. *"The report's main requirement are not covered - you are not discussing the datasets in depth or
   the limitations of each, but rather iterating the phases that I already shared on the course outlines"*

**Hard requirements he then stated:**
- **At least 300,000 rows** (he first said 500,000, then settled on 300,000).
- Columns must be **real named features**, not fields identified only by numeric codes.

**Phase 2 is due 2026-10-04:** a report on the chosen dataset, the pre-processing and cleaning
process, and a full description of the generated features with the rationale for each.

**The topic so far:** predicting the asking price of residential units for sale in Greater Cairo,
Egypt. Target was the logarithm of price per square metre, multiplied back by area.

---

## 2. The research verdict on Egyptian data (4 agents, 2026-10-02, measured not estimated)

**Egypt cannot meet the 300,000-row requirement. This is settled, with primary evidence.**

- **No single Egyptian dataset reaches 300,000 rows.** The ceiling with a usable licence is
  `mohammedhassan1112/egypt-property-finder` on Kaggle: **64,106 rows, CC0 public domain**,
  23 named English columns (`size` in square metres, `bedrooms`, `bathrooms`, `latitude`,
  `longitude`, `property_type`, `price_period`, `price_type`, `amenities`, `scraped_at_utc` …),
  scraped 2026-02-19, zero duplicate rows, ships a data dictionary. Residential **sale** subset
  is **19,967 rows**. Limitation: every row was scraped inside one 3.5-hour window, so the
  dataset has no time variation at all.
- **The one Egyptian file above 100,000 rows is unusable.** `kamelfares122/house-prices-egypt`
  publishes 225,000 rows but its Kaggle licence field reads **"Unknown"**, its description is
  empty, 145,097 rows are byte-identical duplicates (63,499 distinct listings), locations are
  Arabic-only, and its date column is relative text such as "6 days ago".
- **Merging every Egyptian dataset gives roughly 190,000–280,000 unique rows**, and only because
  the merge includes rentals, commercial units, every governorate and four different years.
  Three Kaggle "datasets" are byte-identical re-uploads of one 27,361-row file (identical
  `totalBytes` of 339,465). Measured cross-portal overlap is 10–20%.
- **The live inventory of Egypt's largest portal is itself under 300,000.** Read off Aqarmap's own
  search pages on 2026-10-02: **214,009 properties for sale** nationwide, **65,516 for rent**,
  **128,044 for sale in Greater Cairo**. Property Finder Egypt showed 177,074 for sale and
  55,069 for rent. So even a complete scrape of one portal, which its terms forbid, misses the floor.
- **Egyptian official data does not exist for this problem.** `data.gov.eg`, the portal named in
  Egypt's own open-data policy, **has no domain-name-system record** on Google or Cloudflare
  resolvers. The Real Estate Registry publishes no transaction microdata; records are obtained in
  person. The Financial Regulatory Authority and the Central Bank publish aggregates only. The
  census microdata (1.74 million household records via IPUMS International) has **no price or rent
  variable**, and its licence forbids redistribution and commercial use.
- **The official platform `realestate.gov.eg` advertises "more than 500,000 documented units"** on
  its App Store page while its own page data reports **151 listings**. Its terms forbid extracting
  or reusing data.
- Hugging Face, Zenodo and Figshare hold **zero** Egyptian property datasets (swept via their APIs).

**All of the above is exactly the dataset-depth analysis the professor said was missing. It belongs
in the Phase 2 report.**

---

## 3. The open decision (nothing has been decided)

| Option | What it means | State |
|---|---|---|
| **1. Switch market to Iran** | `divarofficial/real_estate_ads` on Hugging Face: ~1,000,000 rows, 57 named columns, Open Database Licence, **published by the platform itself** (no scraping, no terms problem). Same emerging-market inflation story, so most of Phase 1 survives. | ⚠️ row count, columns and licence NOT yet verified from the dataset page |
| **2. Keep Egypt, argue the number down** | Take him the measured evidence above and ask whether a documented merge across portals satisfies the requirement. | message to him not drafted |
| **3. Switch to Zillow Prize data** | **VERIFIED 2026-10-02 by reading the real file bytes. Two deal-breakers.** | **not recommended** |
| **4. Keep Egypt and keep 300,000 by changing the problem** | The only verified Egyptian option above the floor is weather: 311,030 station-day records from 23 Egyptian stations, 1884–2026, free, predicting next-day maximum temperature. Throws away all property work. | fallback only |

**Recommendation given to Zeyad: option 1, and send him the option-2 message anyway.**

### Verified facts on the Zillow Prize data (option 3), measured from the files themselves

- `properties_2016.csv` and `properties_2017.csv` each hold **2,985,217 rows and 58 columns**.
  Coverage is Los Angeles (2,012,741 parcels), Orange (745,800) and Ventura (223,744) counties,
  California, snapshots for 2016 and 2017.
- **Deal-breaker one: there is no sale price anywhere in the data.** Both training files hold exactly
  three columns, `parcelid, logerror, transactiondate`. Kaggle's own page defines the target as
  `logerror = log(Zestimate) - log(SalePrice)`, and Zillow withheld both the Zestimate and the sale
  price, so neither can be recovered from one combined number. The target is also near-unlearnable:
  the winning error was 0.0744 against 0.0765 for predicting the constant zero.
- **Deal-breaker two: the licence grants no use outside the competition**, which closed on
  2018-01-10. Quoted from the official rules: participants must use the data *"solely for the purpose
  and duration of the Competition"* and *"shall not transmit, duplicate, publish, redistribute or
  otherwise provide or make available the Data to any party not participating in the Competition."*
  There is no academic, educational or non-commercial exception, and downloading the data is itself
  acceptance of those rules. Third-party re-uploads on Kaggle and Hugging Face exist and even claim
  CC0, but that label is an uploader's self-assertion over data they were never licensed to
  redistribute.
- **Sparsity:** 20 of the 58 columns are over 90% empty and 34 are over 30% empty, so the usable
  feature set is roughly 20 columns.
- **Opaque identifiers:** 21 columns hold numeric codes, and 14 of those have no decode table at all,
  including `regionidcity`, `regionidzip` and `regionidneighborhood`. The location features most
  useful for pricing cannot be mapped to names.
- **The one rescue, if it is ever revisited:** drop the competition framing and predict
  `taxvaluedollarcnt`, the county assessor's total assessed parcel value, which is populated in
  2,950,947 of 2,985,217 rows. Then `structuretaxvaluedollarcnt`, `landtaxvaluedollarcnt` and
  `taxamount` must be dropped as leakage, because the total is their sum and the tax is computed from
  it. An assessed value is still not a market price, and the licence problem remains.

---

## 4. What is in this repository

```
report/
  Phase1_Report/                 main.tex, references.bib (37 entries), main.tex.bak (long pre-rewrite version)
  Phase1_Report_Overleafupdate/  the copy Zeyad opens in Overleaf, plus the matching .zip
  Phase1_Report_Overleaf/        older copy, plus its .zip
data/developers/
  developers.csv        225 Egyptian developers: identity, founding year, website, stock ticker,
                        units delivered, land bank, yearly sales, sales rank, delivery track record
  projects.csv          1,090 projects (696 in Greater Cairo): city, district, land area, launch and
                        delivery year, unit types, finishing, facilities, prices with type and date
  sources.csv           549 sources (257 primary, 280 secondary, 12 weak); every value cites one
  compound_lookup.csv   611 listing compound names linked to developer and project identifiers
  developers_master.csv the 280-name master list with developer identifiers
  build_master.py       merges the stage-1 lists into developers_master.csv
  build_tables.py       merges the six stage-2 slices into the three final tables
  build_compound_lookup.py  links listing compound names to projects (rerunnable)
  stage1/, stage2/      raw per-agent output and the shared Stage 2 brief
```

**Known quality facts about the developer tables:** prices exist for only 24 of 1,090 projects
(developers rarely publish them); delivery status is "unknown" for 637; the compound lookup matches
64.5% of listings, 20.1% name only a developer brand such as "Palm Hills", and 73 rows need a human
yes-or-no review. Two fake entries were already removed ("Rabaa Investment" is a Nasr City place
name; MRB is a facility-management consultancy). The government blog `realestate.gov.eg` proved
unreliable and the rows relying on it are flagged.

**Nothing is committed yet.** The repository has no commits. Remote is
`https://github.com/Zab-17/ML_project.git`. Git identity must be
`Zab-17 <128490152+Zab-17@users.noreply.github.com>`, never any other address.

---

## 5. Decisions Zeyad locked in earlier (do not re-litigate)

- Sale listings only; rent rows deleted. **May need revisiting if the merge route is chosen.**
- Target: logarithm of price per square metre, converted back by multiplying by area.
- Geography: Cairo and Giza governorates plus every new city in Greater Cairo; coastal excluded.
- The 2022 OLX dataset is motivation and comparison only, never training or testing.
- An **area-quality index** is his own contribution: median price per square metre of a district or
  compound divided by the training median, shrunk towards the parent town by listing count, then
  binned into ordered tiers. It must be fitted inside the model pipeline so it cannot leak the answer.
- **Cash prices only**, no instalment prices.
- The web application is the intended use, not a commitment; build-or-not is decided after Phase 3.
- Output to a user: a headline price plus a tight band plus five comparable listings. A fixed
  ±100,000 Egyptian pound band was rejected as impossible to deliver honestly.

## 6. Four preprocessing decisions still unanswered

1. Fuzzy de-duplication coarseness (proposed: area to 5 square metres, price to 1%, coordinates to
   about 100 metres).
2. The hub list for distance features (proposed: Downtown, Fifth Settlement 90th Street, Sheikh
   Zayed centre, 6th of October centre, Smart Village, New Capital government district).
3. The area-quality index form: ordered tiers, the continuous value, or both.
4. The validation split: ordinary stratified folds, folds grouped by district, or both.

## 7. Tool stack chosen by research, NOT yet installed

Already installed: Python 3.12.5, pandas 2.3.3, numpy 2.4.3, scikit-learn 1.8.0, RapidFuzz 3.14.5,
pyarrow 21, matplotlib 3.10.8.

Proposed: `pip install pandera skrub geopandas quackosm`, after a `pip install --dry-run` check that
nothing downgrades numpy. Rejected with evidence: ydata-profiling (pins `numpy<2.4`), Polars and
DuckDB (no benefit at this scale), recordlinkage and dedupe (unmaintained), Great Expectations (too
heavy), pyjanitor (requires pandas 3). Splink and cleanlab are optional later additions. Distances
use a NumPy haversine formula; metric work uses European Petroleum Survey Group code 32636
(Universal Transverse Mercator zone 36 North).

## 8. Licence facts that matter for a sellable product

- The Property Finder March 2026 dataset is **CC BY-NC-SA 4.0: non-commercial**.
- The Dubizzle GitHub dataset has **no licence file**, so all rights reserved, and its features were
  extracted by a language model rather than observed.
- `mohammedhassan1112/egypt-property-finder` is **CC0**, the only clean commercial option.
- Nawy, Property Finder, Aqarmap, Bayut, Dubizzle and Emaar Misr all forbid automated collection.
  Do not fetch them, with the browser extension or otherwise.
- OpenStreetMap is Open Database Licence: credit it and keep derived tables internal.

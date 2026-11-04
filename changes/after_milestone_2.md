# After Milestone 2: everything fixed in the Zillow rerun, and why

Code in `zillow/`. Written 1 November 2026, before the Milestone 3 models ran.

King County (the Seattle area) is the sale-price dataset we ran first. That work is on the `king-county` branch, and many fixes below copy its checks.

The Zillow dataset from Phase 2 (the one Mohamed's feature list was built on, marked down in Milestones 1 and 2) was rebuilt from the raw file. It went through every check King County went through, and then through every model in the slides. This file lists each fix: what Phase 2 did, what changed, and the measured reason.

The label is still the county's **2016 assessed value** (`taxvaluedollarcnt`), not a sale price. That is a limit of the Zillow file itself and no fix can change it (see the end).

---

## Part 1. Fixes to the rows

### Fix 1. Only ordinary homes with a building are kept

**Before:** Phase 2 kept every row that had a label: 2,950,951 rows. Those included shops and offices, parcels with no building, homes with 0 bedrooms, and assessed values as low as $1. 2,010 training rows were under $100.

**Now:** the King County row rules are copied as far as the Zillow file allows. They run in this order, and each home is counted once, under the first rule it fails:

| Rule | Homes removed |
|---|---|
| No assessed value | 34,266 |
| Assessed in a year other than 2016 (the label must be from one year) | 3,379 |
| Assessed value under $10,000 (King County's price floor) | 14,505 |
| Not a residential home: shops and offices with flats (codes 31 and 47), a residential common area (270), one row of code 279, or no type | 13,768 |
| No living area recorded, which means no building | 25,926 |
| 0 bedrooms or no bathroom (the King County rule) | 34,071 |
| **Homes kept** | **2,859,302** of 2,985,217 |

The kept types are single family, condominium, duplex, triplex, quadruplex, planned unit development, mobile home, cluster home, residential general, cooperative, manufactured home and townhouse.

**Why:** a shop's assessed value and a house's assessed value are different questions. A $1 assessment is a bookkeeping entry, not a value. Rows like those teach the models nothing about homes.

### Fix 2. A new 80/20 split, and five folds by parcel

**Before:** Phase 2 split all 2,950,951 labelled rows.

**Now:** the kept homes are split 80/20 with seed 42: **2,287,442 training homes and 571,860 test homes**. The test homes are sealed: no setting is chosen by looking at them.

The training homes are split into five folds from a hash of the parcel number, as in King County, with about 457,000 homes per fold.

**Why:** the rows changed (Fix 1), so the old split no longer matched the data. In the Zillow file each home is exactly one row (all 2,985,217 parcel numbers are different). That means the King County problem of one house appearing on both sides of the split cannot happen here.

### Fix 3. The label is the log of the assessed value

**Before:** raw dollars, from $1 to $287 million.

**Now:** the natural log, exactly as King County did with the sale price. Every dollar score converts the log back to dollars.

**Why:** home values are skewed. On a log scale, a 10% miss counts the same on a cheap home as on an expensive one, and a few very expensive homes no longer dominate the training.

---

## Part 2. Fixes to the columns

Each of the 58 raw columns now has a written decision and reason in `zillow/zillow_columns.py`. The measured numbers sit next to each decision in `zillow/out/checks/column_check.csv`: share blank, number of values, strength with the label, and strongest partner.

### Fix 4. Location: no more raw coordinates; city, ZIP, neighbourhood and census tract brought back

This was the main thing marked down.

**Before:** Phase 2 fed raw latitude and longitude into the model as two plain numbers. Mohamed's analysis dropped city, county, ZIP code, neighbourhood and census tract with the reason "location is already represented by fips, latitude and longitude".

**What was wrong with that:** a raw coordinate is a straight-line measure. Value does not rise steadily as you go north or east. Expensive and cheap neighbourhoods sit side by side. A linear model given only latitude and longitude can only learn "further north is worth X more per degree", which is not how location works.

**Now:**
- **City, ZIP code, neighbourhood and census tract** are brought back as categories, one-hot encoded. The census tract is the raw tract-and-block number with the block digits removed.
- **Latitude and longitude** became *miles north* and *miles east* of the median training home. A degree of longitude is 57 miles here and a degree of latitude is 69, so raw degrees are not even distances. The miles columns were only candidates: the correlation rules decide (Part 3).

**What the correlation checks decided on the training homes:**
- ZIP code is the strongest location feature: 0.490 with the label on held-out homes.
- Miles north (0.983 with ZIP code) and miles east (0.988) were dropped, because ZIP code already carries them and is far closer to the label (0.490, against 0.119 and 0.013). King County's miles columns went the same way.
- City was dropped: 0.921 with ZIP code, and weaker with the label (0.391 against 0.490).
- Neighbourhood (98 columns) and census tract (16 columns) stayed.

### Fix 5. The county code (FIPS) is a named category, and its duplicate is removed

**Before:** the county's Federal Information Processing Standard code (6037, 6059, 6111) was one-hot encoded next to raw coordinates. Its copy `regionidcounty` was dropped with the reason "already represented", not with a measurement.

**Now:**
- The county code is read as a name: Los Angeles, Orange or Ventura.
- `regionidcounty` is dropped with a measured reason: Cramér's V of 1.000 with the county code. It is the same three counties under different numbers.
- The correlation rules then dropped the county itself. It is 0.998 with the county land-use code, because each county writes its own land-use codes, and the land-use code is closer to the label (0.290 against 0.109). The model still knows the county through the land-use code and ZIP code columns.

### Fix 6. Columns that some counties never record (a silent fault in Phase 2)

**Found:** several columns are not missing at random. They are recorded by some counties and never by others:

| Column | Los Angeles | Orange | Ventura |
|---|---|---|---|
| Garage area | 100% blank | 4% blank | 1.5% blank |
| Fireplaces | 100% blank | 78% blank | 25% blank |
| Building quality | 1.4% blank | 100% blank | 100% blank |
| Heating | 7% blank | 93% blank | 100% blank |
| Stories | 100% blank | 35% blank | 1% blank |
| Zoning | 0.3% blank | 100% blank | 100% blank |
| Room count | 100% zeros | 35% zeros | 0.5% zeros |
| Unit count | 0.3% blank | 99.8% blank | 99.8% blank |

**Before:** Phase 2 filled every blank the same way (training median, or "Unknown") without knowing that a blank in Los Angeles means "never recorded" while a blank in Orange means "none".

**Now:** each column is handled by what its blank actually means:
- **Garage area and fireplaces:** a blank becomes 0. In the counties that record them, a blank means none. In Los Angeles the 0 means "not recorded", and the ZIP code and land-use code columns tell the model which county it is in.
- **Building quality, heating, air conditioning and zoning:** a blank is its own category, "not recorded".
- **Stories:** a blank gets the training median (1 story).
- **Room count:** dropped. Los Angeles writes 0 for every home, and elsewhere bedrooms and bathrooms already count the rooms.
- **Unit count:** dropped. It is recorded only in Los Angeles, and the property type (duplex, triplex, quadruplex) already gives it.

**Caveat:** the correlation rules dropped garage area at 0.906 with the county land-use code. That high value comes mostly from the recording pattern: every Los Angeles home has 0 garage area, and the land-use code says which county a home is in. So the rule removed a column that does carry real information in Orange and Ventura. The rule was applied exactly as written, the same as in King County. Overruling it would be your call.

### Fix 7. Zoning kept instead of dropped (the professor's rule 3)

**Before:** dropped, because "propertyzoningdesc has thousands of codes, so encoding is difficult".

**What was wrong with that:** the professor's rule 3 says a text feature dropped without a measured justification costs marks, and "difficult to encode" is not a measurement. Measured on held-out homes, zoning is 0.411 with the label, the fifth strongest of the 58 columns. It is only 0.505 with ZIP code and 0.680 with city, so it carries something the location columns do not. King County dropped zoning because there it repeated the location columns. Here it does not.

**Now:** zoning is one-hot encoded, with codes under 0.1% of training homes grouped together. It is "not recorded" outside Los Angeles. It passed every correlation rule and gives **120 model columns**.

### Fix 8. One-hot instead of frequency encoding for the county land-use code

**Before:** the county land-use code was replaced by how common it is (frequency encoding). Two different codes that happen to be equally common got the same number.

**Now:** one-hot, with the King County rare-value rule. That gives **19 columns**.

### Fix 9. Columns Phase 2 left out are now measured, and kept where they earn it

Pool (a yes/no column), hot tub or spa (yes/no, recorded only in Orange), fireplaces, stories, air conditioning and zoning were not in Phase 2. Each one now has a measured decision. Garage area was also brought back as a candidate, but the correlation rules dropped it (the caveat under Fix 6).

### Fix 10. Duplicates and parts are dropped with a measured number, not a guess

| Dropped | Measured against | Value |
|---|---|---|
| `calculatedbathnbr` | bathrooms | correlation 1.000 (the same number) |
| `fullbathcnt` | bathrooms | correlation 0.987 (a part of it) |
| `finishedsquarefeet12` | living area | correlation 1.000 |
| `pooltypeid2`, `pooltypeid7`, `pooltypeid10` | the pool and hot tub columns | the same homes in 99.98% to 100% of cases |
| `regionidcounty` | the county code | Cramér's V 1.000 |

### Fix 11. The leak columns: removed, and the 0.85 rule alone would not have caught them

Building value plus land value equals the label exactly, and the tax bill is about 1% of it. All three stay removed.

On the log label they measure only 0.583 to 0.653, under the professor's 0.85 limit, because they are in dollars while the label is a log. That is why they are removed by name, as leaks, and are barred from the "comes back" rule (Part 3, rule 3).

### Fix 12. The AI-use statement is in every file

Undisclosed AI use was one of the marked-down items. Every new or edited code file carries the line "AI was used to help write this code, but the resulting code was reviewed and edited by a human."

---

## Part 3. The King County checks, run on Zillow

Run by `zillow/zillow_checks.py` on the 2,287,442 training homes. The outputs are in `zillow/out/checks/`.

1. **One column at a time:** all 58 columns, in `column_check.csv`.
2. **Association tables and heatmaps:**
   - `original_58_columns_*` and `chosen_23_features_*`, with the King County measures: Pearson correlation for two numbers, correlation ratio on held-out homes (fold 0) for a category and a number, and bias-corrected Cramér's V for two categories.
   - The heatmap file is named for 23 features because the check runs on all 23 candidates before the rules remove 6, which leaves the 17 in the table.
3. **Professor's rule 4, features at 0.85 or more with the label:** none. The strongest is the living area at 0.525.
4. **Zeyad's pair rule, features at 0.85 or more with each other** (drop the one weaker with the label, strongest pair first). Six went:

   | Dropped | Value | Partner that stayed |
   |---|---|---|
   | county | 0.998 | county land-use code |
   | miles east | 0.988 | ZIP code |
   | miles north | 0.983 | ZIP code |
   | property type | 0.935 | county land-use code |
   | city | 0.921 | ZIP code |
   | garage area | 0.906 | county land-use code |

5. **Zeyad's rule 3, bring back a dropped column** that is between 0.60 and 0.80 with the label and under 0.85 with every chosen feature: **nothing qualified.** The only columns in that window are the three leaks, which are barred.
6. **Second look**, a dropped column at 0.15 or more with the label that no chosen feature covers at 0.7: 10 columns were flagged. Three are the leaks. The other seven are covered by their written reasons:
   - parts of the living area that are 93% or more blank;
   - basement area (99.95% blank);
   - garage car count (the same garages as garage area);
   - patio area (Ventura only);
   - pool size (99% blank);
   - three-quarter bathrooms (part of bathrooms, never recorded in Los Angeles).
7. **One baseline column dropped per category group:** 8 columns, so no group's columns add up to 1.
8. **Model columns at 0.85 or more with each other:** 28 columns went. Most were a neighbourhood or census tract that covers exactly the same homes as one ZIP code.
9. **No column is an exact combination of the others:** the smallest eigenvalue of the correlation matrix is 0.0041, where 0 would mean an exact copy.

**Result:** 17 features, which become 606 model columns: 7 standardized numbers, 2 yes/no columns and 597 one-hot columns.

---

## Part 4. Changes to the shared King County code

None of these changes any King County result. The Zillow run goes through the same copy of each model.

| File | Change | Why |
|---|---|---|
| `train_models.py`, `neural_search.py` | The table folder can be switched with `MODEL_TABLE_DIR`; it defaults to the King County table | One copy of every model for both datasets |
| `final_training.py` | Also saves each model's test predictions | Needed by voting and the price-band table |
| `slide_models.py` | K-nearest neighbours sizes its graphics processing unit batches from the training-set size (1,099 rows for King County, close to the old 1,024; 204 for Zillow), and writes distances into one buffer | A batch of 1,024 against 1.8 million homes needs 7.5 GB. Tested to give exactly the same neighbours as before. |
| `slide_models.py` | `main()` became `run(models, names)` | So the Zillow model list reuses the same four steps |
| `src/code_tables.py` | Also reads the air-conditioning names from Zillow's dictionary | Readable column names; Phase 2's output does not change |

---

## Part 5. Model changes, and the models King County never ran

**Every model goes through Zeyad's four steps:**
1. a five-fold search for its best setting;
2. the winner chosen on validation scores only;
3. one final fit on all 2,287,442 training homes;
4. one score on the 571,860 sealed test homes.

| Model (lecture) | Same as King County? | Change and reason |
|---|---|---|
| Baselines, linear regression (4), neural network (7) | Same | — |
| Support vector machine (12), K-nearest neighbours (3) | Same settings | — |
| Decision tree (9, 10) | Changed | Pre-pruning by the smallest leaf allowed (1, 5, 20, 50, 200 homes) instead of cost-complexity pruning. In King County each pruned tree took about 6 minutes on 341,000 homes. A full tree here takes about 3 minutes on 1.8 million homes, so the King County grid would take days. |
| Random forest and bagging (10, 13) | Changed | At least 5 homes per leaf. Measured: one fully grown tree on these homes has about 2 million nodes, so 50 of them would not fit in 16 GB. With 5 homes per leaf one tree has 276,000 nodes. |
| **AdaBoost (13)** | **New** | AdaBoost for a number (AdaBoost.R2), 50 rounds of trees of depth 4 or 8 |
| **Voting (13)** | **New** | The average prediction of the three models with the best validation scores; the test homes play no part in choosing them |
| **K-means clustering (14)** | **New** | The training homes are grouped into 50, 200 or 1,000 clusters without looking at the label. A home is priced at the median value of its cluster. |
| **Hierarchical clustering (14)** | **New** | The same, with Ward linkage, on a random sample of 10,000 training homes. Ward needs the distance between every pair of homes: 1.7 × 10¹² pairs for all of them. |
| **Logistic regression (5), perceptron (6), naive Bayes (11)** | **New** | These predict classes, so the value is cut into five equal price bands at the training quintiles. Every number model is also scored on the same bands, so all models share one table. Naive Bayes uses a normal distribution for each number and a yes/no probability for each 0/1 column. |

Text embedding (Lecture 8) is not used. Zillow has no free text; its text columns are codes, which are one-hot encoded, as the professor's rule 3 allows.

---

## Part 6. What is still open

1. **The label is an assessed value, not a sale price.** Under California's Proposition 13 a home's assessed value is reset only when the home sells. Two identical neighbouring houses can carry very different labels, and no feature can explain that.
2. **The Zillow licence** allows use only for a competition that closed on 10 January 2018. The professor has not been told.
3. **Fine location is lost.** Measured on held-out homes, the raw census block (about 94,000 values) is 0.592 with the label, stronger than ZIP code (0.490 to 0.501). But one-hot at block level would need about 94,000 columns, and the 0.1% rule leaves the census tract only 16 columns. The usual fix is target encoding: each home gets the average value of other homes in its block, learned on other folds. It is not used in King County, so it is not used here without your decision.
4. **Garage area** was dropped by the pair rule because of the recording pattern (Fix 6).
5. **The report's references** were the other marked-down item. That is a report task, not a code task.

# King County house prices: what we did, how, and the results so far

Zeyad Khaled, 26 October 2026. Branch `zeyad`.

## The short version

We predict the price a house in King County, Washington (the Seattle area) sold for. The inputs are the county assessor's records of the house, its lot, its location and the sale date. After cleaning there are 532,890 sales from 2000 onwards. Each sale is described by 1,604 model columns built from 55 features.

On homes the models had never seen:

| Model | Median miss | Homes within 10% of the price |
|---|---|---|
| Predict the average price ($663,874) for every home (fold 0 only) | 50.9% ($283,924) | 11.5% |
| Linear regression | 12.9% ($65,173) | 39.9% |
| Neural network, 128 → 64 neurons | 8.2% ($39,574) | 58.4% |

The sealed test set (106,755 sales) has not been touched yet. It gets scored once, after the final network is trained.

Every column decision, with its measured reason, is in `FEATURE_DECISIONS.md`. This document is the overview.

---

## 1. The data

**Why we changed dataset.** The Zillow file had no sale prices. Its label was the assessor's estimated value, and its licence covered only a competition that closed in January 2018. The King County Assessor publishes recorded sale prices next to the house records, free to download.

**Source:** King County Department of Assessments, data download page, <https://info.kingcounty.gov/assessor/DataDownload/default.aspx>. Four files:

| File | One row is |
|---|---|
| `EXTR_RPSale.csv` | one recorded sale |
| `EXTR_ResBldg.csv` | one residential building |
| `EXTR_Parcel.csv` | one piece of land (a parcel) |
| `EXTR_LookUp.csv` | the names behind the numeric codes |

The files join on the parcel number (`Major` + `Minor`). Together they hold **151 columns**.

**Which sales count.** A sale is kept only when it is an ordinary market sale of one house:

- sale reason 1 (no special circumstance), and the deed is a warranty deed or a statutory warranty deed (codes 2 and 3);
- no sale warning, and only one parcel in the sale;
- principal use 6 (residential), and property class 8 (residential land with a building on it);
- price at least $10,000, and sold in 2000 or later;
- the parcel has exactly one house, and the house was not built or renovated after the sale.

Result: **533,055 sales of 328,776 homes**, against the course minimum of 500,000 rows and 50 columns. After that, 165 sales with 0 bedrooms or no bathroom were removed (124 training, 41 test; decisions 10 to 13).

## 2. The split: 80% training, 20% test, by home

Many houses sold more than once. If one sale of a house sat in training and a later sale sat in the test set, the model could recognise the house instead of pricing it. So the split is **by home** (parcel), with seed 42, and every sale of one house lands on the same side.

| Part | Sales |
|---|---|
| Training | 426,135 |
| Test, sealed | 106,755 |

Everything the pipeline learns from data comes from the training sales only: category lists, rare-value groups, baselines, and the averages and spreads used for scaling. The test sales are transformed with those training numbers.

## 3. The label: the natural log of the price

House prices are lopsided. Most sales are in the hundreds of thousands, and a few run into the millions.

In raw dollars, the most expensive 1% of sales carry 44.9% of the squared error. A model trained on dollars would spend its effort on mansions. After the natural log they carry 9.9%, and the skew of the label drops from 6.15 to 0.47.

On the log scale, the difference between two values reads as a percentage, so a 10% miss counts the same on a $300,000 house and on a $3,000,000 house:

| House | Real price | Prediction | Off by | Squared error in dollars | Squared error in log price |
|---|---|---|---|---|---|
| Small house | $300,000 | $330,000 | 10% | 900,000,000 | 0.0091 |
| Mansion | $3,000,000 | $3,300,000 | 10% | 90,000,000,000 | 0.0091 |

This is why the models are trained, and mainly scored, on log price. The dollar scores are still reported further down. They undo the log first.

## 4. Choosing the features

### 4.1 One column at a time

Every one of the 151 columns was measured on the training sales before Zeyad decided to keep it, transform it or remove it. For each column the check showed:

- a table of its values, with each value's share of sales and its price compared with the typical sale in the same month;
- the share of the price differences that the column explains;
- its correlation with the label.

Columns with many names, such as locations, subdivisions and streets, were measured on held-out homes. The column was fitted on 4 of every 5 homes and scored on the 5th, so a column with thousands of names could not score itself on its own homes.

What stayed, in groups:

| Group | Features |
|---|---|
| Size | above-ground living area (log), finished basement, other unfinished basement, total garage, open porch, deck, lot area (log) |
| Rooms | bedrooms, full / three-quarter / half bathrooms, stories, total fireplaces |
| Age and quality | year built, decade renovated, building grade (5 to 12), finished basement grade |
| Location | appraisal area + sub-area (299 codes), subdivision name (at least 100 sales), street name (at least 200 sales), street type |
| Views and water | 10 views as yes/no, which water the house is on, waterfront length, waterfront bank height |
| Lot and services | hilly lot, traffic noise (0 to 3), airport noise, green belt, water supply, sewer, road access, street surface, parking |
| Legal and nuisances | power lines, golf course, easements, deed restrictions, protected native-plant strip, other designation, state land lease, water problems |
| Property type | single family, townhouse, duplex and three smaller types |
| Date | years since 1 January 2000, with decimals (14 June 2010 = 10.45) |

What went, and why, in short (each with its own number in `FEATURE_DECISIONS.md`):

- record numbers and identifiers that never repeat on a new sale;
- columns that are empty, or say the same thing for almost every house;
- **buyer and seller names**, which are personal data (the King County licence restricts the use of lists of individuals) and not a property of the house;
- parts that add up to a column we already keep (the floor areas add up to the living area);
- weak extras that add almost nothing once the neighbourhood is known (for example the house condition, at −0.005 with the label);
- ZIP code, tax district and zoning, which mostly repeat the location columns we kept (and the same zoning code means different rules in different cities).

### 4.2 Encoding and scaling (professor's rules 1 and 3)

| Kind | Treatment | Columns |
|---|---|---|
| Numbers | standardized: minus the training average, divided by the training spread | 17 |
| Yes/no | stay 0 or 1 | 23 |
| Categories and text | one-hot: one 0/1 column per value. The most common value of each group is dropped as the baseline, and values under 0.1% of sales are grouped together | 1,564 |

The text columns are subdivision name, street name and street type. They are fixed place names, not free text, so each name gets its own one-hot column, which rule 3 allows. Only names with enough sales get their own column, because a name seen a dozen times can't teach the model a reliable price. Subdivisions need at least 100 sales (800 columns) and streets at least 200 (404 columns). Every other name shares one "other" column.

### 4.3 Correlation checks (professor's rules 2 and 4)

Three measures, one for each type of pair:
- **Pearson correlation** for number against number.
- **Correlation ratio** for category against number, measured on held-out homes.
- **Bias-corrected Cramér's V** for category against category.

Two full tables, with heatmaps, are in `king_county/kc_features/out/`:
- all 151 original columns (`original_151_columns_*`);
- the 60 chosen features (`chosen_60_features_*`).

Zeyad's rules, applied in this order:

1. **Features at 0.85 or more with each other:** drop the one weaker with the label. Five went:
   - appraisal area, 1.000 with area + sub-area;
   - miles north, 0.998;
   - miles east, 0.996;
   - tidelands, 0.986 with which water;
   - city, 0.868 with area + sub-area.
2. **Features at 0.85 or more with the label:** none. The highest is the sale date at 0.641.
3. **Bring back a dropped column** if it is under 0.85 with every chosen feature and between 0.60 and 0.80 with the label. Only the parcel number (0.662) fell in the label window, and it is 0.998 with miles north, so nothing came back.
4. **One baseline column dropped per category group:** 15 columns.
5. **Model columns at 0.85 or more with each other:** two subdivision columns that copied a sub-area column went.
6. **Check:** no column can be computed exactly from the others. The smallest eigenvalue of the correlation matrix is 0.0067; zero would mean an exact copy.

**Result: 55 features and 1,604 columns.**

---

## 5. How the models were trained and checked

### 5.1 Five-fold cross-validation on the training sales

The 426,135 training sales are split by home into 5 folds of about 85,000. Each model is trained 5 times, from scratch each time. Each run learns from 4 folds (about 341,000 sales) and is checked on the fifth, so every training home is checked exactly once by a model that never saw it.

| Run | Learns from | Checked on |
|---|---|---|
| 1 | folds 1–4: 340,909 sales | fold 0: 85,226 |
| 2 | folds 0, 2–4: 340,863 | fold 1: 85,272 |
| 3 | folds 0, 1, 3, 4: 340,520 | fold 2: 85,615 |
| 4 | folds 0–2, 4: 341,198 | fold 3: 84,937 |
| 5 | folds 0–3: 341,050 | fold 4: 85,085 |

The test sales play no part here. They wait for the final model.

### 5.2 The models

- **Average baseline:** predicts $663,874, the average training price, for every home. Any model must beat this to be worth anything.
- **Median baseline:** predicts $505,000, the median training price, for every home.
- **Linear regression:** one weight per column, fitted by least squares. Each weight reads as "this column adds X% to the price".
- **Neural network:** the 1,604 columns feed a first layer of neurons, then a second, then one output, the predicted log price. "128 → 64" means 128 neurons in the first layer and 64 in the second. Six settings were tried: three sizes (128 → 64, 256 → 128, 512 → 256), each with and without dropout 0.2.

The neural network's training terms:

| Term | Meaning here |
|---|---|
| Pass (epoch) | every training sale seen once, in batches of 1,024 sales: about 333 weight updates per pass |
| Loss | mean squared error on log price, the number each update tries to shrink |
| Optimizer | stochastic gradient descent, which nudges every weight a small step downhill on the loss after each batch |
| Learning rate 0.1 | the size of that step (Zeyad's choice) |
| Dropout 0.2 | during training, each neuron is switched off at random 20% of the time, so the network can't lean on a few neurons and memorise |
| Early stopping | after every pass, score the checking fold. If 5 passes in a row bring no improvement, stop and keep the best pass. At most 100 passes |
| Hardware | the M1 Pro's graphics chip, through PyTorch |

Adam (Adaptive Moment Estimation) at learning rate 0.1 was tried first in a short test and blew up: its error was 0.37 after one pass and got worse. Stochastic gradient descent at 0.1 went from 0.084 to 0.070 in two passes, so the search used it. Adam at its standard rate of 0.001 reached 0.146 after two passes, and has not been run in full yet.

### 5.3 The scores

| Score | Plain meaning | Better when |
|---|---|---|
| R-squared | share of the price differences between homes that the model explains. 1 = all of them, 0 = no better than predicting the average | higher |
| Mean squared error (log) | average squared miss on log price. The number the network is trained to shrink | lower |
| Root mean squared error (log) | its square root. 0.14 is about a 15% miss, with large misses weighing extra | lower |
| Mean absolute error (log) | average miss on log price. 0.10 is about 10 to 11% | lower |
| R-squared (dollars) | R-squared counted in dollars, where expensive homes dominate | higher |
| Root mean squared error (dollars) | typical dollar miss, pulled up by expensive homes | lower |
| Mean absolute error (dollars) | average dollar miss | lower |
| Median dollars off | half the homes miss by less than this | lower |
| Mean absolute percentage error | average percent miss | lower |
| Median percent off | half the homes miss by less than this | lower |
| Within 5% / 10% / 20% | share of homes whose prediction lands that close to the price | higher |

---

## 6. Results

### 6.1 Baselines and linear regression (fold 0, 85,226 homes not seen in fitting)

| Score | Average baseline | Median baseline | Linear regression |
|---|---|---|---|
| R-squared (log) | −0.123 | −0.005 | 0.900 |
| Mean squared error (log) | 0.466 | 0.417 | 0.0415 |
| R-squared (dollars) | 0.000 | −0.073 | 0.877 |
| Root mean squared error (dollars) | $600,860 | $622,354 | $211,014 |
| Median dollars off | $283,924 | $205,500 | $65,637 |
| Median percent off | 50.9% | 41.7% | 13.0% |
| Within 10% | 11.5% | 11.3% | 39.7% |
| Within 20% | 22.6% | 23.4% | 70.1% |

- The average baseline scores below zero on the log scale. A few very expensive homes pull the average price above what a typical home sells for, so on the log scale its guess sits too high for most homes.
- Linear regression scores R-squared 0.902 on the homes it was fitted on and 0.900 on new homes, so it is not memorising.
- Linear regression's fit takes about 5 seconds.

### 6.2 Neural networks: averages over the five folds

| Setting | Folds that trained | Mean squared error (log), average ± spread | R-squared (log) | Median percent off | Within 10% | Best pass (median) | Minutes for 5 runs |
|---|---|---|---|---|---|---|---|
| 512 → 256, no dropout ⚠️ | 4 of 5 | 0.0192 ± 0.0018 | 0.954 | 7.9% | 60.4% | 28 | 5.5 |
| 512 → 256, dropout 0.2 ⚠️ | 4 of 5 | 0.0203 ± 0.0046 | 0.951 | 8.0% | 59.9% | 49 | 10.6 |
| 256 → 128, dropout 0.2 | 5 of 5 | 0.0204 ± 0.0031 | 0.951 | 8.1% | 59.4% | 56 | 9.4 |
| **128 → 64, no dropout** | **5 of 5** | **0.0205 ± 0.0014** | **0.951** | **8.2%** | **58.4%** | **18** | **3.6** |
| 256 → 128, no dropout | 5 of 5 | 0.0218 ± 0.0064 | 0.947 | 8.4% | 57.8% | 27 | 4.4 |
| 128 → 64, dropout 0.2 | 5 of 5 | 0.0231 ± 0.0043 | 0.944 | 8.7% | 56.0% | 49 | 7.2 |
| *Linear regression* | *5 of 5* | *0.0412 ± 0.0002* | *0.901* | *12.9%* | *39.9%* | — | *under 1* |

⚠️ Both 512 → 256 settings blew up on fold 1: their error became infinite on the first pass. Their averages cover the other 4 folds only.

### 6.3 The five networks of the steadiest setting (128 → 64, no dropout)

| Checked on | Best pass | Mean squared error (log) | R-squared (log) | Median dollars off | Median percent off | Within 10% | Within 20% |
|---|---|---|---|---|---|---|---|
| Fold 0 | 29 | 0.0188 | 0.955 | $37,046 | 7.7% | 61.2% | 89.1% |
| Fold 1 | 21 | 0.0195 | 0.953 | $38,546 | 7.9% | 59.8% | 87.6% |
| Fold 2 | 14 | 0.0205 | 0.951 | $39,472 | 8.2% | 58.2% | 87.2% |
| Fold 3 | 17 | 0.0219 | 0.947 | $41,536 | 8.6% | 56.6% | 85.6% |
| Fold 4 | 18 | 0.0220 | 0.948 | $41,267 | 8.7% | 56.2% | 86.7% |

Across all 28 networks that trained:
- **Best:** 512 → 256 with dropout 0.2, checked on fold 0. Mean squared error 0.0164, median 6.9% off, 65.5% of homes within 10%.
- **Worst:** 256 → 128 with no dropout, checked on fold 2. It stopped at pass 6 with 0.0332, and still beat linear regression (0.041).

### 6.4 Every score: the two leading networks against linear regression (five-fold averages)

| Score | 128 → 64, no dropout (5 folds) | 512 → 256, no dropout (4 folds) | Linear regression (5 folds) |
|---|---|---|---|
| R-squared (log) | 0.951 | 0.954 | 0.901 |
| Mean squared error (log) | 0.0205 | 0.0192 | 0.0412 |
| Root mean squared error (log) | 0.143 | 0.139 | 0.203 |
| Mean absolute error (log) | 0.106 | 0.102 | 0.158 |
| R-squared (dollars) | 0.926 | 0.930 | 0.882 |
| Root mean squared error (dollars) | $161,722 | $157,527 | $203,927 |
| Mean absolute error (dollars) | $76,171 | $73,436 | $107,389 |
| Median dollars off | $39,574 | $37,727 | $65,173 |
| Mean absolute percentage error | 10.7% | 10.4% | 16.2% |
| Median percent off | 8.2% | 7.9% | 12.9% |
| Within 5% | 32.2% | 33.7% | 20.5% |
| Within 10% | 58.4% | 60.4% | 39.9% |
| Within 20% | 87.2% | 87.9% | 70.1% |

### 6.5 What the results mean

- **The neural network halves linear regression's squared error** (0.0205 against 0.0412) and beats it on every score. Linear regression gives each column one fixed effect everywhere. The network can learn combinations of columns, for example a different value for the same view in different areas.
- **On a typical home the network is about 8% off.** About 6 in 10 homes land within 10% of the sale price, and almost 9 in 10 within 20%. Linear regression manages 4 in 10 and 7 in 10.
- **Dollar scores look worse than percent scores because of expensive homes.** An 8% miss on a $3,000,000 house is $240,000. That is why the root mean squared error in dollars ($161,722) is four times the median dollar miss ($39,574).
- **The top four settings are too close to rank by error alone.** Their averages, 0.0192 to 0.0205, differ by less than the fold-to-fold spread. 128 → 64 with no dropout is the steadiest: its folds disagree least, every fold trained, it peaks after 18 passes and it is the smallest.
- **Linear regression still wins on three things:** steadiness (its folds differ by 0.0004), speed (5 seconds against about 45 seconds per network) and readability (one weight per column).

### 6.6 Problems found

- **Learning rate 0.1 is too jumpy for the larger networks.** Both 512 → 256 runs on fold 1 blew up on the first pass, and one 256 → 128 run quit at pass 6.
- **The networks checked on fold 4 stopped early** (best passes 15 to 22). Linear regression finds fold 4 as easy as the other folds (0.0411 to 0.0415), so the homes are not harder. The early stops come from the error bouncing at learning rate 0.1 and triggering early stopping too soon. A smaller, steadier step (Adam at 0.001, or stochastic gradient descent at 0.05) should help.

### 6.7 Time

- The 30 networks took **41 minutes** in total (25 October, 23:01 to 23:42) on the M1 Pro.
- One pass over about 341,000 sales takes 1.8 seconds (128 → 64) to 2.7 seconds (512 → 256).

---

## 7. What comes next

1. **Five-fold search for Adam at learning rate 0.001** on the same six networks (Zeyad's request): about 20 to 60 minutes.
2. **Pick the winner** by checking-fold error across both optimizers, counting only settings where all five folds trained.
3. **Full training:** the winner, from fresh weights, on all 426,135 training sales, for its median best pass. About 1 to 3 minutes.
4. **Score the sealed test set once:** every score, next to linear regression refitted on all 426,135 sales and the average baseline. These are the numbers for the report.

Still open:
- a "new house" yes/no feature (sold within a year of being built), measured at +3.9 points of price explained and not yet decided;
- a check on the 211 very cheap sales (under $89,756);
- the report figures;
- telling the professor about the switch to King County.

## 8. Running it again

The King County files are not in this repository: they come to about 1 gigabyte. Download the four files listed in section 1 into one folder, then:

```bash
pip install numpy pandas scipy scikit-learn torch matplotlib
cd king_county/kc_features
python3 pipeline.py <king folder>      # about 1 minute: builds out/model_table/
python3 train_models.py                # baselines and linear regression, seconds
python3 neural_search.py --smoke       # short test of the network code
python3 neural_search.py               # the five-fold search, about 41 minutes on an M1 Pro
python3 associations.py <king folder>  # correlation tables and heatmaps
```

Versions used: Python 3.12.5, PyTorch 2.11.0, scikit-learn 1.8.0, SciPy 1.16.2, NumPy 2.4.3, pandas 2.3.3, Matplotlib 3.10.8.

## 9. Where the files are

| Path | What it holds |
|---|---|
| `FEATURE_DECISIONS.md` | every column decision with its measured reason |
| `KING_COUNTY_SUMMARY.md` | this document |
| `king_county/kc_cards/` | reads the four files, applies the sale filter and the split (`measure.py`), and builds the feature-card page |
| `king_county/kc_check/` | the one-column-at-a-time measurements behind the decisions |
| `king_county/kc_features/features.py` | turns the decisions into features |
| `king_county/kc_features/pipeline.py` | the final table: encoding, scaling, correlation rules, folds |
| `king_county/kc_features/associations.py` | correlation tables and heatmaps for the report |
| `king_county/kc_features/train_models.py` | baselines, linear regression and the scoring functions |
| `king_county/kc_features/neural_search.py` | the five-fold neural-network search |
| `king_county/kc_features/out/` | every result table and heatmap; `neural_search/log.txt` has the error after every pass of every network |

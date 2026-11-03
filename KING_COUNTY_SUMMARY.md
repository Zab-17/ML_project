# King County house prices: what we did, how, and the results

Zeyad Khaled, 27 October 2026. Branch `zeyad`.

## The short version

We predict the price a house in King County, Washington (the Seattle area) sold for. The inputs are the county assessor's records of the house, its lot, its location and the sale date. After cleaning there are 532,890 sales from 2000 onwards. Each sale is described by 1,604 model columns built from 55 features.

Every model went through the same four steps:
1. a five-fold search for its best setting on the 426,135 training sales;
2. one final training on all of those sales;
3. one score on the 106,755 sealed test sales;
4. no choice made by looking at the test sales.

On the test sales, scored once per model:

| Model | Median miss | Average miss | Homes within 10% | Homes within 20% |
|---|---|---|---|---|
| **Neural network, 256 → 128 neurons** | **8.5% ($40,511)** | **11.4%** | **57.2%** | **85.3%** |
| Random forest | 9.7% ($46,872) | 13.8% | 51.2% | 77.6% |
| Bagging | 9.5% ($46,500) | 13.9% | 51.7% | 77.4% |
| Linear regression | 12.9% ($65,423) | 16.2% | 40.0% | 70.0% |
| Support vector machine | 12.7% ($64,725) | 16.2% | 40.8% | 70.2% |
| Decision tree | 13.9% ($66,876) | 18.5% | 37.2% | 65.9% |
| K-nearest neighbours | 16.8% ($79,786) | 22.2% | 31.4% | 57.6% |
| Predict the median price ($505,000) for every home | 41.9% ($206,050) | 53.2% | 11.5% | 23.2% |

How to read the table:
- Median miss: half the homes miss by less than this.
- Average miss: the mean absolute percentage error.
- The models are ordered by mean squared error on log price, the score each model was chosen on.

Findings:
- The neural network wins all 13 scores.
- Every model misses the most expensive 1% of homes by the most, and prices those homes too low (section 6.8).

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

The 426,135 training sales are split by home into 5 folds of about 85,000. Each setting of each model is trained 5 times, from scratch each time. Each run learns from 4 folds (about 341,000 sales) and is checked on the fifth, so every training home is checked exactly once by a model that never saw it. Every model uses the same five folds, so all models are compared on the same homes.

| Run | Learns from | Checked on |
|---|---|---|
| 1 | folds 1–4: 340,909 sales | fold 0: 85,226 |
| 2 | folds 0, 2–4: 340,863 | fold 1: 85,272 |
| 3 | folds 0, 1, 3, 4: 340,520 | fold 2: 85,615 |
| 4 | folds 0–2, 4: 341,198 | fold 3: 84,937 |
| 5 | folds 0–3: 341,050 | fold 4: 85,085 |

### 5.2 Zeyad's rule for every model

1. **Search:** every setting of the model trains 5 times, once per fold, as in 5.1.
2. **Winner:** the setting with the lowest average checking-fold mean squared error (log). A setting counts only if all 5 of its runs trained.
3. **Final training:** the winner trains once, from scratch, on all 426,135 training sales.
4. **Test:** that final model is scored once on the 106,755 test sales. The code refuses to score a model on the test sales a second time.

Why this order:
- No setting was picked by looking at the test sales, so the test sales stay a fair stand-in for homes nobody has seen.
- Training the final model on all 426,135 sales, instead of about 341,000, gives that model 25% more examples.

### 5.3 The models

- **Average baseline:** predicts $663,874, the average training price, for every home. Any model must beat this to be worth anything.
- **Median baseline:** predicts $505,000, the median training price, for every home.
- **Linear regression:** one weight per column, fitted by least squares. Each weight reads as "this column adds X% to the price".
- **Neural network:** the 1,604 columns feed a first layer of neurons, then a second, then one output, the predicted log price. "256 → 128" means 256 neurons in the first layer and 128 in the second. Twelve settings were tried:
  - three sizes (128 → 64, 256 → 128, 512 → 256);
  - each size with and without dropout 0.2;
  - each of those with two optimizers.

The neural network's training terms:

| Term | Meaning here |
|---|---|
| Pass (epoch) | every training sale seen once, in batches of 1,024 sales: about 333 weight updates per pass |
| Loss | mean squared error on log price, the number each update tries to shrink |
| Optimizer | the rule that turns the loss into weight changes. Stochastic gradient descent nudges every weight a small step downhill on the loss after each batch. Adam (Adaptive Moment Estimation) also adapts the step size of each weight to how that weight's slope has behaved so far |
| Learning rate | the size of the step: 0.1 for stochastic gradient descent (Zeyad's choice), 0.001 for Adam (its standard rate) |
| Dropout 0.2 | during training, each neuron is switched off at random 20% of the time, so the network can't lean on a few neurons and memorise |
| Early stopping | after every pass, score the checking fold. If 5 passes in a row bring no improvement, stop and keep the best pass. At most 100 passes |
| Hardware | the M1 Pro's graphics chip, through PyTorch |

Adam at learning rate 0.1 was tried first in a short test and blew up: its error was 0.37 after one pass and got worse. So the first search used stochastic gradient descent at 0.1. A second search then ran Adam at 0.001 on the same six networks (section 6.2).

**The other models from the course slides**, each through the same four steps:

| Model | Lecture | How the model prices a home | Settings searched |
|---|---|---|---|
| K-nearest neighbours | 3 | finds the k training sales closest to the home and averages their log prices | k = 1, 3, 5, 10, 15, 20, 30, 50, each with Euclidean or cosine distance: 16 settings |
| Decision tree (CART: Classification And Regression Trees) | 10 | asks yes/no questions about the columns (is the living area above some value? is the home in sub-area Y?) until the home reaches a leaf. The prediction is the average log price of the training sales in that leaf | pruning strength 0, 0.0000001, 0.000001, 0.000003, 0.00001, 0.00003, 0.0001: 7 settings |
| Bagging | 10 | 50 full decision trees, each trained on a bootstrap sample (426,135 sales drawn at random with replacement). The prediction is the average of the 50 trees | 1 setting |
| Random forest | 10 | bagging, except that each split may only choose from a random share of the columns | share of columns per split: the square root of 1,604 (40 columns), 10% or 33%: 3 settings |
| Support vector machine, linear, for regression | 2 | a straight-line model like linear regression, fitted with a different loss | margin penalty 0.01 or 1, each with margin width 0 or 0.05: 4 settings |

What the settings mean:
- **Euclidean distance** is the straight-line distance between two rows of columns. **Cosine distance** compares the angle between the two rows, so it compares the pattern of the columns and ignores the overall length of each row.
- K-nearest neighbours compares every checking sale with every training sale: 85,000 × 341,000 = 29 billion distances per fold. The distances are computed on the graphics chip.
- **Pruning strength** (scikit-learn's `ccp_alpha`, cost-complexity pruning): a branch of the tree stays only if the branch lowers the tree's squared error by at least this amount for every extra leaf the branch adds. 0 keeps the full tree; larger values cut more branches.
- **Why bagging and random forests average many trees.** One full tree follows its own training sales closely, so its predictions change a lot when those sales change. The average of 50 trees trained on different samples changes much less. In a random forest each split sees only some of the columns, so the trees differ more from each other, and their average gains more.
- **Margin width** (epsilon) in the support vector machine: misses smaller than this, in log price, cost nothing. 0.05 means misses under about 5% are free. Larger misses cost in proportion to their size, not their square, so a few wild sales pull the line less than in linear regression.
- **Margin penalty** (C): how much each sale outside the margin costs, against keeping the weights small.
- The forests use 50 trees, not 100, to keep the run time down.

Not run:
- **Logistic regression** (lecture 5), **the perceptron** (lecture 6), and **naive Bayes and Bayesian networks** (lecture 11). These models predict a class, not a number, so they cannot predict a price. They could run on price bands, which is a different task.
- **A support vector machine with a kernel** (curved instead of straight). That model compares every pair of training sales: 426,135 × 426,135 = 181 billion pairs, far more than 16 gigabytes of memory can hold.

### 5.4 The scores

| Score | Plain meaning | Better when |
|---|---|---|
| R-squared | share of the price differences between homes that the model explains. 1 = all of them, 0 = no better than predicting the average | higher |
| Mean squared error (log) | average squared miss on log price. The number every model was chosen on | lower |
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

### 6.2 Neural networks: both searches, averages over the five folds

"Gradient descent" below means stochastic gradient descent at learning rate 0.1; "Adam" means Adam at learning rate 0.001.

| Setting | Folds that trained | Mean squared error (log), average ± spread | R-squared (log) | Median percent off | Within 10% | Best pass (median) | Minutes for 5 runs |
|---|---|---|---|---|---|---|---|
| 512 → 256, gradient descent, no dropout ⚠️ | 4 of 5 | 0.0192 ± 0.0018 | 0.954 | 7.9% | 60.4% | 28 | 5.5 |
| 512 → 256, gradient descent, dropout 0.2 ⚠️ | 4 of 5 | 0.0203 ± 0.0046 | 0.951 | 8.0% | 59.9% | 49 | 10.6 |
| **256 → 128, gradient descent, dropout 0.2 (winner)** | **5 of 5** | **0.0204 ± 0.0031** | **0.951** | **8.1%** | **59.4%** | **56** | **9.4** |
| 128 → 64, gradient descent, no dropout | 5 of 5 | 0.0205 ± 0.0014 | 0.951 | 8.2% | 58.4% | 18 | 3.6 |
| 128 → 64, Adam, no dropout | 5 of 5 | 0.0211 ± 0.0012 | 0.949 | 8.2% | 58.2% | 29 | 4.9 |
| 256 → 128, gradient descent, no dropout | 5 of 5 | 0.0218 ± 0.0064 | 0.947 | 8.4% | 57.8% | 27 | 4.4 |
| 128 → 64, gradient descent, dropout 0.2 | 5 of 5 | 0.0231 ± 0.0043 | 0.944 | 8.7% | 56.0% | 49 | 7.2 |
| 256 → 128, Adam, no dropout | 5 of 5 | 0.0265 ± 0.0062 | 0.936 | 9.4% | 53.2% | 29 | 5.4 |
| 256 → 128, Adam, dropout 0.2 | 5 of 5 | 0.0297 ± 0.0105 | 0.928 | 10.3% | 50.2% | 38 | 8.5 |
| 512 → 256, Adam, no dropout | 5 of 5 | 0.0322 ± 0.0069 | 0.922 | 10.6% | 48.0% | 37 | 7.6 |
| 128 → 64, Adam, dropout 0.2 | 5 of 5 | 0.0357 ± 0.0106 | 0.914 | 11.6% | 44.9% | 28 | 5.5 |
| 512 → 256, Adam, dropout 0.2 | 5 of 5 | 0.0390 ± 0.0231 | 0.906 | 11.5% | 47.0% | 39 | 8.2 |
| *Linear regression* | *5 of 5* | *0.0412 ± 0.0002* | *0.901* | *12.9%* | *39.9%* | — | *under 1* |

⚠️ Both 512 → 256 gradient-descent settings blew up on fold 1: their error became infinite on the first pass. Their averages cover the other 4 folds only.

- **The winner is 256 → 128 with dropout 0.2, trained by gradient descent.** The two 512 → 256 gradient-descent settings score lower, but each lost one of its five runs, so by the rule in 5.2 they do not count. An average over the four surviving runs hides the run that failed.
- **Adam at 0.001 did not beat gradient descent.** Adam's best setting, 128 → 64 without dropout, is the steadiest network of all (spread 0.0012), but its error sits 0.0007 above the winner's. Adam's other five settings are worse and uneven (spreads 0.006 to 0.023). Every Adam run trained, including both 512 → 256 runs on fold 1.

### 6.3 The five runs of the winning setting (256 → 128, dropout 0.2)

| Checked on | Best pass | Passes run | Mean squared error (log) | R-squared (log) | Median dollars off | Median percent off | Within 10% | Within 20% |
|---|---|---|---|---|---|---|---|---|
| Fold 0 | 77 | 82 | 0.0184 | 0.956 | $36,276 | 7.5% | 62.4% | 88.9% |
| Fold 1 | 86 | 91 | 0.0192 | 0.953 | $36,819 | 7.6% | 61.4% | 88.6% |
| Fold 2 | 56 | 61 | 0.0176 | 0.958 | $35,162 | 7.3% | 63.1% | 89.3% |
| Fold 3 | 25 | 30 | 0.0219 | 0.947 | $40,570 | 8.4% | 57.4% | 86.1% |
| Fold 4 | 21 | 26 | 0.0251 | 0.940 | $44,855 | 9.4% | 52.6% | 83.5% |

- The runs that trained longest scored best. The runs on folds 3 and 4 stopped at passes 25 and 21 and scored worst (section 6.9).
- **Final training:** the winner trained from fresh weights on all 426,135 training sales for 56 passes, the median best pass of these five runs. No fold is left to check on, so the search decides how long to train. The training took 166 seconds. The weights are in `out/final_model/network.pt`.

### 6.4 Every score on the test set (106,755 sales, one score per model)

| Score | Neural network | Random forest | Bagging | Linear regression | Support vector machine | Decision tree | K-nearest neighbours | Median baseline |
|---|---|---|---|---|---|---|---|---|
| R-squared (log) | **0.947** | 0.912 | 0.910 | 0.901 | 0.900 | 0.860 | 0.808 | −0.006 |
| Mean squared error (log) | **0.0221** | 0.0363 | 0.0374 | 0.0411 | 0.0414 | 0.0582 | 0.0798 | 0.4171 |
| Root mean squared error (log) | **0.149** | 0.191 | 0.193 | 0.203 | 0.203 | 0.241 | 0.283 | 0.646 |
| Mean absolute error (log) | **0.108** | 0.136 | 0.136 | 0.158 | 0.157 | 0.180 | 0.214 | 0.515 |
| R-squared (dollars) | **0.924** | 0.870 | 0.872 | 0.885 | 0.883 | 0.795 | 0.715 | −0.077 |
| Root mean squared error (dollars) | **$161,546** | $211,017 | $209,399 | $197,716 | $200,266 | $264,791 | $311,896 | $606,267 |
| Mean absolute error (dollars) | **$76,884** | $98,837 | $98,950 | $106,962 | $107,293 | $129,202 | $154,430 | $334,966 |
| Median dollars off | **$40,511** | $46,872 | $46,500 | $65,423 | $64,725 | $66,876 | $79,786 | $206,050 |
| Mean absolute percentage error | **11.4%** | 13.8% | 13.9% | 16.2% | 16.2% | 18.5% | 22.2% | 53.2% |
| Median percent off | **8.5%** | 9.7% | 9.5% | 12.9% | 12.7% | 13.9% | 16.8% | 41.9% |
| Within 5% | **31.4%** | 29.6% | 30.2% | 20.6% | 21.5% | 19.4% | 16.1% | 5.6% |
| Within 10% | **57.2%** | 51.2% | 51.7% | 40.0% | 40.8% | 37.2% | 31.4% | 11.5% |
| Within 20% | **85.3%** | 77.6% | 77.4% | 70.0% | 70.2% | 65.9% | 57.6% | 23.2% |

The average baseline ($663,874 for every home) scores mean squared error 0.4661 (log) and misses the median home by 51.0%.

Every model on the test set, the two baselines and linear regression included, was fitted on all 426,135 training sales.

### 6.5 Did the search predict the test?

| Model | Winning setting | Checking folds: mean squared error (log), average ± spread | Test: mean squared error (log) |
|---|---|---|---|
| Neural network | 256 → 128, dropout 0.2, gradient descent, 56 passes | 0.0204 ± 0.0031 | 0.0221 |
| Random forest | 50 trees, 33% of the columns per split | 0.0381 ± 0.0005 | 0.0363 |
| Bagging | 50 trees, every column | 0.0397 ± 0.0005 | 0.0374 |
| Linear regression | — | 0.0412 ± 0.0002 | 0.0411 |
| Support vector machine | margin penalty 1, margin width 0.05 | 0.0416 ± 0.0002 | 0.0414 |
| Decision tree | pruning strength 0.000003 | 0.0595 ± 0.0007 | 0.0582 |
| K-nearest neighbours | cosine distance, k = 15 | 0.0820 ± 0.0008 | 0.0798 |

- **The models finish in the same order on the test sales as in the search.**
- **Six of the seven models score a little better on the test sales than in the search.** The likely reason is that each final model learned from 426,135 sales instead of about 341,000.
  - The trees and K-nearest neighbours gain the most (0.0013 to 0.0023). These models split on, or look up, the training sales directly, so more sales mean finer leaves and closer neighbours.
  - The two straight-line models gain the least (0.0001 to 0.0002). A line with 1,604 weights is already settled at 341,000 sales.
- **The network scores worse on the test sales than in the search:** 0.0221 against 0.0204.
  - The gap is inside the network's fold-to-fold spread (± 0.0031), so the gap is no sign of memorising.
  - A likely cause, not yet tested: the final training cannot stop early. At learning rate 0.1 the training error kept bouncing between 0.0188 and 0.0215 over the last ten passes, so the weights after pass 56 are wherever the last bounce left them.

### 6.6 What each model's search showed

**K-nearest neighbours.** k = 15 with cosine distance won (0.0820).
- One neighbour is the worst setting (0.139 with Euclidean distance, 0.145 with cosine), because a single sale's price is noisy.
- Above k = 15 the error rises again (0.0857 at k = 50), as homes further away get averaged in.
- Cosine distance beats Euclidean distance from k = 5 up (0.0820 against 0.0849 at k = 15). Euclidean distance wins at k = 1 and k = 3.

**Decision tree.** The full tree scores 0.0736.
- Light pruning helps: 0.0598 at strength 0.000001, and 0.0595 at 0.000003.
- Heavier pruning hurts: 0.0672 at 0.00001, and 0.0883 at 0.0001.
- The full tree keeps splitting until its leaves hold very few sales, so it learns their noise. Heavy pruning leaves too few leaves to tell neighbourhoods apart.

**Random forest.** 33% of the columns per split won (0.0381), ahead of 10% (0.0401) and the square root, 40 columns (0.0529).
- A likely reason the square root does badly: most of the 1,604 columns are location columns that are 0 for almost every home, so a random 40 columns often hold no column worth splitting on.
- Bagging, which offers every column at every split, scores 0.0397. The random share of columns helps, but only a little.

**Support vector machine.** All four settings land between 0.0416 and 0.0419, next to linear regression (0.0412).
- Both are straight-line models of the same 1,604 columns, so the different loss changes little.
- The runs with margin penalty 1 took about 150 seconds per fit, against 12 seconds at 0.01. Some of the penalty-1 runs did not fully settle within 2,000 passes (11 warnings). Those runs still scored within 0.0003 of the rest, so the result stands.

### 6.7 What the results mean

- **The network halves linear regression's squared error on the test sales** (0.0221 against 0.0411) and wins all 13 scores. The network can learn combinations of columns, for example a different value for the same view in different areas. Linear regression and the support vector machine give each column one fixed effect everywhere.
- **The tree ensembles come second.** Random forest and bagging also learn combinations of columns, through their splits, and cut linear regression's error by 9 to 12%. Their median miss (9.5 to 9.7%) is close to the network's (8.5%).
- **One tree is worse than a straight line** (0.0582 against 0.0411). A single tree's prediction jumps between leaf averages. Averaging 50 trees is what makes trees competitive.
- **K-nearest neighbours comes last** (0.0798). A likely reason: every one of the 1,604 columns counts equally in the distance, so a match on an unimportant column counts as much as a match on living area.
- **On a typical home the network is 8.5% off.** 57% of the test homes land within 10% of the sale price, and 85% within 20%. Linear regression manages 40% and 70%.
- **Dollar scores look worse than percent scores because of expensive homes.** An 8% miss on a $3,000,000 house is $240,000. That is why the network's root mean squared error in dollars ($161,546) is four times its median dollar miss ($40,511).
- **Linear regression still wins on three things:** steadiness (spread 0.0002), speed (about 5 seconds per fit) and readability (one weight per column).

### 6.8 The most expensive 1% of homes

The most expensive 1% of the test sales are 1,068 sales, from $2,942,000 to $28,500,000.

| Model | Median miss on the top 1% | Share of all squared dollar error that comes from the top 1% | Root mean squared error (dollars), top 1% | Root mean squared error (dollars), other 99% |
|---|---|---|---|---|
| Neural network | 15.8% too low | 47.6% | $1,113,940 | $117,565 |
| Linear regression | 17.2% too low | 38.5% | $1,227,327 | $155,772 |
| Random forest | 20.3% too low | 45.0% | $1,414,769 | $157,327 |
| Bagging | 18.1% too low | 41.9% | $1,354,724 | $160,452 |
| Support vector machine | 18.1% too low | 38.8% | $1,246,818 | $157,488 |
| Decision tree | 22.1% too low | 41.6% | $1,706,640 | $203,445 |
| K-nearest neighbours | 30.4% too low | 40.7% | $1,990,472 | $241,299 |

- **Every model prices the most expensive homes too low.** Few training sales sit at those prices. On log price, a 20% miss on a $5,000,000 home counts the same as a 20% miss on a $500,000 home, even though the first miss costs ten times as many dollars.
- **These 1,068 sales cause 38% to 48% of all squared dollar error.** That is why the dollar R-squared ranks the models differently from the log scores. The random forest scores 0.870 in dollars against linear regression's 0.885, even though the forest is better on log price.
  - On the other 99% of homes the two models are level ($157,327 against $155,772).
  - On the top 1% the forest misses by more ($1,414,769 against $1,227,327).
- **A likely reason the tree models and K-nearest neighbours fall furthest short at the top, not yet tested:**
  - A tree's prediction is the average log price of the training sales in one leaf. So a tree can never predict above the most expensive sale in that leaf, and a leaf that mixes a mansion with ordinary homes pulls the mansion's price down.
  - K-nearest neighbours averages in the same way.
  - A straight line can keep rising with the living area.

### 6.9 Problems found

- **Learning rate 0.1 is too jumpy for the larger networks.** Both 512 → 256 runs on fold 1 blew up on the first pass.
- **Early stopping cuts some runs short.**
  - The winning setting stopped at passes 25 and 21 on folds 3 and 4, and scored worst there: 0.0219 and 0.0251, against 0.0176 to 0.0192 on folds 0 to 2.
  - Linear regression finds folds 3 and 4 as easy as the other folds, so the homes are not harder.
  - The error bounces at learning rate 0.1, so 5 passes without improvement can come from one bounce rather than from the end of learning.
  - Adam at 0.001 did not solve the problem overall (section 6.2).
  - Untested options: wait 10 passes instead of 5, or shrink the step size as training goes on.
- **The slide-model run took 7.6 hours, three times the 2.5-hour estimate.**
  - Each pruned tree took about 6 minutes to fit, instead of 40 seconds for the full tree, because pruning removes the weakest branch one at a time from a very large tree.
  - K-nearest neighbours ran in the same program after the trees and grew to 13 gigabytes on a 16-gigabyte laptop. The laptop moved memory to disk, which slowed each fold from about 7 minutes to 25 to 35 minutes.
  - Next time: time one fit before sizing a search, and run each heavy model in its own program (`python3 slide_models.py "k-nearest neighbours"`).

### 6.10 Time on the M1 Pro

| Step | Search | Final fit |
|---|---|---|
| Neural networks, gradient descent (30 runs) | 41 minutes (25 October, 23:01 to 23:42) | — |
| Neural networks, Adam (30 runs) | 40 minutes | — |
| Final network, 56 passes on 426,135 sales | — | 166 seconds |
| Decision tree (35 fits) | 3.4 hours | 10.5 minutes |
| K-nearest neighbours (5 folds × 2 distances) | 2.3 hours | 27 minutes |
| Random forest (15 fits) | 14 minutes | 2.2 minutes |
| Support vector machine (20 fits) | 27 minutes | 3.3 minutes |
| Bagging (5 fits) | 21 minutes | 5.4 minutes |
| **All slide models together** | **7.6 hours (26 October, about 15:14 to 22:50)** | |

---

## 7. What comes next

The four steps of the old plan are done: the Adam search, the choice of the winning network, the final training, and one test score per model.

Still open:
- a "new house" yes/no feature (sold within a year of being built), measured at +3.9 points of price explained and not yet decided. Adding the feature now changes the table, so every model would need a new search, and the test sales would be scored a second time. If that happens, the report has to say so;
- a check on the 211 very cheap sales (under $89,756);
- the report figures: the search tables, the test table and the top-1% table;
- telling the professor about the switch to King County;
- the README still describes the Zillow dataset;
- optional: logistic regression, the perceptron and naive Bayes on price bands.

## 8. Running it again

The King County files are not in this repository: they come to about 1 gigabyte. Download the four files listed in section 1 into one folder, then:

```bash
pip install numpy pandas scipy scikit-learn torch matplotlib
cd king_county/kc_features
python3 pipeline.py <king folder>        # about 1 minute: builds out/model_table/
python3 train_models.py                  # baselines and linear regression, seconds
python3 neural_search.py --smoke         # short test of the network code
python3 neural_search.py                 # the five-fold search, both optimizers: about 81 minutes on an M1 Pro
python3 final_training.py                # picks the winner, trains it on all training sales, scores the test set: about 3 minutes
python3 slide_models.py                  # every other course model: about 7.6 hours
python3 slide_models.py "random forest"  # or one model at a time, which uses less memory
python3 associations.py <king folder>    # correlation tables and heatmaps
```

`slide_models.py` will not score a model on the test sales twice. A model that already has a row in `out/slide_models/test_results.csv` is skipped.

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
| `king_county/kc_features/neural_search.py` | the five-fold neural-network search, both optimizers |
| `king_county/kc_features/final_training.py` | picks the winning network, trains it on all training sales, scores it once on the test sales |
| `king_county/kc_features/slide_models.py` | K-nearest neighbours, decision tree, random forest, bagging and the support vector machine: search, final fit, one test score |
| `king_county/kc_features/out/neural_search/` | every network run (`fold_results.csv`), the averages (`summary.csv`), and the error after every pass (`log.txt`) |
| `king_county/kc_features/out/final_model/` | the final network's weights (`network.pt`), its setting (`choice.json`) and its test scores (`test_results.csv`) |
| `king_county/kc_features/out/slide_models/` | every setting on every fold (`fold_results.csv`), every model on the test sales (`all_test_results.csv`) and the run log. The saved test predictions (`test_predictions/`) stay on the laptop and are not in git |

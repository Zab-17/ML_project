# Handoff: writing the Milestone 3 report

For the AI assistant helping Mohamed finish the Milestone 3 (Phase III) report. Read this whole file before touching the report. The training is finished. **Your job is the writing, not the experiments.**

Project: CSCE 3602 / DSCI 3415, The American University in Cairo. Team: Zeyad Khaled and Mohamed Elsayed. Repository: `https://github.com/Zab-17/ML_project`, branch `main`.

---

## 1. Rules you must follow

1. **Zeyad and Mohamed make every decision.** Propose, then wait for a yes before you change a method, a claim, a table layout or a number. Finding a problem is not permission to fix it.
2. **Do not run the models.** Every result is already saved, and the full pipeline takes many hours. **Never run `zillow/final_training.py`:** it has no skip, so a rerun retrains the network and changes the headline result.
3. **Every number comes from a file in `zillow/out/`** (Section 4). Never estimate, round differently from the tables, or invent a value. If a number you need is not in a file, say so.
4. **Write like a person, not like an AI.** The team screens documents with an AI-writing checker. Vary sentence length. Avoid templates such as "it is important to note", "plays a crucial role" or "in conclusion". Use no quotations as decoration, and treat any word or page limit as hard.
5. **Name what is measured and what is a guess.** Every "why" explanation that was not tested must be worded as likely, not as fact. Section 5 marks which is which.
6. **Cite every method, library and data source where it is used.** Missing citations and missing AI statements were both marked down in Phases 1 and 2. The bibliography is `milestone_3/report/references.bib`.

## 2. What the project is

- **Data:** the Zillow Prize 2017 properties file, with 2,985,217 parcels and 58 columns in Los Angeles, Orange and Ventura counties.
- **Label:** `taxvaluedollarcnt`, the county's 2016 assessed value. Models predict its natural log.
- **Homes:** row rules keep 2,859,302 homes. They are split 80/20 with seed 42 into 2,287,442 training homes and 571,860 test homes. Five folds for the search come from a hash of the parcel number.
- **Features:** 17 features become 606 model columns: 7 standardized numbers, 2 yes/no columns and 597 one-hot columns.
- **Protocol, the same for every model:**
  1. a five-fold search for the best setting on the training homes;
  2. the winner chosen by the lowest average validation MSE (log);
  3. one final fit on all training homes;
  4. one score on the sealed test homes.

  No choice is made by looking at the test homes.
- **Price bands:** the three classifiers (logistic regression, the perceptron and naive Bayes) predict one of five price bands instead of a number. Each band holds a fifth of the training homes: under $165,080, $165,080–$269,041, $269,041–$387,279, $387,279–$580,090, and over $580,090. The number models are cut at the same edges so all models share one band table.
- **What changed since Phase 2, and why:** `milestone_3/professor_comments.md` (one row per professor comment) and `changes/after_milestone_2.md` (all 12 fixes).

## 3. The state of the report

File: `milestone_3/report/main.tex` (ACM format), with `references.bib` and `figures/` next to it. `main.pdf` is the current build: 18 pages, compiled with tectonic, no errors.

**The report is complete.** No red `\X` number or `\pending` note is left. The authors are Zeyad Khaled and Mohamed Elsayed, and the "Use of AI" statement is confirmed.

| The brief asks for | Where it is |
|---|---|
| A report in ACM format | `milestone_3/report/main.pdf` |
| Classification on a binned label (regression projects) | Five price bands: seven classifiers trained on the bands, plus every regression model cut into the bands |
| For each model: setup, parameter choice, evaluation | Section 4, one subsection per model, and the settings table |
| A comment on each model's performance and fit | The "Verdict" paragraph that ends every model subsection |
| The top 3 compared, with the variability discussed | Section 6 |
| A choice from the top 3 | Section 7: the neural network |
| The source code | `zillow/` (the pipeline) and `training_code/` (one file per model) |

- **Regression or classifier form:** Section 3, "Regression and classifier forms", says which models ran in which form.
- **Not run** (stated in the Limitations section): the classifier forms of bagging, AdaBoost and voting, and the ID3 tree of Lecture 9.
- **Still to decide before submitting:** how the code reaches the professor. Either zip `zillow/`, `src/` and `training_code/` without the data files, or link the GitHub repository. The repository is public, and `data/processed/` holds Zillow rows that the licence forbids sharing.

## 4. Where every number lives

Every model also has its own results page in `training_code/results/`, with every metric. `milestone_3/all_training_results.md` collects them in one file.

| File | What it holds |
|---|---|
| `zillow/out/results/all_test_results.csv` | Every model's test scores: R² and MSE, RMSE and MAE on log value; R², RMSE and MAE in dollars; median dollar and percent miss; MAPE (average percent miss); share within 5%, 10% and 20% |
| `zillow/out/results/band_results.csv` | Exact-band accuracy, macro F1 and within-one-band share for every model |
| `zillow/out/results/top_one_percent.csv` | The most expensive 1% of test homes (5,719): median miss and the share of squared dollar error |
| `zillow/out/results/validation_ranking_used_for_voting.csv` | The three models voting averages, chosen by validation MSE (log) |
| `zillow/out/slide_models/fold_results.csv` | Every model, setting and fold of the search, with every score: use for the per-model sections and the variability section |
| `zillow/out/slide_models/test_results.csv` | The winning setting of each slide model |
| `zillow/out/neural_search/` | The network search: `fold_results.csv`, `summary.csv`, `linear_regression_folds.csv` |
| `zillow/out/final_model/choice.json` | The final network: 256 → 128 neurons, Adam, learning rate 0.001, dropout 0.2, 27 passes, 272 seconds |
| `zillow/out/band_models/` | Band edges, the classifiers' search and their test scores |
| `zillow/out/checks/` | The column check, the association tables and the heatmaps |
| `zillow/out/run_log*.txt` | Run times |
| `milestone_3/report/figures/distribution_stats.json` | The skew numbers the distribution figure shows |

### The test results (571,860 homes)

Median miss: half the homes are off by less than this. Average miss: the mean absolute percentage error, which a few huge misses pull up.

| Model | Chosen setting | Test MSE (log) | Test R² (log) | Median miss | Average miss | Within 10% |
|---|---|---|---|---|---|---|
| neural network | 256-128, Adam, learning rate 0.001, dropout 0.2 | 0.3725 | 0.467 | 32.5% | 59.2% | 15.8% |
| voting (neural network + random forest + k-nearest neighbours) | — | 0.3731 | 0.466 | 32.5% | 58.9% | 16.1% |
| random forest | 10% of columns per split | 0.3788 | 0.458 | 32.6% | 59.5% | 16.2% |
| k-nearest neighbours | Euclidean, k = 50 | 0.3839 | 0.451 | 33.2% | 59.6% | 15.7% |
| linear regression | — | 0.3944 | 0.436 | 33.8% | 62.0% | 14.9% |
| bagging | no search | 0.3953 | 0.435 | 33.2% | 60.5% | 16.3% |
| decision tree | smallest leaf 200 homes | 0.4109 | 0.412 | 34.7% | 62.2% | 14.9% |
| support vector machine | margin penalty 0.01, margin width 0.05 | 0.4120 | 0.411 | 31.6% | 69.8% | 16.3% |
| k-means clustering | 1,000 clusters | 0.4532 | 0.352 | 34.6% | 71.8% | 15.1% |
| hierarchical clustering | 1,000 clusters | 0.4556 | 0.349 | 34.7% | 72.1% | 15.1% |
| AdaBoost | tree depth 8 | 0.4903 | 0.299 | 42.4% | 59.3% | 10.9% |
| median baseline | — | 0.7021 | −0.004 | 45.4% | 91.1% | 11.2% |
| average baseline | — | 0.8278 | −0.184 | 53.8% | 129.6% | 10.8% |

### The price bands

| Model | Kind | Exact band | Macro F1 | Within one band |
|---|---|---|---|---|
| logistic regression (C = 1.0) | band model | 41.2% | 0.398 | 75.3% |
| neural network | number model, cut into bands | 38.8% | 0.394 | 81.8% |
| voting | number model, cut into bands | 38.8% | 0.394 | 81.8% |
| bagging | number model, cut into bands | 38.7% | 0.395 | 80.6% |
| random forest | number model, cut into bands | 38.7% | 0.393 | 81.6% |
| support vector machine | number model, cut into bands | 38.7% | 0.383 | 82.4% |
| naive Bayes (smoothing 0.1) | band model | 38.5% | 0.362 | 70.6% |
| k-nearest neighbours | number model, cut into bands | 38.3% | 0.390 | 81.1% |
| linear regression | number model, cut into bands | 37.3% | 0.379 | 80.7% |
| decision tree | number model, cut into bands | 36.7% | 0.371 | 80.0% |
| perceptron (5 passes) | band model | 34.9% | 0.339 | 71.7% |
| k-means clustering | number model, cut into bands | 34.8% | 0.335 | 79.7% |
| hierarchical clustering | number model, cut into bands | 34.7% | 0.333 | 79.8% |
| AdaBoost | number model, cut into bands | 30.6% | 0.300 | 70.8% |
| average baseline | number model, cut into bands | 20.1% | 0.067 | 60.1% |
| median baseline | number model, cut into bands | 20.0% | 0.067 | 60.0% |

### The classifiers added on 5 November (as the slides teach them)

| Model | Lecture | Trained on | Winning setting | Test accuracy | Macro F1 | Within one band |
|---|---|---|---|---|---|---|
| Bayesian network | 11 | all homes | band's parents: ZIP code, living area, building quality, year built | 42.63% | 0.411 | 74.9% |
| Mixture of experts | 13 | 100,000-home sample | 8 experts | 42.25% | 0.411 | 76.3% |
| K-nearest neighbours, majority vote | 3 | 100,000-home sample | k = 30, Euclidean | 39.57% | 0.386 | 73.2% |
| Linear SVM classifier | 12 | 20,000-home sample | C = 0.1 | 34.15% | 0.294 | 64.4% |

### The most expensive 1% of homes (5,719 test homes)

| Model | Median miss (minus = too low) | Share of all squared dollar error |
|---|---|---|
| neural network | −44.8% | 68.1% |
| voting | −46.7% | 71.1% |
| random forest | −48.3% | 72.2% |
| k-nearest neighbours | −47.2% | 70.7% |
| linear regression | −50.5% | 70.9% |
| bagging | −44.3% | 67.5% |
| decision tree | −53.5% | 70.1% |
| support vector machine | −47.9% | 71.7% |
| k-means clustering | −61.8% | 73.2% |
| hierarchical clustering | −61.4% | 73.2% |
| AdaBoost | −61.6% | 66.8% |

## 5. What the results show

Each point is marked **measured** (it is in the files) or **likely** (an explanation nobody has tested).

1. **Voting did not beat its best member** (measured): 0.3731 against the network's 0.3725. Averaging pulled the network's score toward the two weaker members. Averaging helps most when the members make different mistakes. Here all three miss the same homes (likely; the cause is point 6).
2. **Random forest beat bagging** (measured): 0.3788 against 0.3953. Both average many trees grown on resampled homes. The forest also lets each split choose from only a random 10% of the columns, so its trees are less alike and their errors cancel more.
3. **The model choice matters little on this file** (measured): the top eight models are within 11% of each other on MSE (log).
4. **The support vector machine has the best median miss but a poor average miss** (measured): 31.6% median, 69.8% average. Its absolute-error loss fits the middle home, the median, rather than the mean, so it wins the median score and loses the squared-error score.
5. **Every model prices the most expensive 1% of homes far too low** (measured): by 44% to 62% at the median. These 5,719 homes cause 67% to 73% of all squared dollar error.
6. **Why every model plateaus around 0.37** (likely): the label. Under California's Proposition 13, a home's assessed value is set at its last sale and then rises at most 2% a year. Two identical neighbouring houses, one bought in 1985 and one in 2015, carry very different labels, and no column says when a home last sold. The same explanation fits the cheapest band: 24% of band-1 homes are predicted as band 4 or 5.
7. **Logistic regression picks the exact band best** (measured), at 41.2%. The number models cut into bands are more often off by at most one band: about 82%, against 75%.
8. **AdaBoost is the weakest model above the baselines** (measured), at 0.4903. A likely reason is that boosting focuses on the hardest homes, and here many of those are homes whose assessed value lags their market value, so it chases noise.

9. **The Bayesian network is the best band model** (measured): 42.6%, ahead of the mixture of experts (42.3%) and logistic regression (41.2%). Each parent added to the band helped, and year built helped more than building quality.
10. **The mixture of experts beat logistic regression with a twentieth of the homes** (measured). Accuracy rose with every expert added: 41.31%, 41.58% and 42.06% for 2, 4 and 8 experts. A likely reason is that experts can specialise in different parts of the market.

## 6. Limits the report must state

- The label is an assessed value, not a sale price (point 6 above).
- The Zillow licence allows use only during a competition that closed on 10 January 2018, and the professor has not been told. Zeyad and Mohamed decide how to raise it.
- Fine location is lost. The census block (about 94,000 values) is the strongest location column, but it is too fine to one-hot encode. The usual fix, target encoding, was not used.

## 7. Building the report

Compile `milestone_3/report/main.tex` with LaTeX and BibTeX, either in Overleaf or locally with `pdflatex`, `bibtex`, `pdflatex`, `pdflatex`. Then search the PDF for red text: every `\X` and `\pending` must be gone before submission.

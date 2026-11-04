# Machine Learning class project: CSCE 3602 / DSCI 3415

**Predicting the 2016 assessed value of a home** from the Zillow Prize 2017 properties file. The file has 2,985,217 parcels and 58 columns, covering Los Angeles, Orange and Ventura counties, California. The label is `taxvaluedollarcnt`, the value the county assessor places on the parcel.

Zeyad Khaled and Mohamed Elsayed, The American University in Cairo.

**Writing the Milestone 3 report? Start with [`HANDOFF.md`](HANDOFF.md).**

---

## Where everything is

| Path | What it is |
|---|---|
| [`HANDOFF.md`](HANDOFF.md) | Everything needed to finish the Milestone 3 report: its state, where every number is, the rules |
| [`changes/`](changes/) | What changed after each submission, and why. One file per milestone. |
| [`milestone_2/`](milestone_2/) | Milestone 2 (Phase 2, data preparation) as submitted |
| `milestone_2/report/` | The Phase 2 report (`main.tex`, `main.pdf`) and its seven figures |
| `milestone_2/REFERENCE.md` | Every Phase 2 cleaning decision, with its number |
| `milestone_2/feature_analysis/` | Mohamed's feature analysis (`changes.md`, `changes.txt`) and his scratch script `test.py` |
| [`milestone_3/`](milestone_3/) | Milestone 3 (Phase III, the supervised models) |
| `milestone_3/professor_comments.md` | **Read first:** each professor comment, what we changed, and the number behind it |
| `milestone_3/report/` | The Phase III report (`main.tex`, `references.bib`, `figures/`) |
| `src/` | The Phase 2 cleaning pipeline (`clean.py`, `figures.py`, `column_profile.py`), and `code_tables.py`, which turns Zillow's numeric codes into names. Milestone 3 still uses `code_tables.py`. |
| `zillow/` | The Milestone 3 pipeline: row rules, features, checks, the model table, every model, voting and the result tables |
| `zillow/out/` | Every Milestone 3 result. The final tables are in `zillow/out/results/`. |
| `data/processed/` | The Phase 2 cleaned output |
| `data/raw/` | Not in git. Put `properties_2017.csv` and `zillow_data_dictionary.xlsx` from the Zillow Prize Kaggle page here. |

## Branches

| Branch | What it holds |
|---|---|
| `main` | The Zillow project only |
| `greater-cairo` | Milestone 1: the Greater Cairo version, the rejected Phase 1 report and the Egyptian developer tables |
| `king-county` | The King County (Seattle area) sale-price project. The Zillow pipeline copies its checks and models. The branch also has its experimental report. |

## Running it

```bash
python3 src/clean.py        # Milestone 2 cleaning, about 20 seconds
python3 src/figures.py      # Milestone 2 figures, about 13 seconds
zsh zillow/run_all.sh       # Milestone 3, every step in order: many hours on an M1 Pro laptop
cd zillow && python3 -m pytest test_zillow.py   # unit tests
```

- Each Milestone 3 step skips work it has already finished, so after a stop the same command picks up where it left off.
- The model matrices, labels and predictions (`.npy` and `.npz`) are not in git. `run_all.sh` rebuilds them.
- **Never rerun `zillow/final_training.py` on its own.** It has no skip, so a rerun would retrain the network and change the reported result.

## Milestone 3 results in one table

Test homes: 571,860, scored once per model. MSE (log) is the mean squared error on the natural log of the assessed value; lower is better.

| Model | Test MSE (log) | Median miss |
|---|---|---|
| Neural network (256 → 128, Adam, dropout 0.2) | **0.3725** | 32.5% |
| Voting (neural network + random forest + k-nearest neighbours) | 0.3731 | 32.5% |
| Random forest | 0.3788 | 32.6% |
| Linear regression | 0.3944 | 33.8% |
| Median baseline | 0.7021 | 45.4% |

Every model, the price bands and the most expensive 1% of homes are in `HANDOFF.md` and `zillow/out/results/`.

## Use of AI

AI (Claude, by Anthropic) helped write and debug this code. Every code file carries the statement "AI was used to help write this code, but the resulting code was reviewed and edited by a human."

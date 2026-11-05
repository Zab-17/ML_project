# AdaBoost.R2, regression form (Lecture 13)

50 rounds of regression trees, each round resampling the homes by the previous round's errors; the prediction is the weighted median.

Regression on the natural log of the 2016 assessed value. Trained on all 2,287,442 training homes. Tested once on all 571,860 test homes.

## Search: average over the five validation folds

| Setting | Folds | MSE spread | R² (log) | MSE (log) | RMSE (log) | MAE (log) | R² ($) | RMSE ($) | MAE ($) | Median $ miss | Average % miss | Median % miss | Within 5% | Within 10% | Within 20% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tree depth 8 | 5 | 0.0048 | 0.3081 | 0.4843 | 0.6959 | 0.5663 | 0.3940 | 470,419 | 209,986 | 127,474 | 59.39 | 41.98 | 5.58 | 11.23 | 22.80 |
| tree depth 4 | 5 | 0.0085 | 0.2833 | 0.5017 | 0.7083 | 0.5669 | 0.2988 | 506,011 | 216,925 | 130,966 | 63.03 | 41.56 | 5.86 | 11.73 | 23.49 |

## Test (571,860 homes)

| Model | R² (log) | MSE (log) | RMSE (log) | MAE (log) | R² ($) | RMSE ($) | MAE ($) | Median $ miss | Average % miss | Median % miss | Within 5% | Within 10% | Within 20% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| AdaBoost | 0.2989 | 0.4903 | 0.7002 | 0.5715 | 0.3871 | 483,984 | 212,254 | 127,382 | 59.27 | 42.45 | 5.39 | 10.94 | 22.27 |

Winning setting: tree depth 8; final fit and test prediction 1,821 seconds.

## The same test predictions cut into the five price bands

| Model | Accuracy (%) | Macro F1 | Within one band (%) |
|---|---|---|---|
| AdaBoost | 30.62 | 0.3004 | 70.77 |


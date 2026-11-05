# Linear regression (Lecture 4)

Ordinary least squares on all 606 columns.

Regression on the natural log of the 2016 assessed value. Trained on all 2,287,442 training homes. Tested once on all 571,860 test homes.

## Search: average over the five validation folds

| Setting | Folds | MSE spread | R² (log) | MSE (log) | RMSE (log) | MAE (log) | R² ($) | RMSE ($) | MAE ($) | Median $ miss | Average % miss | Median % miss | Within 5% | Within 10% | Within 20% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| no setting | 5 | 0.0010 | 0.4368 | 0.3943 | 0.6279 | 0.4809 | 0.5017 | 426,559 | 180,289 | 114,988 | 62.00 | 33.74 | 7.51 | 15.03 | 30.04 |

## Test (571,860 homes)

| Model | R² (log) | MSE (log) | RMSE (log) | MAE (log) | R² ($) | RMSE ($) | MAE ($) | Median $ miss | Average % miss | Median % miss | Within 5% | Within 10% | Within 20% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| linear regression | 0.4361 | 0.3944 | 0.6280 | 0.4813 | 0.4902 | 441,385 | 181,367 | 115,415 | 62.01 | 33.82 | 7.46 | 14.94 | 30.03 |

## The same test predictions cut into the five price bands

| Model | Accuracy (%) | Macro F1 | Within one band (%) |
|---|---|---|---|
| linear regression | 37.29 | 0.3790 | 80.67 |


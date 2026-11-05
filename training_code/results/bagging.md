# Bagging, regression form (Lecture 13)

50 regression trees on bootstrap samples, every column open at every split; the prediction is the average.

Regression on the natural log of the 2016 assessed value. Trained on all 2,287,442 training homes. Tested once on all 571,860 test homes.

## Search: average over the five validation folds

| Setting | Folds | MSE spread | R² (log) | MSE (log) | RMSE (log) | MAE (log) | R² ($) | RMSE ($) | MAE ($) | Median $ miss | Average % miss | Median % miss | Within 5% | Within 10% | Within 20% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| no setting | 5 | 0.0016 | 0.4339 | 0.3962 | 0.6295 | 0.4764 | 0.5424 | 408,786 | 175,980 | 109,514 | 60.55 | 33.21 | 8.47 | 16.45 | 31.50 |

## Test (571,860 homes)

| Model | R² (log) | MSE (log) | RMSE (log) | MAE (log) | R² ($) | RMSE ($) | MAE ($) | Median $ miss | Average % miss | Median % miss | Within 5% | Within 10% | Within 20% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bagging | 0.4348 | 0.3953 | 0.6287 | 0.4762 | 0.5428 | 417,993 | 176,328 | 109,642 | 60.50 | 33.20 | 8.42 | 16.33 | 31.46 |

Winning setting: no setting; final fit and test prediction 6,662 seconds.

## The same test predictions cut into the five price bands

| Model | Accuracy (%) | Macro F1 | Within one band (%) |
|---|---|---|---|
| bagging | 38.72 | 0.3950 | 80.64 |


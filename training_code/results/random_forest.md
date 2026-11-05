# Random forest, regression (Lecture 10)

50 CART regression trees on bootstrap samples, each split choosing from a random share of the columns; the prediction is the average.

Regression on the natural log of the 2016 assessed value. Trained on all 2,287,442 training homes. Tested once on all 571,860 test homes.

## Search: average over the five validation folds

| Setting | Folds | MSE spread | R² (log) | MSE (log) | RMSE (log) | MAE (log) | R² ($) | RMSE ($) | MAE ($) | Median $ miss | Average % miss | Median % miss | Within 5% | Within 10% | Within 20% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| column share 0.1 | 5 | 0.0011 | 0.4573 | 0.3799 | 0.6164 | 0.4681 | 0.5012 | 426,786 | 174,592 | 109,424 | 59.51 | 32.62 | 8.15 | 16.20 | 31.69 |
| column share sqrt | 5 | 0.0011 | 0.4526 | 0.3831 | 0.6190 | 0.4711 | 0.4327 | 455,144 | 178,555 | 110,615 | 60.00 | 32.93 | 7.92 | 15.82 | 31.29 |
| column share 0.33 | 5 | 0.0013 | 0.4484 | 0.3861 | 0.6214 | 0.4710 | 0.5376 | 410,931 | 174,441 | 109,132 | 59.83 | 32.80 | 8.38 | 16.45 | 31.73 |

## Test (571,860 homes)

| Model | R² (log) | MSE (log) | RMSE (log) | MAE (log) | R² ($) | RMSE ($) | MAE ($) | Median $ miss | Average % miss | Median % miss | Within 5% | Within 10% | Within 20% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| random forest | 0.4584 | 0.3788 | 0.6155 | 0.4677 | 0.4949 | 439,367 | 175,025 | 109,589 | 59.46 | 32.62 | 8.13 | 16.16 | 31.67 |

Winning setting: column share 0.1; final fit and test prediction 1,283 seconds.

## The same test predictions cut into the five price bands

| Model | Accuracy (%) | Macro F1 | Within one band (%) |
|---|---|---|---|
| random forest | 38.68 | 0.3930 | 81.56 |


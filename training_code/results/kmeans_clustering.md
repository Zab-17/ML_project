# K-means clustering (Lecture 14)

Mini-batch K-means on the 606 columns without the label; a home is priced at its cluster's median.

Regression on the natural log of the 2016 assessed value. Trained on all 2,287,442 training homes. Tested once on all 571,860 test homes.

## Search: average over the five validation folds

| Setting | Folds | MSE spread | R² (log) | MSE (log) | RMSE (log) | MAE (log) | R² ($) | RMSE ($) | MAE ($) | Median $ miss | Average % miss | Median % miss | Within 5% | Within 10% | Within 20% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| clusters 1000 | 5 | 0.0019 | 0.3513 | 0.4541 | 0.6739 | 0.5014 | 0.3646 | 481,682 | 194,517 | 121,458 | 72.04 | 34.64 | 7.60 | 15.10 | 29.75 |
| clusters 200 | 5 | 0.0029 | 0.3174 | 0.4778 | 0.6913 | 0.5206 | 0.3147 | 500,223 | 202,944 | 126,543 | 73.52 | 36.43 | 7.10 | 14.14 | 28.09 |
| clusters 50 | 5 | 0.0037 | 0.2751 | 0.5074 | 0.7123 | 0.5407 | 0.2355 | 528,334 | 212,604 | 132,706 | 75.95 | 38.17 | 6.72 | 13.43 | 26.73 |

## Test (571,860 homes)

| Model | R² (log) | MSE (log) | RMSE (log) | MAE (log) | R² ($) | RMSE ($) | MAE ($) | Median $ miss | Average % miss | Median % miss | Within 5% | Within 10% | Within 20% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| k-means clustering | 0.3520 | 0.4532 | 0.6732 | 0.5012 | 0.3545 | 496,685 | 195,299 | 121,170 | 71.84 | 34.58 | 7.59 | 15.07 | 29.79 |

Winning setting: clusters 1000; final fit and test prediction 96 seconds.

## The same test predictions cut into the five price bands

| Model | Accuracy (%) | Macro F1 | Within one band (%) |
|---|---|---|---|
| k-means clustering | 34.79 | 0.3349 | 79.72 |


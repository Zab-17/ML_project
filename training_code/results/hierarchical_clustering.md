# Hierarchical clustering (Lecture 14)

Ward agglomerative clustering on 10,000 sampled homes; every home joins the nearest cluster centre and is priced at its cluster's median.

Regression on the natural log of the 2016 assessed value. Trained on all 2,287,442 training homes. Tested once on all 571,860 test homes.

## Search: average over the five validation folds

| Setting | Folds | MSE spread | R² (log) | MSE (log) | RMSE (log) | MAE (log) | R² ($) | RMSE ($) | MAE ($) | Median $ miss | Average % miss | Median % miss | Within 5% | Within 10% | Within 20% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| clusters 1000 | 5 | 0.0013 | 0.3487 | 0.4559 | 0.6752 | 0.5027 | 0.3733 | 478,394 | 195,002 | 121,793 | 72.27 | 34.76 | 7.57 | 15.04 | 29.66 |
| clusters 200 | 5 | 0.0011 | 0.3168 | 0.4782 | 0.6915 | 0.5202 | 0.3157 | 499,878 | 202,998 | 126,370 | 73.68 | 36.36 | 7.11 | 14.18 | 28.19 |
| clusters 50 | 5 | 0.0031 | 0.2756 | 0.5071 | 0.7121 | 0.5398 | 0.2270 | 531,191 | 212,513 | 131,757 | 75.75 | 38.04 | 6.79 | 13.47 | 26.79 |

## Test (571,860 homes)

| Model | R² (log) | MSE (log) | RMSE (log) | MAE (log) | R² ($) | RMSE ($) | MAE ($) | Median $ miss | Average % miss | Median % miss | Within 5% | Within 10% | Within 20% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hierarchical clustering | 0.3486 | 0.4556 | 0.6750 | 0.5025 | 0.3499 | 498,451 | 195,812 | 121,387 | 72.13 | 34.74 | 7.65 | 15.09 | 29.77 |

Winning setting: clusters 1000; final fit and test prediction 48 seconds.

## The same test predictions cut into the five price bands

| Model | Accuracy (%) | Macro F1 | Within one band (%) |
|---|---|---|---|
| hierarchical clustering | 34.69 | 0.3335 | 79.80 |


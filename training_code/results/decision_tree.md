# Decision tree, CART regression (Lecture 10)

CART regression tree: binary splits on squared error, minimum leaf size searched.

Regression on the natural log of the 2016 assessed value. Trained on all 2,287,442 training homes. Tested once on all 571,860 test homes.

## Search: average over the five validation folds

| Setting | Folds | MSE spread | R² (log) | MSE (log) | RMSE (log) | MAE (log) | R² ($) | RMSE ($) | MAE ($) | Median $ miss | Average % miss | Median % miss | Within 5% | Within 10% | Within 20% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| smallest leaf 200 | 5 | 0.0015 | 0.4091 | 0.4136 | 0.6431 | 0.4919 | 0.4590 | 444,450 | 186,442 | 116,243 | 62.43 | 34.82 | 7.46 | 14.89 | 29.60 |
| smallest leaf 50 | 5 | 0.0013 | 0.4090 | 0.4137 | 0.6432 | 0.4898 | 0.4948 | 429,505 | 183,487 | 113,667 | 62.45 | 34.41 | 7.60 | 15.15 | 29.91 |
| smallest leaf 20 | 5 | 0.0021 | 0.3811 | 0.4332 | 0.6582 | 0.4986 | 0.5019 | 426,483 | 185,228 | 113,383 | 64.07 | 34.86 | 7.70 | 15.20 | 29.72 |
| smallest leaf 5 | 5 | 0.0011 | 0.2414 | 0.5310 | 0.7287 | 0.5419 | 0.4504 | 447,941 | 200,069 | 119,066 | 71.63 | 37.06 | 8.01 | 15.29 | 28.98 |
| smallest leaf 1 | 5 | 0.0018 | -0.0347 | 0.7242 | 0.8510 | 0.6125 | 0.2259 | 531,631 | 227,354 | 131,165 | 84.37 | 39.22 | 9.51 | 16.78 | 29.46 |

## Test (571,860 homes)

| Model | R² (log) | MSE (log) | RMSE (log) | MAE (log) | R² ($) | RMSE ($) | MAE ($) | Median $ miss | Average % miss | Median % miss | Within 5% | Within 10% | Within 20% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| decision tree | 0.4124 | 0.4109 | 0.6410 | 0.4906 | 0.4541 | 456,740 | 186,469 | 116,207 | 62.22 | 34.74 | 7.42 | 14.90 | 29.59 |

Winning setting: smallest leaf 200; final fit and test prediction 937 seconds.

## The same test predictions cut into the five price bands

| Model | Accuracy (%) | Macro F1 | Within one band (%) |
|---|---|---|---|
| decision tree | 36.65 | 0.3708 | 79.98 |


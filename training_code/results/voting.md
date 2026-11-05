# Voting, regression form (Lecture 13)

The average of the three regression models with the lowest validation MSE: the neural network, the random forest and k-nearest neighbours. No search and no validation score of its own.

Regression on the natural log of the 2016 assessed value. Trained on all 2,287,442 training homes. Tested once on all 571,860 test homes.

No five-fold search: nothing to tune.

## Test (571,860 homes)

| Model | R² (log) | MSE (log) | RMSE (log) | MAE (log) | R² ($) | RMSE ($) | MAE ($) | Median $ miss | Average % miss | Median % miss | Within 5% | Within 10% | Within 20% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| voting | 0.4665 | 0.3731 | 0.6108 | 0.4651 | 0.5265 | 425,368 | 173,229 | 109,169 | 58.93 | 32.46 | 8.03 | 16.10 | 31.76 |

## The same test predictions cut into the five price bands

| Model | Accuracy (%) | Macro F1 | Within one band (%) |
|---|---|---|---|
| voting | 38.77 | 0.3940 | 81.78 |


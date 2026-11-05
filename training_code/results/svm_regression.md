# Support vector machine, regression form (Lecture 12)

Linear support vector regression: misses smaller than the margin cost nothing (Lecture 12 teaches the classifier, which is in linear_svm_classifier.md).

Regression on the natural log of the 2016 assessed value. Trained on all 2,287,442 training homes. Tested once on all 571,860 test homes.

## Search: average over the five validation folds

| Setting | Folds | MSE spread | R² (log) | MSE (log) | RMSE (log) | MAE (log) | R² ($) | RMSE ($) | MAE ($) | Median $ miss | Average % miss | Median % miss | Within 5% | Within 10% | Within 20% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| margin penalty 0.01, margin width 0.05 | 5 | 0.0012 | 0.4114 | 0.4120 | 0.6419 | 0.4691 | 0.5109 | 422,621 | 177,029 | 113,438 | 69.81 | 31.60 | 8.18 | 16.32 | 32.27 |
| margin penalty 0.01, margin width 0.0 | 5 | 0.0014 | 0.4113 | 0.4121 | 0.6419 | 0.4691 | 0.5109 | 422,639 | 177,007 | 113,460 | 69.82 | 31.61 | 8.21 | 16.34 | 32.28 |
| margin penalty 1.0, margin width 0.05 | 5 | 0.0014 | 0.4112 | 0.4121 | 0.6420 | 0.4690 | 0.5168 | 420,066 | 176,882 | 113,487 | 69.90 | 31.58 | 8.19 | 16.32 | 32.27 |
| margin penalty 1.0, margin width 0.0 | 5 | 0.0013 | 0.4107 | 0.4125 | 0.6423 | 0.4690 | 0.5169 | 420,006 | 176,865 | 113,533 | 70.00 | 31.58 | 8.22 | 16.35 | 32.32 |

## Test (571,860 homes)

| Model | R² (log) | MSE (log) | RMSE (log) | MAE (log) | R² ($) | RMSE ($) | MAE ($) | Median $ miss | Average % miss | Median % miss | Within 5% | Within 10% | Within 20% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| support vector machine | 0.4109 | 0.4120 | 0.6419 | 0.4693 | 0.4956 | 439,039 | 178,053 | 113,796 | 69.80 | 31.61 | 8.17 | 16.29 | 32.21 |

Winning setting: margin penalty 0.01, margin width 0.05; final fit and test prediction 37 seconds.

## The same test predictions cut into the five price bands

| Model | Accuracy (%) | Macro F1 | Within one band (%) |
|---|---|---|---|
| support vector machine | 38.68 | 0.3835 | 82.41 |


# Logistic regression (Lecture 5)

Softmax over the five price bands with an L2 penalty C, L-BFGS solver, at most 300 passes.

Classifier on the five price bands (equal fifths of the training homes). Trained on all 2,287,442 training homes; tested once on all 571,860 test homes. Guessing scores about 20%.

## Search: average over the five validation folds

| Setting | Folds | Accuracy spread | Accuracy (%) | Macro F1 | Within one band (%) |
|---|---|---|---|---|---|
| penalty c 1.0 | 5 | 0.0490 | 41.16 | 0.3974 | 75.26 |
| penalty c 100.0 | 5 | 0.0374 | 41.16 | 0.3972 | 75.27 |
| penalty c 0.01 | 5 | 0.0373 | 41.04 | 0.3959 | 75.08 |

## Test (571,860 homes)

| Winning setting | Accuracy (%) | Macro F1 | Within one band (%) | Seconds (fit and test) |
|---|---|---|---|---|
| penalty c 1.0 | 41.24 | 0.3980 | 75.33 | 165.3 |


# Naive Bayes (Lecture 11)

A normal distribution per band for the 7 number columns and a yes/no probability per band for each 0/1 column, with Laplace smoothing.

Classifier on the five price bands (equal fifths of the training homes). Trained on all 2,287,442 training homes; tested once on all 571,860 test homes. Guessing scores about 20%.

## Search: average over the five validation folds

| Setting | Folds | Accuracy spread | Accuracy (%) | Macro F1 | Within one band (%) |
|---|---|---|---|---|---|
| smoothing 0.1 | 5 | 0.0354 | 38.39 | 0.3615 | 70.55 |
| smoothing 1.0 | 5 | 0.0358 | 38.39 | 0.3614 | 70.54 |
| smoothing 10.0 | 5 | 0.0365 | 38.36 | 0.3611 | 70.50 |

## Test (571,860 homes)

| Winning setting | Accuracy (%) | Macro F1 | Within one band (%) | Seconds (fit and test) |
|---|---|---|---|---|
| smoothing 0.1 | 38.48 | 0.3623 | 70.56 | 6.1 |


# Perceptron (Lecture 6)

The perceptron rule, one perceptron per band against the rest.

Classifier on the five price bands (equal fifths of the training homes). Trained on all 2,287,442 training homes; tested once on all 571,860 test homes. Guessing scores about 20%.

## Search: average over the five validation folds

| Setting | Folds | Accuracy spread | Accuracy (%) | Macro F1 | Within one band (%) |
|---|---|---|---|---|---|
| passes 5 | 5 | 1.7779 | 33.90 | 0.3320 | 70.42 |
| passes 20 | 5 | 3.1084 | 31.72 | 0.3105 | 67.71 |

## Test (571,860 homes)

| Winning setting | Accuracy (%) | Macro F1 | Within one band (%) | Seconds (fit and test) |
|---|---|---|---|---|
| passes 5 | 34.88 | 0.3394 | 71.69 | 42.6 |


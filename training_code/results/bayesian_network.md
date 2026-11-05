# Bayesian network

Lecture 11: the price band is a child of location, size, quality and age; probability tables learned by counting with Laplace smoothing; the best of four structures chosen by cross-validation.

Trained on all 2,287,442 training homes, in five folds. Tested once on all 571,860 test homes. Five price bands: equal fifths of the training homes. Guessing scores about 20%.

## Search: average over the five validation folds

| Setting | Accuracy (%) | Spread | Macro F1 | Within one band (%) | Seconds per fold |
|---|---|---|---|---|---|
| {"parents": ["ZIP code", "living area (log)", "building quality", "year built"]} | 42.44 | 0.10 | 0.4100 | 74.94 | 12.0 |
| {"parents": ["ZIP code", "living area (log)", "year built"]} | 42.09 | 0.07 | 0.4039 | 74.49 | 13.7 |
| {"parents": ["ZIP code", "living area (log)", "building quality"]} | 41.73 | 0.06 | 0.4015 | 75.01 | 11.8 |
| {"parents": ["ZIP code", "living area (log)"]} | 40.74 | 0.05 | 0.3887 | 74.49 | 12.5 |

## Test

| Winning setting | Accuracy (%) | Macro F1 | Within one band (%) | Seconds (fit and test) |
|---|---|---|---|---|
| {"parents": ["ZIP code", "living area (log)", "building quality", "year built"]} | 42.63 | 0.4113 | 74.93 | 4.4 |

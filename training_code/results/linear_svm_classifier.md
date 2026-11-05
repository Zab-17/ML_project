# support vector machine (linear classifier)

Lecture 12: linear maximum-margin classifier with the hinge loss, a soft margin weighted by C, one machine per band against the rest.

Trained on a random sample of 20,000 training homes (seed 42), in five folds. Tested once on all 571,860 test homes. Five price bands: equal fifths of the training homes. Guessing scores about 20%.

## Search: average over the five validation folds

| Setting | Accuracy (%) | Spread | Macro F1 | Within one band (%) | Seconds per fold |
|---|---|---|---|---|---|
| {"margin_penalty": 0.1} | 34.38 | 1.03 | 0.3095 | 67.83 | 3.0 |
| {"margin_penalty": 0.01} | 33.81 | 0.73 | 0.3007 | 65.85 | 2.4 |
| {"margin_penalty": 10.0} | 33.50 | 0.91 | 0.2934 | 65.51 | 5.9 |
| {"margin_penalty": 1.0} | 33.31 | 1.39 | 0.3066 | 66.35 | 4.2 |

## Test

| Winning setting | Accuracy (%) | Macro F1 | Within one band (%) | Seconds (fit and test) |
|---|---|---|---|---|
| {"margin_penalty": 0.1} | 34.15 | 0.2944 | 64.44 | 1.8 |

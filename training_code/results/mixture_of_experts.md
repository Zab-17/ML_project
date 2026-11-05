# mixture of experts

Lecture 13: linear softmax experts trained together with a softmax gating network that weights the experts for each home.

Trained on a random sample of 100,000 training homes (seed 42), in five folds. Tested once on all 571,860 test homes. Five price bands: equal fifths of the training homes. Guessing scores about 20%.

## Search: average over the five validation folds

| Setting | Accuracy (%) | Spread | Macro F1 | Within one band (%) | Seconds per fold |
|---|---|---|---|---|---|
| {"experts": 8} | 42.06 | 0.45 | 0.4092 | 76.21 | 9.2 |
| {"experts": 4} | 41.58 | 0.37 | 0.4054 | 76.17 | 10.5 |
| {"experts": 2} | 41.31 | 0.47 | 0.4025 | 76.15 | 14.2 |

## Test

| Winning setting | Accuracy (%) | Macro F1 | Within one band (%) | Seconds (fit and test) |
|---|---|---|---|---|
| {"experts": 8} | 42.25 | 0.4114 | 76.27 | 11.1 |

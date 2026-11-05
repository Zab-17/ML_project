# k-nearest neighbours (majority vote)

Lecture 3: majority vote of the k nearest training homes, MinMax scaling, Euclidean or cosine distance, k chosen by five-fold cross-validation. One neighbour search per distance serves every k, so each fold's search time is listed under the first k of that distance.

Trained on a random sample of 100,000 training homes (seed 42), in five folds. Tested once on all 571,860 test homes. Five price bands: equal fifths of the training homes. Guessing scores about 20%.

## Search: average over the five validation folds

| Setting | Accuracy (%) | Spread | Macro F1 | Within one band (%) | Seconds per fold |
|---|---|---|---|---|---|
| {"distance": "euclidean", "k": 30} | 39.27 | 0.33 | 0.3823 | 72.88 | 0.0 |
| {"distance": "euclidean", "k": 15} | 39.22 | 0.33 | 0.3847 | 73.20 | 0.0 |
| {"distance": "euclidean", "k": 20} | 39.14 | 0.31 | 0.3826 | 73.04 | 0.0 |
| {"distance": "euclidean", "k": 50} | 39.10 | 0.09 | 0.3783 | 72.57 | 0.0 |
| {"distance": "euclidean", "k": 10} | 38.97 | 0.39 | 0.3838 | 73.09 | 0.0 |
| {"distance": "cosine", "k": 30} | 38.89 | 0.27 | 0.3768 | 72.67 | 0.0 |
| {"distance": "cosine", "k": 20} | 38.81 | 0.20 | 0.3776 | 72.86 | 0.0 |
| {"distance": "cosine", "k": 10} | 38.75 | 0.36 | 0.3804 | 72.95 | 0.0 |
| {"distance": "cosine", "k": 15} | 38.72 | 0.37 | 0.3781 | 72.82 | 0.0 |
| {"distance": "cosine", "k": 50} | 38.48 | 0.08 | 0.3704 | 72.09 | 0.0 |
| {"distance": "euclidean", "k": 5} | 38.16 | 0.50 | 0.3781 | 72.53 | 0.0 |
| {"distance": "cosine", "k": 5} | 38.10 | 0.43 | 0.3767 | 72.37 | 0.0 |
| {"distance": "euclidean", "k": 3} | 37.57 | 0.12 | 0.3742 | 71.85 | 0.2 |
| {"distance": "cosine", "k": 3} | 37.54 | 0.21 | 0.3732 | 71.73 | 0.1 |
| {"distance": "euclidean", "k": 1} | 36.55 | 0.17 | 0.3658 | 71.04 | 8.2 |
| {"distance": "cosine", "k": 1} | 36.54 | 0.29 | 0.3653 | 71.02 | 7.0 |

## Test

| Winning setting | Accuracy (%) | Macro F1 | Within one band (%) | Seconds (fit and test) |
|---|---|---|---|---|
| {"distance": "euclidean", "k": 30} | 39.57 | 0.3861 | 73.18 | 165.0 |

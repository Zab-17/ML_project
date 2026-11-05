# K-nearest neighbours, regression form

The average log value of the k nearest training homes (Lecture 3 teaches the majority-vote classifier, which is in knn_classifier.md).

Regression on the natural log of the 2016 assessed value. Trained on all 2,287,442 training homes. Tested once on all 571,860 test homes.

## Search: average over the five validation folds

| Setting | Folds | MSE spread | R² (log) | MSE (log) | RMSE (log) | MAE (log) | R² ($) | RMSE ($) | MAE ($) | Median $ miss | Average % miss | Median % miss | Within 5% | Within 10% | Within 20% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| distance euclidean, k 50 | 5 | 0.0013 | 0.4496 | 0.3853 | 0.6207 | 0.4739 | 0.5117 | 422,285 | 176,318 | 110,525 | 59.67 | 33.19 | 7.89 | 15.68 | 30.94 |
| distance cosine, k 50 | 5 | 0.0012 | 0.4487 | 0.3859 | 0.6212 | 0.4736 | 0.5209 | 418,259 | 176,518 | 110,834 | 60.32 | 33.21 | 7.92 | 15.74 | 30.95 |
| distance euclidean, k 30 | 5 | 0.0012 | 0.4468 | 0.3872 | 0.6223 | 0.4739 | 0.5238 | 417,028 | 175,650 | 109,825 | 59.86 | 33.14 | 8.02 | 15.87 | 31.11 |
| distance cosine, k 30 | 5 | 0.0011 | 0.4460 | 0.3878 | 0.6227 | 0.4737 | 0.5293 | 414,586 | 175,965 | 110,096 | 60.40 | 33.15 | 8.07 | 15.91 | 31.11 |
| distance euclidean, k 20 | 5 | 0.0012 | 0.4407 | 0.3915 | 0.6257 | 0.4755 | 0.5292 | 414,629 | 175,701 | 109,595 | 60.22 | 33.21 | 8.14 | 15.99 | 31.13 |
| distance cosine, k 20 | 5 | 0.0011 | 0.4401 | 0.3920 | 0.6261 | 0.4752 | 0.5331 | 412,891 | 176,040 | 109,774 | 60.68 | 33.20 | 8.15 | 16.02 | 31.18 |
| distance euclidean, k 15 | 5 | 0.0013 | 0.4337 | 0.3964 | 0.6296 | 0.4776 | 0.5322 | 413,329 | 176,070 | 109,549 | 60.62 | 33.30 | 8.20 | 16.04 | 31.12 |
| distance cosine, k 15 | 5 | 0.0012 | 0.4331 | 0.3968 | 0.6299 | 0.4774 | 0.5347 | 412,206 | 176,491 | 109,789 | 61.04 | 33.31 | 8.22 | 16.07 | 31.15 |
| distance euclidean, k 10 | 5 | 0.0013 | 0.4183 | 0.4072 | 0.6381 | 0.4825 | 0.5318 | 413,505 | 177,336 | 109,691 | 61.50 | 33.58 | 8.28 | 16.10 | 31.03 |
| distance cosine, k 10 | 5 | 0.0014 | 0.4179 | 0.4075 | 0.6383 | 0.4824 | 0.5331 | 412,890 | 177,816 | 109,838 | 61.85 | 33.60 | 8.29 | 16.14 | 31.04 |
| distance euclidean, k 5 | 5 | 0.0016 | 0.3692 | 0.4416 | 0.6645 | 0.4987 | 0.5223 | 417,648 | 182,160 | 111,564 | 64.21 | 34.51 | 8.45 | 16.12 | 30.62 |
| distance cosine, k 5 | 5 | 0.0015 | 0.3687 | 0.4419 | 0.6648 | 0.4987 | 0.5169 | 420,019 | 182,843 | 111,750 | 64.52 | 34.52 | 8.44 | 16.12 | 30.60 |
| distance euclidean, k 3 | 5 | 0.0014 | 0.3013 | 0.4891 | 0.6994 | 0.5202 | 0.4907 | 431,233 | 189,466 | 114,581 | 67.88 | 35.72 | 8.68 | 16.20 | 30.23 |
| distance cosine, k 3 | 5 | 0.0010 | 0.3007 | 0.4895 | 0.6997 | 0.5203 | 0.4808 | 435,382 | 190,385 | 114,710 | 68.16 | 35.74 | 8.68 | 16.20 | 30.22 |
| distance cosine, k 1 | 5 | 0.0020 | -0.0434 | 0.7304 | 0.8546 | 0.6129 | 0.2088 | 537,421 | 224,901 | 132,486 | 84.32 | 39.80 | 10.69 | 18.09 | 30.38 |
| distance euclidean, k 1 | 5 | 0.0021 | -0.0438 | 0.7307 | 0.8548 | 0.6130 | 0.2802 | 512,605 | 223,064 | 132,411 | 84.15 | 39.77 | 10.71 | 18.11 | 30.40 |

## Test (571,860 homes)

| Model | R² (log) | MSE (log) | RMSE (log) | MAE (log) | R² ($) | RMSE ($) | MAE ($) | Median $ miss | Average % miss | Median % miss | Within 5% | Within 10% | Within 20% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| k-nearest neighbours | 0.4510 | 0.3839 | 0.6196 | 0.4733 | 0.5085 | 433,416 | 176,768 | 110,531 | 59.62 | 33.21 | 7.91 | 15.67 | 30.92 |

Winning setting: distance euclidean, k 50; final fit and test prediction 2,715 seconds.

## The same test predictions cut into the five price bands

| Model | Accuracy (%) | Macro F1 | Within one band (%) |
|---|---|---|---|
| k-nearest neighbours | 38.28 | 0.3900 | 81.07 |


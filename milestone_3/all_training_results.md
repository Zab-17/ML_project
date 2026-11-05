# Zillow: every training result, every metric

All numbers are read from `zillow/out/` by a script; none are typed by hand. The data: 2,287,442 training homes in five folds of about 457,000, and 571,860 sealed test homes, each scored once per model.

## What each metric means

- **MSE (log)**: mean squared error on the natural log of the assessed value. Every model was chosen on this; lower is better.
- **RMSE (log)** and **MAE (log)**: root mean squared error and mean absolute error on the log scale.
- **R² (log)** and **R² ($)**: coefficient of determination, the share of variation explained, on the log scale and in dollars. 1 is perfect, 0 is no better than one number for every home.
- **RMSE ($)**, **MAE ($)**, **Median $ miss**: the miss in dollars, after converting the log prediction back.
- **Average % miss**: the mean absolute percentage error (MAPE). **Median % miss**: half the homes are off by less than this.
- **Within 5% / 10% / 20%**: the share of homes predicted that close to their value.
- **MSE spread**: the sample standard deviation of the five fold scores.
- **Price bands**: five equal-sized bands cut at $165,080, $269,041, $387,279 and $580,090. **Accuracy**: exact band correct. **Macro F1**: the F1 score averaged with equal weight per band. **Within one band**: the right band or the one next to it.

## 1. Test results, every regression model (571,860 homes)

| model | R² (log) | MSE (log) | RMSE (log) | MAE (log) | R² ($) | RMSE ($) | MAE ($) | Median $ miss | Average % miss | Median % miss | Within 5% | Within 10% | Within 20% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| neural network | 0.4673 | 0.3725 | 0.6104 | 0.4653 | 0.5694 | 405,643 | 172,805 | 109,817 | 59.22 | 32.48 | 7.94 | 15.83 | 31.63 |
| voting (network + forest + neighbours) | 0.4665 | 0.3731 | 0.6108 | 0.4651 | 0.5265 | 425,368 | 173,229 | 109,169 | 58.93 | 32.46 | 8.03 | 16.10 | 31.76 |
| random forest | 0.4584 | 0.3788 | 0.6155 | 0.4677 | 0.4949 | 439,367 | 175,025 | 109,589 | 59.46 | 32.62 | 8.13 | 16.16 | 31.67 |
| k-nearest neighbours | 0.4510 | 0.3839 | 0.6196 | 0.4733 | 0.5085 | 433,416 | 176,768 | 110,531 | 59.62 | 33.21 | 7.91 | 15.67 | 30.92 |
| linear regression | 0.4361 | 0.3944 | 0.6280 | 0.4813 | 0.4902 | 441,385 | 181,367 | 115,415 | 62.01 | 33.82 | 7.46 | 14.94 | 30.03 |
| bagging | 0.4348 | 0.3953 | 0.6287 | 0.4762 | 0.5428 | 417,993 | 176,328 | 109,642 | 60.50 | 33.20 | 8.42 | 16.33 | 31.46 |
| decision tree | 0.4124 | 0.4109 | 0.6410 | 0.4906 | 0.4541 | 456,740 | 186,469 | 116,207 | 62.22 | 34.74 | 7.42 | 14.90 | 29.59 |
| support vector machine | 0.4109 | 0.4120 | 0.6419 | 0.4693 | 0.4956 | 439,039 | 178,053 | 113,796 | 69.80 | 31.61 | 8.17 | 16.29 | 32.21 |
| k-means clustering | 0.3520 | 0.4532 | 0.6732 | 0.5012 | 0.3545 | 496,685 | 195,299 | 121,170 | 71.84 | 34.58 | 7.59 | 15.07 | 29.79 |
| hierarchical clustering | 0.3486 | 0.4556 | 0.6750 | 0.5025 | 0.3499 | 498,451 | 195,812 | 121,387 | 72.13 | 34.74 | 7.65 | 15.09 | 29.77 |
| AdaBoost | 0.2989 | 0.4903 | 0.7002 | 0.5715 | 0.3871 | 483,984 | 212,254 | 127,382 | 59.27 | 42.45 | 5.39 | 10.94 | 22.27 |
| median baseline | -0.0039 | 0.7021 | 0.8379 | 0.6386 | -0.0359 | 629,192 | 255,152 | 152,042 | 91.10 | 45.37 | 5.51 | 11.16 | 22.30 |
| average baseline | -0.1836 | 0.8278 | 0.9098 | 0.6909 | -0.0000 | 618,192 | 276,017 | 202,977 | 129.56 | 53.82 | 5.52 | 10.76 | 20.89 |

## 2. Winning settings and final run times

| Model | Winning setting | Final run (seconds) | Final run (minutes) | What the time covers |
|---|---|---|---|---|
| neural network | 256-128 adam lr 0.001 dropout 0.2 | 272.2 | 4.5 | training only |
| decision tree | smallest leaf 200 | 937.0 | 15.6 | fit and test predictions |
| random forest | column share 0.1 | 1,283.4 | 21.4 | fit and test predictions |
| AdaBoost | tree depth 8 | 1,821.1 | 30.4 | fit and test predictions |
| support vector machine | margin penalty 0.01, margin width 0.05 | 36.7 | 0.6 | fit and test predictions |
| k-means clustering | clusters 1000 | 95.6 | 1.6 | fit and test predictions |
| hierarchical clustering | clusters 1000 | 48.3 | 0.8 | fit and test predictions |
| k-nearest neighbours | distance euclidean, k 50 | 2,714.6 | 45.2 | fit and test predictions |
| bagging | no search | 6,661.6 | 111.0 | fit and test predictions |
| logistic regression | penalty c 1.0 | 165.3 | 2.8 | fit and test predictions |
| perceptron | passes 5 | 42.6 | 0.7 | fit and test predictions |
| naive Bayes | smoothing 0.1 | 6.1 | 0.1 | fit and test predictions |

## 3. Search: every setting, averaged over its five validation folds

Sorted by each model's best validation MSE (log), then by setting. The last column is the average time of one fold, in seconds.

| model | setting | folds | MSE spread | R² (log) | MSE (log) | RMSE (log) | MAE (log) | R² ($) | RMSE ($) | MAE ($) | Median $ miss | Average % miss | Median % miss | Within 5% | Within 10% | Within 20% | Seconds per fold |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| neural network | 256-128 adam lr 0.001 dropout 0.2 | 5 | 0.0010 | 0.4672 | 0.3730 | 0.6107 | 0.4640 | 0.5868 | 388,455 | 171,140 | 109,503 | 59.65 | 32.41 | 8.10 | 16.19 | 31.84 | 212.4 |
| neural network | 128-64 adam lr 0.001 dropout 0.2 | 5 | 0.0011 | 0.4669 | 0.3731 | 0.6109 | 0.4646 | 0.5221 | 414,990 | 171,498 | 109,764 | 59.61 | 32.44 | 8.02 | 16.01 | 31.73 | 295.3 |
| neural network | 512-256 adam lr 0.001 dropout 0.2 | 5 | 0.0014 | 0.4661 | 0.3737 | 0.6113 | 0.4652 | 0.5750 | 393,796 | 171,841 | 109,642 | 59.40 | 32.53 | 8.08 | 16.12 | 31.69 | 259.1 |
| neural network | 512-256 sgd lr 0.1 dropout 0.2 | 5 | 0.0033 | 0.4630 | 0.3759 | 0.6131 | 0.4672 | 0.5651 | 398,484 | 172,784 | 110,972 | 59.87 | 32.77 | 7.93 | 15.87 | 31.33 | 319.4 |
| neural network | 256-128 sgd lr 0.1 dropout 0 | 5 | 0.0029 | 0.4629 | 0.3760 | 0.6132 | 0.4686 | 0.5703 | 395,955 | 172,721 | 110,860 | 59.60 | 32.94 | 7.85 | 15.74 | 31.18 | 162.1 |
| neural network | 256-128 sgd lr 0.1 dropout 0.2 | 5 | 0.0026 | 0.4629 | 0.3760 | 0.6132 | 0.4671 | 0.5662 | 397,951 | 172,641 | 110,917 | 59.93 | 32.68 | 7.94 | 15.87 | 31.37 | 172.5 |
| neural network | 512-256 sgd lr 0.1 dropout 0 | 5 | 0.0025 | 0.4621 | 0.3765 | 0.6136 | 0.4698 | 0.5633 | 399,145 | 173,279 | 111,197 | 59.43 | 33.03 | 7.79 | 15.58 | 30.99 | 224.2 |
| neural network | 128-64 sgd lr 0.1 dropout 0 | 5 | 0.0028 | 0.4620 | 0.3766 | 0.6137 | 0.4690 | 0.5664 | 397,785 | 173,143 | 111,229 | 59.68 | 32.97 | 7.82 | 15.66 | 31.12 | 160.0 |
| neural network | 128-64 sgd lr 0.1 dropout 0.2 | 5 | 0.0019 | 0.4614 | 0.3770 | 0.6140 | 0.4674 | 0.5696 | 396,426 | 173,083 | 111,265 | 60.15 | 32.72 | 7.92 | 15.81 | 31.38 | 198.6 |
| neural network | 128-64 adam lr 0.001 dropout 0 | 5 | 0.0020 | 0.4588 | 0.3788 | 0.6155 | 0.4693 | -86674.3547 | 79,985,036 | 290,965 | 111,177 | 86.10 | 32.96 | 7.88 | 15.74 | 31.16 | 161.2 |
| neural network | 256-128 adam lr 0.001 dropout 0 | 5 | 0.0008 | 0.4585 | 0.3790 | 0.6157 | 0.4673 | 0.4933 | 428,135 | 172,863 | 110,881 | 60.77 | 32.74 | 8.03 | 15.99 | 31.55 | 164.5 |
| neural network | 512-256 adam lr 0.001 dropout 0 | 5 | 0.0006 | 0.4574 | 0.3798 | 0.6163 | 0.4695 | 0.5877 | 388,036 | 173,271 | 110,693 | 60.23 | 33.01 | 7.98 | 15.92 | 31.37 | 212.9 |
| random forest | column share 0.1 | 5 | 0.0011 | 0.4573 | 0.3799 | 0.6164 | 0.4681 | 0.5012 | 426,786 | 174,592 | 109,424 | 59.51 | 32.62 | 8.15 | 16.20 | 31.69 | 155.8 |
| random forest | column share sqrt | 5 | 0.0011 | 0.4526 | 0.3831 | 0.6190 | 0.4711 | 0.4327 | 455,144 | 178,555 | 110,615 | 60.00 | 32.93 | 7.92 | 15.82 | 31.29 | 106.6 |
| random forest | column share 0.33 | 5 | 0.0013 | 0.4484 | 0.3861 | 0.6214 | 0.4710 | 0.5376 | 410,931 | 174,441 | 109,132 | 59.83 | 32.80 | 8.38 | 16.45 | 31.73 | 400.1 |
| k-nearest neighbours | distance euclidean, k 50 | 5 | 0.0013 | 0.4496 | 0.3853 | 0.6207 | 0.4739 | 0.5117 | 422,285 | 176,318 | 110,525 | 59.67 | 33.19 | 7.89 | 15.68 | 30.94 | 0.3 |
| k-nearest neighbours | distance cosine, k 50 | 5 | 0.0012 | 0.4487 | 0.3859 | 0.6212 | 0.4736 | 0.5209 | 418,259 | 176,518 | 110,834 | 60.32 | 33.21 | 7.92 | 15.74 | 30.95 | 0.3 |
| k-nearest neighbours | distance euclidean, k 30 | 5 | 0.0012 | 0.4468 | 0.3872 | 0.6223 | 0.4739 | 0.5238 | 417,028 | 175,650 | 109,825 | 59.86 | 33.14 | 8.02 | 15.87 | 31.11 | 0.3 |
| k-nearest neighbours | distance cosine, k 30 | 5 | 0.0011 | 0.4460 | 0.3878 | 0.6227 | 0.4737 | 0.5293 | 414,586 | 175,965 | 110,096 | 60.40 | 33.15 | 8.07 | 15.91 | 31.11 | 0.2 |
| k-nearest neighbours | distance euclidean, k 20 | 5 | 0.0012 | 0.4407 | 0.3915 | 0.6257 | 0.4755 | 0.5292 | 414,629 | 175,701 | 109,595 | 60.22 | 33.21 | 8.14 | 15.99 | 31.13 | 0.2 |
| k-nearest neighbours | distance cosine, k 20 | 5 | 0.0011 | 0.4401 | 0.3920 | 0.6261 | 0.4752 | 0.5331 | 412,891 | 176,040 | 109,774 | 60.68 | 33.20 | 8.15 | 16.02 | 31.18 | 0.1 |
| k-nearest neighbours | distance euclidean, k 15 | 5 | 0.0013 | 0.4337 | 0.3964 | 0.6296 | 0.4776 | 0.5322 | 413,329 | 176,070 | 109,549 | 60.62 | 33.30 | 8.20 | 16.04 | 31.12 | 0.2 |
| k-nearest neighbours | distance cosine, k 15 | 5 | 0.0012 | 0.4331 | 0.3968 | 0.6299 | 0.4774 | 0.5347 | 412,206 | 176,491 | 109,789 | 61.04 | 33.31 | 8.22 | 16.07 | 31.15 | 0.1 |
| k-nearest neighbours | distance euclidean, k 10 | 5 | 0.0013 | 0.4183 | 0.4072 | 0.6381 | 0.4825 | 0.5318 | 413,505 | 177,336 | 109,691 | 61.50 | 33.58 | 8.28 | 16.10 | 31.03 | 0.1 |
| k-nearest neighbours | distance cosine, k 10 | 5 | 0.0014 | 0.4179 | 0.4075 | 0.6383 | 0.4824 | 0.5331 | 412,890 | 177,816 | 109,838 | 61.85 | 33.60 | 8.29 | 16.14 | 31.04 | 0.1 |
| k-nearest neighbours | distance euclidean, k 5 | 5 | 0.0016 | 0.3692 | 0.4416 | 0.6645 | 0.4987 | 0.5223 | 417,648 | 182,160 | 111,564 | 64.21 | 34.51 | 8.45 | 16.12 | 30.62 | 0.1 |
| k-nearest neighbours | distance cosine, k 5 | 5 | 0.0015 | 0.3687 | 0.4419 | 0.6648 | 0.4987 | 0.5169 | 420,019 | 182,843 | 111,750 | 64.52 | 34.52 | 8.44 | 16.12 | 30.60 | 0.1 |
| k-nearest neighbours | distance euclidean, k 3 | 5 | 0.0014 | 0.3013 | 0.4891 | 0.6994 | 0.5202 | 0.4907 | 431,233 | 189,466 | 114,581 | 67.88 | 35.72 | 8.68 | 16.20 | 30.23 | 0.1 |
| k-nearest neighbours | distance cosine, k 3 | 5 | 0.0010 | 0.3007 | 0.4895 | 0.6997 | 0.5203 | 0.4808 | 435,382 | 190,385 | 114,710 | 68.16 | 35.74 | 8.68 | 16.20 | 30.22 | 0.0 |
| k-nearest neighbours | distance cosine, k 1 | 5 | 0.0020 | -0.0434 | 0.7304 | 0.8546 | 0.6129 | 0.2088 | 537,421 | 224,901 | 132,486 | 84.32 | 39.80 | 10.69 | 18.09 | 30.38 | 12,396.8 |
| k-nearest neighbours | distance euclidean, k 1 | 5 | 0.0021 | -0.0438 | 0.7307 | 0.8548 | 0.6130 | 0.2802 | 512,605 | 223,064 | 132,411 | 84.15 | 39.77 | 10.71 | 18.11 | 30.40 | 3,871.2 |
| linear regression | no setting | 5 | 0.0010 | 0.4368 | 0.3943 | 0.6279 | 0.4809 | 0.5017 | 426,559 | 180,289 | 114,988 | 62.00 | 33.74 | 7.51 | 15.03 | 30.04 | — |
| bagging | no search | 5 | 0.0016 | 0.4339 | 0.3962 | 0.6295 | 0.4764 | 0.5424 | 408,786 | 175,980 | 109,514 | 60.55 | 33.21 | 8.47 | 16.45 | 31.50 | 18,249.5 |
| support vector machine | margin penalty 0.01, margin width 0.05 | 5 | 0.0012 | 0.4114 | 0.4120 | 0.6419 | 0.4691 | 0.5109 | 422,621 | 177,029 | 113,438 | 69.81 | 31.60 | 8.18 | 16.32 | 32.27 | 20.1 |
| support vector machine | margin penalty 0.01, margin width 0.0 | 5 | 0.0014 | 0.4113 | 0.4121 | 0.6419 | 0.4691 | 0.5109 | 422,639 | 177,007 | 113,460 | 69.82 | 31.61 | 8.21 | 16.34 | 32.28 | 21.4 |
| support vector machine | margin penalty 1.0, margin width 0.05 | 5 | 0.0014 | 0.4112 | 0.4121 | 0.6420 | 0.4690 | 0.5168 | 420,066 | 176,882 | 113,487 | 69.90 | 31.58 | 8.19 | 16.32 | 32.27 | 1,080.4 |
| support vector machine | margin penalty 1.0, margin width 0.0 | 5 | 0.0013 | 0.4107 | 0.4125 | 0.6423 | 0.4690 | 0.5169 | 420,006 | 176,865 | 113,533 | 70.00 | 31.58 | 8.22 | 16.35 | 32.32 | 1,186.0 |
| decision tree | smallest leaf 200 | 5 | 0.0015 | 0.4091 | 0.4136 | 0.6431 | 0.4919 | 0.4590 | 444,450 | 186,442 | 116,243 | 62.43 | 34.82 | 7.46 | 14.89 | 29.60 | 125.0 |
| decision tree | smallest leaf 50 | 5 | 0.0013 | 0.4090 | 0.4137 | 0.6432 | 0.4898 | 0.4948 | 429,505 | 183,487 | 113,667 | 62.45 | 34.41 | 7.60 | 15.15 | 29.91 | 143.7 |
| decision tree | smallest leaf 20 | 5 | 0.0021 | 0.3811 | 0.4332 | 0.6582 | 0.4986 | 0.5019 | 426,483 | 185,228 | 113,383 | 64.07 | 34.86 | 7.70 | 15.20 | 29.72 | 153.9 |
| decision tree | smallest leaf 5 | 5 | 0.0011 | 0.2414 | 0.5310 | 0.7287 | 0.5419 | 0.4504 | 447,941 | 200,069 | 119,066 | 71.63 | 37.06 | 8.01 | 15.29 | 28.98 | 165.6 |
| decision tree | smallest leaf 1 | 5 | 0.0018 | -0.0347 | 0.7242 | 0.8510 | 0.6125 | 0.2259 | 531,631 | 227,354 | 131,165 | 84.37 | 39.22 | 9.51 | 16.78 | 29.46 | 178.1 |
| k-means clustering | clusters 1000 | 5 | 0.0019 | 0.3513 | 0.4541 | 0.6739 | 0.5014 | 0.3646 | 481,682 | 194,517 | 121,458 | 72.04 | 34.64 | 7.60 | 15.10 | 29.75 | 80.5 |
| k-means clustering | clusters 200 | 5 | 0.0029 | 0.3174 | 0.4778 | 0.6913 | 0.5206 | 0.3147 | 500,223 | 202,944 | 126,543 | 73.52 | 36.43 | 7.10 | 14.14 | 28.09 | 1,109.0 |
| k-means clustering | clusters 50 | 5 | 0.0037 | 0.2751 | 0.5074 | 0.7123 | 0.5407 | 0.2355 | 528,334 | 212,604 | 132,706 | 75.95 | 38.17 | 6.72 | 13.43 | 26.73 | 19.9 |
| hierarchical clustering | clusters 1000 | 5 | 0.0013 | 0.3487 | 0.4559 | 0.6752 | 0.5027 | 0.3733 | 478,394 | 195,002 | 121,793 | 72.27 | 34.76 | 7.57 | 15.04 | 29.66 | 47.9 |
| hierarchical clustering | clusters 200 | 5 | 0.0011 | 0.3168 | 0.4782 | 0.6915 | 0.5202 | 0.3157 | 499,878 | 202,998 | 126,370 | 73.68 | 36.36 | 7.11 | 14.18 | 28.19 | 37.9 |
| hierarchical clustering | clusters 50 | 5 | 0.0031 | 0.2756 | 0.5071 | 0.7121 | 0.5398 | 0.2270 | 531,191 | 212,513 | 131,757 | 75.75 | 38.04 | 6.79 | 13.47 | 26.79 | 38.1 |
| AdaBoost | tree depth 8 | 5 | 0.0048 | 0.3081 | 0.4843 | 0.6959 | 0.5663 | 0.3940 | 470,419 | 209,986 | 127,474 | 59.39 | 41.98 | 5.58 | 11.23 | 22.80 | 1,739.5 |
| AdaBoost | tree depth 4 | 5 | 0.0085 | 0.2833 | 0.5017 | 0.7083 | 0.5669 | 0.2988 | 506,011 | 216,925 | 130,966 | 63.03 | 41.56 | 5.86 | 11.73 | 23.49 | 765.4 |

## 4. Search: MSE (log) of every setting on every fold

| model | setting | fold 0 | fold 1 | fold 2 | fold 3 | fold 4 |
|---|---|---|---|---|---|---|
| AdaBoost | tree depth 4 | 0.4953 | 0.4925 | 0.4993 | 0.5119 | 0.5092 |
| AdaBoost | tree depth 8 | 0.4857 | 0.4870 | 0.4839 | 0.4885 | 0.4763 |
| bagging | no search | 0.3954 | 0.3956 | 0.3977 | 0.3944 | 0.3980 |
| decision tree | smallest leaf 1 | 0.7242 | 0.7213 | 0.7263 | 0.7246 | 0.7249 |
| decision tree | smallest leaf 20 | 0.4320 | 0.4328 | 0.4358 | 0.4307 | 0.4349 |
| decision tree | smallest leaf 200 | 0.4134 | 0.4130 | 0.4148 | 0.4115 | 0.4152 |
| decision tree | smallest leaf 5 | 0.5312 | 0.5309 | 0.5323 | 0.5292 | 0.5314 |
| decision tree | smallest leaf 50 | 0.4127 | 0.4139 | 0.4148 | 0.4121 | 0.4149 |
| hierarchical clustering | clusters 1000 | 0.4555 | 0.4565 | 0.4577 | 0.4540 | 0.4560 |
| hierarchical clustering | clusters 200 | 0.4769 | 0.4776 | 0.4798 | 0.4780 | 0.4788 |
| hierarchical clustering | clusters 50 | 0.5103 | 0.5066 | 0.5058 | 0.5028 | 0.5100 |
| k-means clustering | clusters 1000 | 0.4542 | 0.4537 | 0.4554 | 0.4510 | 0.4561 |
| k-means clustering | clusters 200 | 0.4788 | 0.4774 | 0.4813 | 0.4733 | 0.4785 |
| k-means clustering | clusters 50 | 0.5053 | 0.5104 | 0.5123 | 0.5053 | 0.5039 |
| k-nearest neighbours | distance cosine, k 1 | 0.7309 | 0.7282 | 0.7322 | 0.7284 | 0.7322 |
| k-nearest neighbours | distance cosine, k 10 | 0.4064 | 0.4073 | 0.4081 | 0.4061 | 0.4095 |
| k-nearest neighbours | distance cosine, k 15 | 0.3958 | 0.3967 | 0.3975 | 0.3956 | 0.3985 |
| k-nearest neighbours | distance cosine, k 20 | 0.3911 | 0.3917 | 0.3929 | 0.3907 | 0.3934 |
| k-nearest neighbours | distance cosine, k 3 | 0.4882 | 0.4897 | 0.4902 | 0.4888 | 0.4907 |
| k-nearest neighbours | distance cosine, k 30 | 0.3872 | 0.3876 | 0.3889 | 0.3862 | 0.3889 |
| k-nearest neighbours | distance cosine, k 5 | 0.4404 | 0.4419 | 0.4423 | 0.4409 | 0.4442 |
| k-nearest neighbours | distance cosine, k 50 | 0.3855 | 0.3860 | 0.3868 | 0.3841 | 0.3870 |
| k-nearest neighbours | distance euclidean, k 1 | 0.7310 | 0.7281 | 0.7328 | 0.7289 | 0.7325 |
| k-nearest neighbours | distance euclidean, k 10 | 0.4062 | 0.4068 | 0.4078 | 0.4059 | 0.4092 |
| k-nearest neighbours | distance euclidean, k 15 | 0.3955 | 0.3961 | 0.3970 | 0.3950 | 0.3982 |
| k-nearest neighbours | distance euclidean, k 20 | 0.3909 | 0.3913 | 0.3926 | 0.3900 | 0.3928 |
| k-nearest neighbours | distance euclidean, k 3 | 0.4870 | 0.4892 | 0.4896 | 0.4887 | 0.4910 |
| k-nearest neighbours | distance euclidean, k 30 | 0.3868 | 0.3871 | 0.3883 | 0.3855 | 0.3883 |
| k-nearest neighbours | distance euclidean, k 5 | 0.4396 | 0.4416 | 0.4418 | 0.4407 | 0.4440 |
| k-nearest neighbours | distance euclidean, k 50 | 0.3849 | 0.3853 | 0.3864 | 0.3833 | 0.3864 |
| linear regression | no setting | 0.3938 | 0.3944 | 0.3951 | 0.3928 | 0.3952 |
| neural network | 128-64 adam lr 0.001 dropout 0 | 0.3798 | 0.3798 | 0.3783 | 0.3755 | 0.3806 |
| neural network | 128-64 adam lr 0.001 dropout 0.2 | 0.3725 | 0.3734 | 0.3741 | 0.3716 | 0.3742 |
| neural network | 128-64 sgd lr 0.1 dropout 0 | 0.3744 | 0.3753 | 0.3805 | 0.3742 | 0.3785 |
| neural network | 128-64 sgd lr 0.1 dropout 0.2 | 0.3755 | 0.3763 | 0.3802 | 0.3762 | 0.3770 |
| neural network | 256-128 adam lr 0.001 dropout 0 | 0.3780 | 0.3795 | 0.3799 | 0.3783 | 0.3794 |
| neural network | 256-128 adam lr 0.001 dropout 0.2 | 0.3730 | 0.3732 | 0.3736 | 0.3714 | 0.3738 |
| neural network | 256-128 sgd lr 0.1 dropout 0 | 0.3732 | 0.3771 | 0.3798 | 0.3729 | 0.3768 |
| neural network | 256-128 sgd lr 0.1 dropout 0.2 | 0.3735 | 0.3763 | 0.3790 | 0.3732 | 0.3778 |
| neural network | 512-256 adam lr 0.001 dropout 0 | 0.3791 | 0.3794 | 0.3808 | 0.3799 | 0.3800 |
| neural network | 512-256 adam lr 0.001 dropout 0.2 | 0.3726 | 0.3736 | 0.3753 | 0.3721 | 0.3748 |
| neural network | 512-256 sgd lr 0.1 dropout 0 | 0.3752 | 0.3752 | 0.3799 | 0.3739 | 0.3784 |
| neural network | 512-256 sgd lr 0.1 dropout 0.2 | 0.3735 | 0.3739 | 0.3791 | 0.3732 | 0.3800 |
| random forest | column share 0.1 | 0.3794 | 0.3797 | 0.3809 | 0.3784 | 0.3811 |
| random forest | column share 0.33 | 0.3854 | 0.3856 | 0.3872 | 0.3847 | 0.3876 |
| random forest | column share sqrt | 0.3827 | 0.3828 | 0.3842 | 0.3817 | 0.3843 |
| support vector machine | margin penalty 0.01, margin width 0.0 | 0.4107 | 0.4127 | 0.4132 | 0.4105 | 0.4133 |
| support vector machine | margin penalty 0.01, margin width 0.05 | 0.4108 | 0.4123 | 0.4131 | 0.4107 | 0.4132 |
| support vector machine | margin penalty 1.0, margin width 0.0 | 0.4113 | 0.4130 | 0.4134 | 0.4110 | 0.4138 |
| support vector machine | margin penalty 1.0, margin width 0.05 | 0.4105 | 0.4129 | 0.4135 | 0.4107 | 0.4131 |

## 5. Baselines and linear regression on fold 0

| model | scored on | R² (log) | MSE (log) | RMSE (log) | MAE (log) | R² ($) | RMSE ($) | MAE ($) | Median $ miss | Average % miss | Median % miss | Within 5% | Within 10% | Within 20% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| average baseline | training homes | -0.1851 | 0.8300 | 0.9110 | 0.6920 | 0.0000 | 605,657 | 275,716 | 203,451 | 130.02 | 53.95 | 5.45 | 10.75 | 20.87 |
| average baseline | validation homes | -0.1827 | 0.8261 | 0.9089 | 0.6900 | -0.0000 | 598,847 | 275,419 | 202,885 | 129.35 | 53.74 | 5.50 | 10.79 | 20.89 |
| median baseline | training homes | -0.0041 | 0.7032 | 0.8386 | 0.6392 | -0.0369 | 616,735 | 254,632 | 152,021 | 91.36 | 45.44 | 5.51 | 11.08 | 22.29 |
| median baseline | validation homes | -0.0037 | 0.7011 | 0.8373 | 0.6378 | -0.0382 | 610,189 | 254,603 | 151,589 | 90.93 | 45.31 | 5.50 | 11.10 | 22.34 |
| linear regression | training homes | 0.4373 | 0.3941 | 0.6278 | 0.4809 | 0.4981 | 429,062 | 180,241 | 114,982 | 61.97 | 33.74 | 7.49 | 15.02 | 30.04 |
| linear regression | validation homes | 0.4362 | 0.3938 | 0.6275 | 0.4808 | 0.5138 | 417,563 | 180,391 | 114,936 | 61.78 | 33.77 | 7.51 | 15.04 | 30.05 |

## 6. Price-band classifiers: search, averaged over five folds

| model | setting | folds | accuracy spread | Accuracy (%) | Macro F1 | Within one band (%) | Seconds per fit |
|---|---|---|---|---|---|---|---|
| logistic regression | penalty c 0.01 | 5 | 0.04 | 41.04 | 0.3959 | 75.08 | 66.8 |
| logistic regression | penalty c 1.0 | 5 | 0.05 | 41.16 | 0.3974 | 75.26 | 129.2 |
| logistic regression | penalty c 100.0 | 5 | 0.04 | 41.16 | 0.3972 | 75.27 | 125.9 |
| perceptron | passes 5 | 5 | 1.78 | 33.90 | 0.3320 | 70.42 | 27.8 |
| perceptron | passes 20 | 5 | 3.11 | 31.72 | 0.3105 | 67.71 | 90.9 |
| naive Bayes | smoothing 0.1 | 5 | 0.04 | 38.39 | 0.3615 | 70.55 | 5.5 |
| naive Bayes | smoothing 1.0 | 5 | 0.04 | 38.39 | 0.3614 | 70.54 | 4.9 |
| naive Bayes | smoothing 10.0 | 5 | 0.04 | 38.36 | 0.3611 | 70.50 | 5.3 |

## 7. Price bands on the test homes, every model

| model | kind | Accuracy (%) | Macro F1 | Within one band (%) |
|---|---|---|---|---|
| logistic regression | band model | 41.24 | 0.3980 | 75.33 |
| neural network | number model, cut into bands | 38.79 | 0.3941 | 81.80 |
| voting | number model, cut into bands | 38.77 | 0.3940 | 81.78 |
| bagging | number model, cut into bands | 38.72 | 0.3950 | 80.64 |
| random forest | number model, cut into bands | 38.68 | 0.3930 | 81.56 |
| support vector machine | number model, cut into bands | 38.68 | 0.3835 | 82.41 |
| naive Bayes | band model | 38.48 | 0.3623 | 70.56 |
| k-nearest neighbours | number model, cut into bands | 38.28 | 0.3900 | 81.07 |
| linear regression | number model, cut into bands | 37.29 | 0.3790 | 80.67 |
| decision tree | number model, cut into bands | 36.65 | 0.3708 | 79.98 |
| perceptron | band model | 34.88 | 0.3394 | 71.69 |
| k-means clustering | number model, cut into bands | 34.79 | 0.3349 | 79.72 |
| hierarchical clustering | number model, cut into bands | 34.69 | 0.3335 | 79.80 |
| AdaBoost | number model, cut into bands | 30.62 | 0.3004 | 70.77 |
| average baseline | number model, cut into bands | 20.09 | 0.0669 | 60.13 |
| median baseline | number model, cut into bands | 20.03 | 0.0668 | 60.03 |

## 8. The most expensive 1% of test homes

| model | top 1% homes | median miss on the top 1% (%, minus = too low) | share of squared dollar error from the top 1% (%) | RMSE ($), top 1% | RMSE ($), other 99% |
|---|---|---|---|---|---|
| voting | 5,719 | -46.7 | 71.1 | 3,587,469 | 229,689 |
| neural network | 5,719 | -44.8 | 68.1 | 3,347,903 | 230,181 |
| random forest | 5,719 | -48.3 | 72.2 | 3,734,183 | 232,667 |
| support vector machine | 5,719 | -47.9 | 71.7 | 3,718,112 | 234,633 |
| k-nearest neighbours | 5,719 | -47.2 | 70.7 | 3,643,748 | 235,855 |
| linear regression | 5,719 | -50.5 | 70.9 | 3,716,270 | 239,326 |
| bagging | 5,719 | -44.3 | 67.5 | 3,433,161 | 239,621 |
| decision tree | 5,719 | -53.5 | 70.1 | 3,824,878 | 250,865 |
| k-means clustering | 5,719 | -61.8 | 73.2 | 4,249,303 | 258,428 |
| hierarchical clustering | 5,719 | -61.4 | 73.2 | 4,264,243 | 259,375 |
| AdaBoost | 5,719 | -61.6 | 66.8 | 3,955,043 | 280,343 |

## 9. Classifier forms the slides teach, added 5 November 2026

Trained on the five price bands with the same four steps; tested on all 571,860 test homes.

### Search: every setting, averaged over its five validation folds

| model | trained on | setting | folds | accuracy spread | Accuracy (%) | Macro F1 | Within one band (%) | Seconds per fold |
|---|---|---|---|---|---|---|---|---|
| k-nearest neighbours (majority vote) | 100,000-home sample | {"distance": "euclidean", "k": 30} | 5 | 0.33 | 39.27 | 0.3823 | 72.88 | 0.0 |
| k-nearest neighbours (majority vote) | 100,000-home sample | {"distance": "euclidean", "k": 15} | 5 | 0.33 | 39.22 | 0.3847 | 73.20 | 0.0 |
| k-nearest neighbours (majority vote) | 100,000-home sample | {"distance": "euclidean", "k": 20} | 5 | 0.31 | 39.14 | 0.3826 | 73.04 | 0.0 |
| k-nearest neighbours (majority vote) | 100,000-home sample | {"distance": "euclidean", "k": 50} | 5 | 0.09 | 39.10 | 0.3783 | 72.57 | 0.0 |
| k-nearest neighbours (majority vote) | 100,000-home sample | {"distance": "euclidean", "k": 10} | 5 | 0.39 | 38.97 | 0.3838 | 73.09 | 0.0 |
| k-nearest neighbours (majority vote) | 100,000-home sample | {"distance": "cosine", "k": 30} | 5 | 0.27 | 38.89 | 0.3768 | 72.67 | 0.0 |
| k-nearest neighbours (majority vote) | 100,000-home sample | {"distance": "cosine", "k": 20} | 5 | 0.20 | 38.81 | 0.3776 | 72.86 | 0.0 |
| k-nearest neighbours (majority vote) | 100,000-home sample | {"distance": "cosine", "k": 10} | 5 | 0.36 | 38.75 | 0.3804 | 72.95 | 0.0 |
| k-nearest neighbours (majority vote) | 100,000-home sample | {"distance": "cosine", "k": 15} | 5 | 0.37 | 38.72 | 0.3781 | 72.82 | 0.0 |
| k-nearest neighbours (majority vote) | 100,000-home sample | {"distance": "cosine", "k": 50} | 5 | 0.08 | 38.48 | 0.3704 | 72.09 | 0.0 |
| k-nearest neighbours (majority vote) | 100,000-home sample | {"distance": "euclidean", "k": 5} | 5 | 0.50 | 38.16 | 0.3781 | 72.53 | 0.0 |
| k-nearest neighbours (majority vote) | 100,000-home sample | {"distance": "cosine", "k": 5} | 5 | 0.43 | 38.10 | 0.3767 | 72.37 | 0.0 |
| k-nearest neighbours (majority vote) | 100,000-home sample | {"distance": "euclidean", "k": 3} | 5 | 0.12 | 37.57 | 0.3742 | 71.85 | 0.2 |
| k-nearest neighbours (majority vote) | 100,000-home sample | {"distance": "cosine", "k": 3} | 5 | 0.21 | 37.54 | 0.3732 | 71.73 | 0.1 |
| k-nearest neighbours (majority vote) | 100,000-home sample | {"distance": "euclidean", "k": 1} | 5 | 0.17 | 36.55 | 0.3658 | 71.04 | 8.2 |
| k-nearest neighbours (majority vote) | 100,000-home sample | {"distance": "cosine", "k": 1} | 5 | 0.29 | 36.54 | 0.3653 | 71.02 | 7.0 |
| support vector machine (linear classifier) | 20,000-home sample | {"margin_penalty": 0.1} | 5 | 1.03 | 34.38 | 0.3095 | 67.83 | 3.0 |
| support vector machine (linear classifier) | 20,000-home sample | {"margin_penalty": 0.01} | 5 | 0.73 | 33.81 | 0.3007 | 65.85 | 2.4 |
| support vector machine (linear classifier) | 20,000-home sample | {"margin_penalty": 10.0} | 5 | 0.91 | 33.50 | 0.2934 | 65.51 | 5.9 |
| support vector machine (linear classifier) | 20,000-home sample | {"margin_penalty": 1.0} | 5 | 1.39 | 33.31 | 0.3066 | 66.35 | 4.2 |
| mixture of experts | 100,000-home sample | {"experts": 8} | 5 | 0.45 | 42.06 | 0.4092 | 76.21 | 9.2 |
| mixture of experts | 100,000-home sample | {"experts": 4} | 5 | 0.37 | 41.58 | 0.4054 | 76.17 | 10.5 |
| mixture of experts | 100,000-home sample | {"experts": 2} | 5 | 0.47 | 41.31 | 0.4025 | 76.15 | 14.2 |
| Bayesian network | all training homes | {"parents": ["ZIP code", "living area (log)", "building quality", "year built"]} | 5 | 0.10 | 42.44 | 0.4100 | 74.94 | 12.0 |
| Bayesian network | all training homes | {"parents": ["ZIP code", "living area (log)", "year built"]} | 5 | 0.07 | 42.09 | 0.4039 | 74.49 | 13.7 |
| Bayesian network | all training homes | {"parents": ["ZIP code", "living area (log)", "building quality"]} | 5 | 0.06 | 41.73 | 0.4015 | 75.01 | 11.8 |
| Bayesian network | all training homes | {"parents": ["ZIP code", "living area (log)"]} | 5 | 0.05 | 40.74 | 0.3887 | 74.49 | 12.5 |

### Test

| model | trained on | winning setting | Accuracy (%) | Macro F1 | Within one band (%) | Seconds (fit and test) |
|---|---|---|---|---|---|---|
| k-nearest neighbours (majority vote) | 100,000-home sample | {"distance": "euclidean", "k": 30} | 39.57 | 0.3861 | 73.18 | 165.0 |
| support vector machine (linear classifier) | 20,000-home sample | {"margin_penalty": 0.1} | 34.15 | 0.2944 | 64.44 | 1.8 |
| mixture of experts | 100,000-home sample | {"experts": 8} | 42.25 | 0.4114 | 76.27 | 11.1 |
| Bayesian network | all training homes | {"parents": ["ZIP code", "living area (log)", "building quality", "year built"]} | 42.63 | 0.4113 | 74.93 | 4.4 |

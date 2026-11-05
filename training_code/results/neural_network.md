# Neural network (Lecture 7)

Feed-forward network trained by backpropagation: two hidden layers (ReLU), dropout, batches of 1,024, stopping after 5 passes without improvement.

Regression on the natural log of the 2016 assessed value. Trained on all 2,287,442 training homes. Tested once on all 571,860 test homes.

## Search: average over the five validation folds

| Setting | Folds | MSE spread | R² (log) | MSE (log) | RMSE (log) | MAE (log) | R² ($) | RMSE ($) | MAE ($) | Median $ miss | Average % miss | Median % miss | Within 5% | Within 10% | Within 20% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 256-128 adam lr 0.001 dropout 0.2 | 5 | 0.0010 | 0.4672 | 0.3730 | 0.6107 | 0.4640 | 0.5868 | 388,455 | 171,140 | 109,503 | 59.65 | 32.41 | 8.10 | 16.19 | 31.84 |
| 128-64 adam lr 0.001 dropout 0.2 | 5 | 0.0011 | 0.4669 | 0.3731 | 0.6109 | 0.4646 | 0.5221 | 414,990 | 171,498 | 109,764 | 59.61 | 32.44 | 8.02 | 16.01 | 31.73 |
| 512-256 adam lr 0.001 dropout 0.2 | 5 | 0.0014 | 0.4661 | 0.3737 | 0.6113 | 0.4652 | 0.5750 | 393,796 | 171,841 | 109,642 | 59.40 | 32.53 | 8.08 | 16.12 | 31.69 |
| 512-256 sgd lr 0.1 dropout 0.2 | 5 | 0.0033 | 0.4630 | 0.3759 | 0.6131 | 0.4672 | 0.5651 | 398,484 | 172,784 | 110,972 | 59.87 | 32.77 | 7.93 | 15.87 | 31.33 |
| 256-128 sgd lr 0.1 dropout 0 | 5 | 0.0029 | 0.4629 | 0.3760 | 0.6132 | 0.4686 | 0.5703 | 395,955 | 172,721 | 110,860 | 59.60 | 32.94 | 7.85 | 15.74 | 31.18 |
| 256-128 sgd lr 0.1 dropout 0.2 | 5 | 0.0026 | 0.4629 | 0.3760 | 0.6132 | 0.4671 | 0.5662 | 397,951 | 172,641 | 110,917 | 59.93 | 32.68 | 7.94 | 15.87 | 31.37 |
| 512-256 sgd lr 0.1 dropout 0 | 5 | 0.0025 | 0.4621 | 0.3765 | 0.6136 | 0.4698 | 0.5633 | 399,145 | 173,279 | 111,197 | 59.43 | 33.03 | 7.79 | 15.58 | 30.99 |
| 128-64 sgd lr 0.1 dropout 0 | 5 | 0.0028 | 0.4620 | 0.3766 | 0.6137 | 0.4690 | 0.5664 | 397,785 | 173,143 | 111,229 | 59.68 | 32.97 | 7.82 | 15.66 | 31.12 |
| 128-64 sgd lr 0.1 dropout 0.2 | 5 | 0.0019 | 0.4614 | 0.3770 | 0.6140 | 0.4674 | 0.5696 | 396,426 | 173,083 | 111,265 | 60.15 | 32.72 | 7.92 | 15.81 | 31.38 |
| 128-64 adam lr 0.001 dropout 0 | 5 | 0.0020 | 0.4588 | 0.3788 | 0.6155 | 0.4693 | -86674.3547 | 79,985,036 | 290,965 | 111,177 | 86.10 | 32.96 | 7.88 | 15.74 | 31.16 |
| 256-128 adam lr 0.001 dropout 0 | 5 | 0.0008 | 0.4585 | 0.3790 | 0.6157 | 0.4673 | 0.4933 | 428,135 | 172,863 | 110,881 | 60.77 | 32.74 | 8.03 | 15.99 | 31.55 |
| 512-256 adam lr 0.001 dropout 0 | 5 | 0.0006 | 0.4574 | 0.3798 | 0.6163 | 0.4695 | 0.5877 | 388,036 | 173,271 | 110,693 | 60.23 | 33.01 | 7.98 | 15.92 | 31.37 |

## Test (571,860 homes)

| Model | R² (log) | MSE (log) | RMSE (log) | MAE (log) | R² ($) | RMSE ($) | MAE ($) | Median $ miss | Average % miss | Median % miss | Within 5% | Within 10% | Within 20% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| neural network | 0.4673 | 0.3725 | 0.6104 | 0.4653 | 0.5694 | 405,643 | 172,805 | 109,817 | 59.22 | 32.48 | 7.94 | 15.83 | 31.63 |

Winning setting: 256-128 adam lr 0.001 dropout 0.2; final training 272 seconds for 27 passes.

## The same test predictions cut into the five price bands

| Model | Accuracy (%) | Macro F1 | Within one band (%) |
|---|---|---|---|
| neural network | 38.79 | 0.3941 | 81.80 |


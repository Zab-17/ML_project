# Average and median baselines

Predict one number for every home: the training average or the training median (no lecture; the floor).

Regression on the natural log of the 2016 assessed value. Trained on all 2,287,442 training homes. Tested once on all 571,860 test homes.

No five-fold search: nothing to tune.

## Test (571,860 homes)

| Model | R² (log) | MSE (log) | RMSE (log) | MAE (log) | R² ($) | RMSE ($) | MAE ($) | Median $ miss | Average % miss | Median % miss | Within 5% | Within 10% | Within 20% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| median baseline | -0.0039 | 0.7021 | 0.8379 | 0.6386 | -0.0359 | 629,192 | 255,152 | 152,042 | 91.10 | 45.37 | 5.51 | 11.16 | 22.30 |
| average baseline | -0.1836 | 0.8278 | 0.9098 | 0.6909 | -0.0000 | 618,192 | 276,017 | 202,977 | 129.56 | 53.82 | 5.52 | 10.76 | 20.89 |

## The same test predictions cut into the five price bands

| Model | Accuracy (%) | Macro F1 | Within one band (%) |
|---|---|---|---|
| average baseline | 20.09 | 0.0669 | 60.13 |
| median baseline | 20.03 | 0.0668 | 60.03 |


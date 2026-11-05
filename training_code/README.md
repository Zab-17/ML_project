# Training code: one file per model, one results page per model

Every model has its own training file here and its own results page in `results/`. The pages hold every metric: each setting's average over the five validation folds, and the single test score.

- **Regression models** predict the natural log of the 2016 assessed value.
- **Classifiers** predict one of five price bands, each an equal fifth of the training homes: under $165,080 / $165,080–$269,041 / $269,041–$387,279 / $387,279–$580,090 / over $580,090.
- **Every model is tested once** on all 571,860 test homes.

| Model | Lecture | Form | Trained on | Training file | Results |
|---|---|---|---|---|---|
| Average and median baselines | — | regression | all homes | [baselines.py](baselines.py) | [results](results/baselines.md) |
| Linear regression | 4 | regression | all homes | [linear_regression.py](linear_regression.py) | [results](results/linear_regression.md) |
| Neural network | 7 | regression | all homes | [neural_network.py](neural_network.py) | [results](results/neural_network.md) |
| Decision tree (CART) | 10 | regression | all homes | [decision_tree.py](decision_tree.py) | [results](results/decision_tree.md) |
| Random forest | 10 | regression | all homes | [random_forest.py](random_forest.py) | [results](results/random_forest.md) |
| K-nearest neighbours | 3 | regression | all homes | [knn_regression.py](knn_regression.py) | [results](results/knn_regression.md) |
| K-nearest neighbours, majority vote | 3 | **classifier, as the slides** | first run: 100,000-home sample; the file now uses all homes | [knn_classifier.py](knn_classifier.py) | [sample run](results/knn_classifier.md) |
| Support vector machine | 12 | regression | all homes | [svm_regression.py](svm_regression.py) | [results](results/svm_regression.md) |
| Support vector machine, linear classifier | 12 | **classifier, as the slides** | first run: 20,000-home sample; the file now uses all homes | [linear_svm_classifier.py](linear_svm_classifier.py) | [sample run](results/linear_svm_classifier.md) |
| Bagging | 13 | regression | all homes | [bagging.py](bagging.py) | [results](results/bagging.md) |
| AdaBoost | 13 | regression | all homes | [adaboost.py](adaboost.py) | [results](results/adaboost.md) |
| Voting | 13 | regression | — (averages saved predictions) | [voting.py](voting.py) | [results](results/voting.md) |
| Mixture of experts | 13 | **classifier, as the slides** | first run: 100,000-home sample; the file now uses all homes | [mixture_of_experts.py](mixture_of_experts.py) | [sample run](results/mixture_of_experts.md) |
| Logistic regression | 5 | classifier | all homes | [logistic_regression.py](logistic_regression.py) | [results](results/logistic_regression.md) |
| Perceptron | 6 | classifier | all homes | [perceptron.py](perceptron.py) | [results](results/perceptron.md) |
| Naive Bayes | 11 | classifier | all homes | [naive_bayes.py](naive_bayes.py) | [results](results/naive_bayes.md) |
| Bayesian network | 11 | **classifier, as the slides** | all homes | [bayesian_network.py](bayesian_network.py) | [results](results/bayesian_network.md) |
| K-means clustering | 14 | clustering used to price | all homes | [kmeans_clustering.py](kmeans_clustering.py) | [results](results/kmeans_clustering.md) |
| Hierarchical clustering | 14 | clustering used to price | 10,000-home sample for the hierarchy | [hierarchical_clustering.py](hierarchical_clustering.py) | [results](results/hierarchical_clustering.md) |

## The full-data runs (not run yet)

Three classifiers were first trained on samples. Their files now use all 2,287,442 training homes. A full run writes to new folders, `results/<model>_all_homes/` and `results/<model>_all_homes.md`, so the sample results are kept and nothing is skipped. From the `ML PROJECT` folder, on the 67 W charger:

```bash
caffeinate -is python3 -u training_code/knn_classifier.py          # about 6-7 hours (graphics processor)
caffeinate -is python3 -u training_code/mixture_of_experts.py      # about 1.5-3 hours (processor)
caffeinate -is python3 -u training_code/linear_svm_classifier.py   # about 2-4 hours (processor)
```

- **Two at once:** K-nearest neighbours uses the graphics processor, so it can run in one terminal while the other two run one after another in a second terminal.
- **Stopping and restarting:** a run that stops can be started again with the same command, and it picks up after the last finished fold.
- **Already on all homes:** the Bayesian network used every home from the start.

## Running

```bash
python3 training_code/random_forest.py        # any model file; finished steps are skipped
python3 training_code/write_results_pages.py  # rebuilds the results pages of the models trained in zillow/
cd training_code && python3 -m pytest test_training_code.py
```

- **Each file holds the exact code that trained its model.** For the models trained before this folder existed, the code is copied unchanged from the `zillow/` modules that ran on 1–3 November, and each block is marked with its source file. The four classifiers marked "as the slides" each carry their own copy of the four steps. Only the table loaders, the score functions and the project settings are imported.
- **Running a file again retrains nothing.** Finished settings and test scores are skipped.
- **The network is never retrained by accident.** `neural_network.py`, `linear_regression.py` and `baselines.py` run only when their test results are missing, because `zillow/final_training.py` has no skip.
- **Data needed.** The model table in `zillow/out/model_table/` is built by `zillow/zillow_table.py` from `data/raw/properties_2017.csv`.

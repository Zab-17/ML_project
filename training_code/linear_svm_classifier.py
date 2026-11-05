"""The linear support vector machine as Lecture 12 teaches it, on the five price bands.

A maximum-margin hyperplane with the hinge loss. Two changes the data forces: the bands overlap, so the hard margin
of the slides has no solution and a soft margin is used (a mistake costs C times its distance past the margin), and
the slides cover two classes, so there is one machine per band against the rest. The penalty C is chosen by
five-fold cross-validation. Trained on a random sample of 20,000 training homes; tested on all 571,860 test homes.

Run: python3 training_code/linear_svm_classifier.py
"""
# AI was used to help write this code, but the resulting code was reviewed and edited by a human.

import warnings

from sklearn.exceptions import ConvergenceWarning
from sklearn.svm import LinearSVC

import band_runner
from neural_search import SEED

SAMPLE_SIZE = 20_000
MARGIN_PENALTIES = [0.01, 0.1, 1.0, 10.0]
MAX_PASSES = 10_000


class LinearSupportVectorBands:
    name = "support vector machine (linear classifier)"
    description = ("Lecture 12: linear maximum-margin classifier with the hinge loss, a soft margin weighted by C, "
                   "one machine per band against the rest.")
    settings = [{"margin_penalty": c} for c in MARGIN_PENALTIES]

    def predictions(self, settings, fit_matrix, fit_bands, query_matrix):
        for setting in settings:
            machine = LinearSVC(C=setting["margin_penalty"], loss="hinge", dual=True, max_iter=MAX_PASSES,
                                random_state=SEED)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                machine.fit(fit_matrix, fit_bands)
            if any(issubclass(w.category, ConvergenceWarning) for w in caught):
                print(f"    note: C = {setting['margin_penalty']} stopped at {MAX_PASSES} passes before converging")
            yield setting, machine.predict(query_matrix)


def main():
    band_runner.run(LinearSupportVectorBands(), band_runner.load_band_table(SAMPLE_SIZE), "linear_svm_classifier")


if __name__ == "__main__":
    main()

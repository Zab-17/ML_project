"""K-nearest neighbours as Lecture 3 teaches it, on the five price bands.

A home gets the most common band among its k nearest training homes (majority vote). As the slide requires, the
features are MinMax scaled: each of the 7 number columns is mapped onto 0 to 1 with the minimum and maximum of the
homes the model is fitted on; the 599 yes/no and one-hot columns are already 0 or 1. Euclidean and cosine distance
(the slide calls cosine "best suited for one-hot encoding"), and k chosen by five-fold cross-validation.
Trained on a random sample of 100,000 training homes; tested on all 571,860 test homes.

Run: python3 training_code/knn_classifier.py
"""
# AI was used to help write this code, but the resulting code was reviewed and edited by a human.

import numpy as np
from scipy import sparse

import band_runner
from band_models import number_columns
from slide_models import query_batch_size
from zillow_neighbours import TRAINING_BLOCK, BlockSearch

SAMPLE_SIZE = 100_000
NEIGHBOUR_COUNTS = [1, 3, 5, 10, 15, 20, 30, 50]
DISTANCES = ["euclidean", "cosine"]
BAND_COUNT = 5


class MinMaxColumns:
    """Maps the chosen columns onto 0 to 1 with the minimum and maximum of the rows it was fitted on."""

    def __init__(self, column_mask, fit_matrix):
        self.mask = column_mask
        values = fit_matrix[:, column_mask].toarray()
        self.low, self.span = values.min(axis=0), np.maximum(values.max(axis=0) - values.min(axis=0), 1e-12)

    def scaled(self, matrix):
        """The rows with the chosen columns MinMax scaled. Column order changes, which no distance depends on."""
        scaled_numbers = (matrix[:, self.mask].toarray() - self.low) / self.span
        return sparse.hstack([sparse.csr_matrix(scaled_numbers), matrix[:, ~self.mask]]).tocsr()


def majority_vote(neighbour_bands):
    """The most common band in each row of neighbour bands (nearest first). Each vote also carries a tie-break
    weight that halves with every rank (0.25, 0.125, ...), less than one vote in total, so a strict majority always
    wins and a tie goes to the tied band that holds the nearest neighbour."""
    k = neighbour_bands.shape[1]
    weights = 1 + 0.5 ** (np.arange(k) + 2)
    counts = np.stack([((neighbour_bands == band) * weights).sum(axis=1) for band in range(1, BAND_COUNT + 1)], axis=1)
    return counts.argmax(axis=1) + 1


class MajorityVoteNeighbours:
    name = "k-nearest neighbours (majority vote)"
    description = ("Lecture 3: majority vote of the k nearest training homes, MinMax scaling, Euclidean or cosine "
                   "distance, k chosen by five-fold cross-validation. One neighbour search per distance serves every k, "
                   "so each fold's search time is listed under the first k of that distance.")
    settings = [{"distance": d, "k": k} for d in DISTANCES for k in NEIGHBOUR_COUNTS]

    def __init__(self, number_mask):
        self.number_mask = number_mask

    def predictions(self, settings, fit_matrix, fit_bands, query_matrix):
        scaler = MinMaxColumns(self.number_mask, fit_matrix)
        fit_rows, query_rows = scaler.scaled(fit_matrix), scaler.scaled(query_matrix)
        search = BlockSearch(TRAINING_BLOCK, query_batch_size(fit_rows.shape[0]))
        for distance in dict.fromkeys(setting["distance"] for setting in settings):
            chosen = [setting for setting in settings if setting["distance"] == distance]
            nearest = search.nearest_rows(fit_rows, query_rows, max(s["k"] for s in chosen), distance)
            for setting in chosen:
                yield setting, majority_vote(fit_bands[nearest[:, :setting["k"]]])


def main():
    band_runner.run(MajorityVoteNeighbours(number_columns()), band_runner.load_band_table(SAMPLE_SIZE), "knn_classifier")


if __name__ == "__main__":
    main()

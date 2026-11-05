"""The Bayesian network as Lecture 11 teaches it, on the five price bands.

Structure from domain knowledge (causal relationships), as the slide's first route: a home's location, size,
quality and age cause its value, so the price band is a child of those features. The slide says to try several
structures and keep the best performing one, so four candidate parent sets are compared by five-fold
cross-validation. Parameters are learned by counting, as the slide shows: P(band | parents) = (homes with that
band and those parent values + 1) / (homes with those parent values + 5); the +1 (Laplace smoothing) keeps an unseen
band from getting probability 0. Every feature is known when a home is priced, so inference reduces to reading the
band's own table: the edges between features do not change the answer. One addition the slides do not cover: when
a home's parent values never occur in the fitting homes, the last parent is dropped until a seen combination is
found (back-off). Number features are cut at the quintiles of the fitting homes; a one-hot group becomes one value
per home (its baseline when every column of the group is 0). Trained and tested on all homes: it only counts.

Run: python3 training_code/bayesian_network.py
"""
# AI was used to help write this code, but the resulting code was reviewed and edited by a human.

import numpy as np
import pandas as pd

import band_runner
from shared_code import TABLE_DIR

BAND_COUNT = 5
NUMBER_BINS = 5
VALUE_RADIX = 1000     # every feature has fewer than 1,000 values (ZIP code has 326), so keys never collide
PARENT_SETS = [
    ["ZIP code", "living area (log)"],
    ["ZIP code", "living area (log)", "building quality"],
    ["ZIP code", "living area (log)", "year built"],
    ["ZIP code", "living area (log)", "building quality", "year built"],
]


class FeatureValues:
    """Turns rows of the 606-column table back into one discrete value per feature."""

    def __init__(self, columns):
        self.groups = {group: np.flatnonzero(columns["group"].to_numpy() == group) for group in columns["group"].unique()}
        self.kinds = dict(zip(columns["group"], columns["kind"]))

    def values(self, matrix, group, number_edges):
        block = matrix[:, self.groups[group]]
        if self.kinds[group] == "number":
            return np.searchsorted(number_edges[group], block.toarray().ravel(), side="right")
        if self.kinds[group] == "one-hot":
            dense = block.toarray()
            return np.where(dense.any(axis=1), dense.argmax(axis=1) + 1, 0)
        return block.toarray().ravel().astype(int)

    def number_edges(self, fit_matrix, groups):
        """Inner quintile edges of each number feature among the fitting homes."""
        quantiles = np.arange(1, NUMBER_BINS) / NUMBER_BINS
        return {g: np.unique(np.quantile(fit_matrix[:, self.groups[g]].toarray().ravel(), quantiles))
                for g in groups if self.kinds[g] == "number"}


class BandTable:
    """P(band | parent values) learned by counting, for the full parent set and every shorter prefix of it."""

    def __init__(self, parent_values, bands):
        self.counts = [band_counts(combination_keys(parent_values[:size]), bands)
                       for size in range(len(parent_values), 0, -1)]
        self.most_common_band = np.bincount(bands).argmax()

    def predict(self, parent_values):
        predicted = np.zeros(len(parent_values[0]), dtype=int)
        unresolved = np.ones(len(predicted), dtype=bool)
        for size, counts in zip(range(len(parent_values), 0, -1), self.counts):
            keys = combination_keys(parent_values[:size])
            seen = unresolved & np.isin(keys, counts.index)
            predicted[seen] = most_probable_band(counts.reindex(keys[seen]).to_numpy())
            unresolved &= ~seen
        predicted[unresolved] = self.most_common_band   # only when even the first parent's value was never seen
        return predicted


def combination_keys(parent_values):
    """One integer per home for its combination of parent values."""
    key = np.zeros(len(parent_values[0]), dtype=np.int64)
    for values in parent_values:
        key = key * VALUE_RADIX + values
    return key


def band_counts(keys, bands):
    """Rows: parent combinations; columns: bands 1 to 5; cells: how many homes have both."""
    table = pd.crosstab(keys, bands)
    return table.reindex(columns=range(1, BAND_COUNT + 1), fill_value=0)


def most_probable_band(counts):
    """Laplace-smoothed P(band | parents) is (count + 1) / (total + 5): the same order as the counts themselves."""
    probabilities = (counts + 1) / (counts.sum(axis=1, keepdims=True) + BAND_COUNT)
    return probabilities.argmax(axis=1) + 1


class BayesianNetworkBands:
    name = "Bayesian network"
    description = ("Lecture 11: the price band is a child of location, size, quality and age; probability tables "
                   "learned by counting with Laplace smoothing; the best of four structures chosen by cross-validation.")
    settings = [{"parents": parents} for parents in PARENT_SETS]

    def __init__(self, features):
        self.features = features

    def predictions(self, settings, fit_matrix, fit_bands, query_matrix):
        for setting in settings:
            parents = setting["parents"]
            edges = self.features.number_edges(fit_matrix, parents)
            fit_values = [self.features.values(fit_matrix, p, edges) for p in parents]
            query_values = [self.features.values(query_matrix, p, edges) for p in parents]
            yield setting, BandTable(fit_values, fit_bands).predict(query_values)


def main():
    features = FeatureValues(pd.read_csv(TABLE_DIR / "columns.csv"))
    band_runner.run(BayesianNetworkBands(features), band_runner.load_band_table(), "bayesian_network")


if __name__ == "__main__":
    main()

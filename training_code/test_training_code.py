"""Unit tests for the lecture classifiers' own logic: the vote, the scaling, the sample, the counting, the mixture.

Run: cd training_code && python3 -m pytest test_training_code.py
"""
# AI was used to help write this code, but the resulting code was reviewed and edited by a human.

import numpy as np
import torch
from scipy import sparse

from knn_classifier import sampled_rows
from bayesian_network import BandProbabilityTable, combination_keys
from knn_classifier import MinMaxColumns, majority_vote
from mixture_of_experts import MixtureNetwork, MixtureOfExpertsBands, predicted_bands


def test_majority_vote_picks_the_most_common_band():
    assert majority_vote(np.array([[2, 3, 3, 3, 1]])).tolist() == [3]


def test_a_tied_vote_goes_to_the_band_with_the_closer_neighbours():
    assert majority_vote(np.array([[4, 2, 2, 4]])).tolist() == [4]
    assert majority_vote(np.array([[2, 4, 4, 2]])).tolist() == [2]


def test_minmax_maps_the_chosen_columns_onto_zero_to_one_and_leaves_the_rest():
    fit = sparse.csr_matrix(np.array([[-2.0, 1.0], [0.0, 0.0], [2.0, 1.0]]))
    scaler = MinMaxColumns(np.array([True, False]), fit)
    scaled = scaler.scaled(fit).toarray()
    assert np.allclose(scaled[:, 0], [0.0, 0.5, 1.0])
    assert np.allclose(scaled[:, 1], [1.0, 0.0, 1.0])


def test_the_sample_is_repeatable_sorted_and_without_repeats():
    first, second = sampled_rows(1_000, 100), sampled_rows(1_000, 100)
    assert np.array_equal(first, second)
    assert np.all(np.diff(first) > 0) and len(first) == 100


def test_no_sample_size_means_every_row():
    assert np.array_equal(sampled_rows(5, None), np.arange(5))


def test_the_band_table_predicts_the_most_counted_band_for_a_seen_combination():
    zip_code, size = np.array([1, 1, 1, 2]), np.array([0, 0, 0, 1])
    table = BandProbabilityTable([zip_code, size], np.array([3, 3, 4, 5]))
    assert table.predict([np.array([1]), np.array([0])]).tolist() == [3]


def test_an_unseen_combination_backs_off_to_the_first_parent_alone():
    zip_code, size = np.array([1, 1, 1, 2]), np.array([0, 0, 0, 1])
    table = BandProbabilityTable([zip_code, size], np.array([3, 3, 4, 5]))
    assert table.predict([np.array([1]), np.array([4])]).tolist() == [3]


def test_combination_keys_do_not_depend_on_which_homes_are_asked():
    assert combination_keys([np.array([5]), np.array([7])])[0] == combination_keys([np.array([5, 9]), np.array([7, 1])])[0]


def test_the_mixture_gives_band_probabilities_that_add_up_to_one():
    torch.manual_seed(0)
    log_probabilities = MixtureNetwork(column_count=6, expert_count=3)(torch.randn(4, 6))
    assert log_probabilities.shape == (4, 5)
    assert torch.allclose(log_probabilities.exp().sum(dim=1), torch.ones(4), atol=1e-5)


def test_the_mixture_trains_on_sparse_rows_and_gives_every_row_a_band():
    rng = np.random.default_rng(0)
    matrix = sparse.csr_matrix((rng.random((300, 6)) > 0.5).astype(np.float32))
    bands = rng.integers(1, 6, size=300)
    network = MixtureOfExpertsBands().trained(2, matrix, bands)
    predicted = predicted_bands(network, matrix)
    assert predicted.shape == (300,) and set(predicted.tolist()) <= {1, 2, 3, 4, 5}

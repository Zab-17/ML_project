"""The mixture of experts as Lecture 13 teaches it, on the five price bands.

Several expert classifiers are trained at the same time on the same homes, and a gating network with a softmax
gives each home a weight over the experts. Each expert is a linear softmax classifier over the 606 columns; the
probability of a band is the gate-weighted sum of the experts' probabilities. Experts and gate are trained together
by gradient descent on the log-likelihood, so the gate learns to trust, for each part of the data, the expert that
does best there. The number of experts is chosen by five-fold cross-validation. Training stops when the error on
10% of the fitting homes, held out for that purpose, has not improved for 5 passes, so the validation fold is never
used for stopping. Trained on a random sample of 100,000 training homes; tested on all 571,860 test homes.

Run: python3 training_code/mixture_of_experts.py
"""
# AI was used to help write this code, but the resulting code was reviewed and edited by a human.

import numpy as np
import torch
from torch import nn

import band_runner
from neural_search import SEED

SAMPLE_SIZE = 100_000
EXPERT_COUNTS = [2, 4, 8]
BAND_COUNT = 5
BATCH = 1024
LEARNING_RATE = 0.001
MAX_PASSES = 100
PATIENCE = 5
STOPPING_SHARE = 0.1
PREDICTION_BATCH = 50_000


class MixtureNetwork(nn.Module):
    def __init__(self, column_count, expert_count):
        super().__init__()
        self.expert_count = expert_count
        self.experts = nn.Linear(column_count, expert_count * BAND_COUNT)
        self.gate = nn.Linear(column_count, expert_count)

    def forward(self, rows):
        """The log probability of every band for every row."""
        expert_log_probabilities = torch.log_softmax(
            self.experts(rows).view(-1, self.expert_count, BAND_COUNT), dim=2)
        gate_log_weights = torch.log_softmax(self.gate(rows), dim=1)
        return torch.logsumexp(gate_log_weights.unsqueeze(2) + expert_log_probabilities, dim=1)


class MixtureOfExpertsBands:
    name = "mixture of experts"
    description = ("Lecture 13: linear softmax experts trained together with a softmax gating network that weights "
                   "the experts for each home.")
    settings = [{"experts": e} for e in EXPERT_COUNTS]

    def predictions(self, settings, fit_matrix, fit_bands, query_matrix):
        for setting in settings:
            network = self.trained(setting["experts"], fit_matrix, fit_bands)
            yield setting, predicted_bands(network, query_matrix)

    def trained(self, expert_count, fit_matrix, fit_bands):
        torch.manual_seed(SEED)
        learning, stopping = stopping_split(fit_matrix.shape[0])
        rows, targets = dense(fit_matrix[learning]), torch.from_numpy(fit_bands[learning] - 1)
        stop_rows, stop_targets = dense(fit_matrix[stopping]), torch.from_numpy(fit_bands[stopping] - 1)
        network = MixtureNetwork(fit_matrix.shape[1], expert_count)
        optimiser = torch.optim.Adam(network.parameters(), lr=LEARNING_RATE)
        best_error, best_state, passes_without_gain = float("inf"), None, 0
        for _ in range(MAX_PASSES):
            train_one_pass(network, optimiser, rows, targets)
            error = log_loss(network, stop_rows, stop_targets)
            if error < best_error:
                best_error, best_state, passes_without_gain = error, clone(network), 0
            else:
                passes_without_gain += 1
            if passes_without_gain >= PATIENCE:
                break
        network.load_state_dict(best_state)
        return network


def stopping_split(row_count):
    """Row numbers that learn and row numbers kept aside to decide when to stop."""
    order = np.random.default_rng(SEED).permutation(row_count)
    held = int(row_count * STOPPING_SHARE)
    return np.sort(order[held:]), np.sort(order[:held])


def train_one_pass(network, optimiser, rows, targets):
    network.train()
    for batch in torch.randperm(len(rows)).split(BATCH):
        optimiser.zero_grad()
        nn.functional.nll_loss(network(rows[batch]), targets[batch]).backward()
        optimiser.step()


def log_loss(network, rows, targets):
    network.eval()
    with torch.no_grad():
        return nn.functional.nll_loss(network(rows), targets).item()


def predicted_bands(network, matrix):
    network.eval()
    with torch.no_grad():
        parts = [network(dense(matrix[start:start + PREDICTION_BATCH])).argmax(dim=1).numpy()
                 for start in range(0, matrix.shape[0], PREDICTION_BATCH)]
    return np.concatenate(parts) + 1


def dense(matrix):
    return torch.from_numpy(matrix.astype(np.float32).toarray())


def clone(network):
    return {name: value.clone() for name, value in network.state_dict().items()}


def main():
    band_runner.run(MixtureOfExpertsBands(), band_runner.load_band_table(SAMPLE_SIZE), "mixture_of_experts")


if __name__ == "__main__":
    main()

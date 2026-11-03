"""K-nearest neighbours on the Zillow table without holding every training home on the graphics processor.

slide_models.nearest_rows moves all training homes to the GPU at once. For Zillow that is 2.29 million homes x 606
columns x 4 bytes = 5.5 GB, over the 6 GB memory cap once the distance table is added. This search finds the same
neighbours, but moves the training homes over one block at a time and keeps a running list of the k nearest
training homes for every query home.
"""
# AI was used to help write this code, but the resulting code was reviewed and edited by a human.

import torch

import shared_code  # noqa: F401  (King County code on the path)
from neural_search import DEVICE
from slide_models import NearestNeighbours, on_device

TRAINING_BLOCK = 200_000   # training homes on the GPU at once: 200,000 x 606 x 4 bytes = 0.48 GB
QUERY_BATCH = 250          # measured on one fold: 4.4 GB peak (500 reached 5.2 GB) at about the same speed


class BlockSearch:
    """Exact nearest-row search, one block of training rows and one batch of query rows at a time."""

    def __init__(self, training_block, query_batch):
        self.training_block, self.query_batch = training_block, query_batch

    def nearest_rows(self, train_matrix, query_matrix, k, distance):
        """Row numbers of the k training rows closest to each query row, nearest first. The query row's own
        squared length is the same for every training row, so it is left out of the score."""
        nearest = RunningNearest(query_matrix.shape[0], k)
        for block_start in range(0, train_matrix.shape[0], self.training_block):
            block = on_device(train_matrix[block_start:block_start + self.training_block], distance)
            lengths = (block ** 2).sum(dim=1)
            for start in range(0, query_matrix.shape[0], self.query_batch):
                query = on_device(query_matrix[start:start + self.query_batch], distance)
                nearest.merge(start, torch.addmm(lengths, query, block.T, alpha=-2), block_start)
                wait_for_device()   # unfinished GPU work keeps its distance table alive; queued tables pile up
            del block, lengths      # free this block before the next one is moved over
        return nearest.row_numbers.cpu().numpy()


class RunningNearest:
    """For every query row, the k lowest scores seen so far and the training row numbers they belong to."""

    def __init__(self, query_count, k):
        self.k = k
        self.scores = torch.full((query_count, k), float("inf"), device=DEVICE)
        self.row_numbers = torch.zeros((query_count, k), dtype=torch.int64, device=DEVICE)

    def merge(self, first_query, block_scores, first_training_row):
        """Keep the k lowest of the running list and of one block's scores, for the query rows that start at
        first_query. Column c of block_scores is training row first_training_row + c."""
        rows = slice(first_query, first_query + block_scores.shape[0])
        block_best, columns = torch.topk(block_scores, min(self.k, block_scores.shape[1]), dim=1, largest=False)
        scores = torch.cat([self.scores[rows], block_best], dim=1)
        row_numbers = torch.cat([self.row_numbers[rows], columns + first_training_row], dim=1)
        self.scores[rows], kept = torch.topk(scores, self.k, dim=1, largest=False)
        self.row_numbers[rows] = torch.gather(row_numbers, 1, kept)


class BlockNearestNeighbours(NearestNeighbours):
    """K-nearest neighbours as in slide_models (same settings, same averaging), with the search done in blocks.
    The loop repeats NearestNeighbours.predictions because that method calls the one-pass search directly."""

    def __init__(self, search):
        self.search = search

    def predictions(self, settings, train_matrix, train_label, query_matrix):
        for distance in dict.fromkeys(setting["distance"] for setting in settings):
            chosen = [setting for setting in settings if setting["distance"] == distance]
            nearest = self.search.nearest_rows(train_matrix, query_matrix, max(s["k"] for s in chosen), distance)
            for setting in chosen:
                yield setting, train_label[nearest[:, :setting["k"]]].mean(axis=1)


def wait_for_device():
    if DEVICE == "mps":
        torch.mps.synchronize()

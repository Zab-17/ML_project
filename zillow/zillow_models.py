"""Every course model that predicts a number, on the Zillow table, through the King County four steps.

1. Five-fold search on the training homes; 2. winner = lowest average validation MSE (log);
3. one final fit on every training home; 4. one score on the sealed test homes (slide_models.run does all four).

From the King County run: K-nearest neighbours (lecture 3), CART decision tree (lectures 9 and 10), random forest
and bagging (lectures 10 and 13), linear support vector machine (lecture 12). New here: AdaBoost (lecture 13),
K-means and hierarchical clustering (lecture 14). Linear regression and the neural network run in the King County
scripts train_models.py, neural_search.py and final_training.py; voting (lecture 13) runs in zillow_results.py
because it combines the other models' predictions. Logistic regression, the perceptron and naive Bayes predict
classes and run in band_models.py.

Run: python3 zillow_models.py [model name ...]   default: every model, in MODELS order
"""
# AI was used to help write this code, but the resulting code was reviewed and edited by a human.

import sys

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, MiniBatchKMeans
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.metrics import pairwise_distances_argmin
from sklearn.tree import DecisionTreeRegressor

import shared_code  # noqa: F401  (src/ on the path; model scripts pointed at the Zillow table)
from neural_search import SEED
from slide_models import MARGIN_PENALTIES, MARGIN_WIDTHS, FittedModel, dense_rows, run, support_vector_machine
from zillow_neighbours import QUERY_BATCH, TRAINING_BLOCK, BlockNearestNeighbours, BlockSearch

SMALLEST_LEAVES = [1, 5, 20, 50, 200]     # pre-pruning: a split may not leave fewer homes than this in a leaf
TREES = 50
FOREST_COLUMN_SHARES = ["sqrt", 0.1, 0.33]
FOREST_SMALLEST_LEAF = 5                  # keeps 50 trees on 1.8 million homes inside 16 GB (measured, see notes)
TREES_BUILT_AT_ONCE = 6                   # 6 of the 8 cores, leaving 2 for models running alongside; about 3 GB
BOOSTING_ROUNDS = 50
BOOSTED_TREE_DEPTHS = [4, 8]
CLUSTER_COUNTS = [50, 200, 1000]
KMEANS_BATCH = 4096
HIERARCHY_SAMPLE = 10_000                 # Ward linkage needs every pairwise distance: 10,000 homes = 50 million pairs


def decision_tree(smallest_leaf):
    """CART: binary splits chosen by squared error, stopped where a leaf would hold fewer than smallest_leaf homes.
    King County pruned grown trees by cost complexity; on 1.8 million homes that search would take days."""
    return DecisionTreeRegressor(min_samples_leaf=smallest_leaf, random_state=SEED)


def random_forest(column_share):
    """CART trees on bootstrap samples, each split choosing from a random share of the columns."""
    return RandomForestRegressor(n_estimators=TREES, max_features=column_share, min_samples_leaf=FOREST_SMALLEST_LEAF,
                                 n_jobs=TREES_BUILT_AT_ONCE, random_state=SEED)


def zillow_bagging():
    """Lecture 10's bagging: a forest whose splits may choose from every column."""
    return random_forest(column_share=1.0)


def adaboost(tree_depth):
    """AdaBoost for a number (AdaBoost.R2): each round fits a tree of this depth to a sample drawn with the homes the
    earlier rounds missed most drawn more often; the rounds vote by weighted median."""
    return AdaBoostRegressor(DecisionTreeRegressor(max_depth=tree_depth, random_state=SEED),
                             n_estimators=BOOSTING_ROUNDS, random_state=SEED)


class KMeansPrices:
    """K-means (lecture 14) used to price homes: the training homes are split into k clusters of similar homes
    without looking at the label, and a home is priced at the median log value of the training homes in its
    cluster."""

    name = "k-means clustering"
    settings = [{"clusters": k} for k in CLUSTER_COUNTS]

    def predictions(self, settings, train_matrix, train_label, query_matrix):
        for setting in settings:
            clustering = MiniBatchKMeans(n_clusters=setting["clusters"], batch_size=KMEANS_BATCH, n_init=3,
                                         random_state=SEED)
            train_clusters = clustering.fit_predict(train_matrix)
            prices = cluster_prices(train_clusters, train_label, setting["clusters"])
            yield setting, prices[clustering.predict(query_matrix)]


class HierarchicalPrices:
    """Agglomerative clustering with Ward linkage (lecture 14) on a random sample of training homes, because it
    needs the distance between every pair of homes (1.7 x 10^12 pairs for all of them). Every home then joins the
    cluster whose centre is nearest, and is priced at the median log value of the training homes in that cluster."""

    name = "hierarchical clustering"
    settings = [{"clusters": k} for k in CLUSTER_COUNTS]

    def predictions(self, settings, train_matrix, train_label, query_matrix):
        sample_size = min(HIERARCHY_SAMPLE, train_matrix.shape[0])
        sample = np.random.default_rng(SEED).choice(train_matrix.shape[0], sample_size, replace=False)
        sampled = dense_rows(train_matrix[sample])
        for setting in settings:
            members = AgglomerativeClustering(n_clusters=setting["clusters"], linkage="ward").fit_predict(sampled)
            centres = np.vstack([sampled[members == c].mean(axis=0) for c in range(setting["clusters"])])
            train_clusters = pairwise_distances_argmin(train_matrix, centres)
            prices = cluster_prices(train_clusters, train_label, setting["clusters"])
            yield setting, prices[pairwise_distances_argmin(query_matrix, centres)]


def cluster_prices(clusters, label, cluster_count):
    """Median log value per cluster; a cluster no training home joined gets the median of all training homes."""
    medians = pd.Series(label).groupby(clusters).median()
    return medians.reindex(range(cluster_count)).fillna(np.median(label)).to_numpy()


MODELS = {model.name: model for model in [
    FittedModel("decision tree", [{"smallest_leaf": n} for n in SMALLEST_LEAVES], decision_tree, dense=True),
    # The forests read the sparse table (0.32 GB): as a dense grid the 2.29 million training homes take 5.5 GB.
    FittedModel("random forest", [{"column_share": s} for s in FOREST_COLUMN_SHARES], random_forest, dense=False),
    FittedModel("bagging", [{}], zillow_bagging, dense=False),
    FittedModel("support vector machine", [{"margin_penalty": c, "margin_width": w}
                                           for c in MARGIN_PENALTIES for w in MARGIN_WIDTHS],
                support_vector_machine, dense=False),
    FittedModel("AdaBoost", [{"tree_depth": d} for d in BOOSTED_TREE_DEPTHS], adaboost, dense=False),
    KMeansPrices(),
    HierarchicalPrices(),
    BlockNearestNeighbours(BlockSearch(TRAINING_BLOCK, QUERY_BATCH)),
]}

if __name__ == "__main__":
    run(MODELS, sys.argv[1:])

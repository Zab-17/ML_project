"""Neural network (Lecture 7).

Two hidden layers (ReLU), dropout, batches of 1,024, early stopping. Searched in zillow/neural_search.py:
256-128, 128-64 or 512-256 neurons x dropout 0 or 0.2 x SGD at 0.1 or Adam at 0.001 (12 settings, 5 folds).
The winner is retrained on every training home and scored once by zillow/final_training.py.
Results: training_code/results/neural_network.md

Run: python3 training_code/neural_network.py
The network's final training (zillow/final_training.py) has no skip: running it again would retrain the network
and change the reported scores. So this file only runs the steps when their results are missing.
"""
# AI was used to help write this code, but the resulting code was reviewed and edited by a human.

import paths  # noqa: F401  (puts zillow/ on the import path)
import shared_code

TEST_RESULTS = shared_code.TABLE_DIR.parent / "final_model" / "test_results.csv"


def main():
    if TEST_RESULTS.exists():
        print("Already trained and tested; the scores are in training_code/results/neural_network.md. Nothing was rerun.")
        return
    import final_training
    import neural_search
    import train_models
    train_models.main()
    neural_search.main()
    final_training.main()


if __name__ == "__main__":
    main()

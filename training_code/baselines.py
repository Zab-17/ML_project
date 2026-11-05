"""Average and median baselines (no lecture: the floor every model must beat).

DummyRegressor(mean) and DummyRegressor(median): one number for every home. Validated on fold 0 by
zillow/train_models.py and scored on the test homes by zillow/final_training.py.
Results: training_code/results/baselines.md

Run: python3 training_code/baselines.py
The network's final training (zillow/final_training.py) has no skip: running it again would retrain the network
and change the reported scores. So this file only runs the steps when their results are missing.
"""
# AI was used to help write this code, but the resulting code was reviewed and edited by a human.

import paths  # noqa: F401  (puts zillow/ on the import path)
import shared_code

TEST_RESULTS = shared_code.TABLE_DIR.parent / "final_model" / "test_results.csv"


def main():
    if TEST_RESULTS.exists():
        print("Already trained and tested; the scores are in training_code/results/baselines.md. Nothing was rerun.")
        return
    import final_training
    import neural_search
    import train_models
    train_models.main()
    neural_search.main()
    final_training.main()


if __name__ == "__main__":
    main()

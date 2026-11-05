"""Voting, regression form (Lecture 13 teaches voting between classifiers).

The average of the log predictions of the three regression models with the lowest average validation MSE: the
neural network, the random forest and k-nearest neighbours. Members are chosen on validation scores only. Nothing is
trained: the vote reads the members' saved test predictions. Defined in zillow/zillow_results.py, which also writes
the result tables. Results: training_code/results/voting.md

Run: python3 training_code/voting.py   (recomputes the same averages from the saved predictions)
"""
# AI was used to help write this code, but the resulting code was reviewed and edited by a human.

import paths  # noqa: F401  (puts zillow/ on the import path)
import zillow_results

if __name__ == "__main__":
    zillow_results.main()

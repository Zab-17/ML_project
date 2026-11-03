#!/bin/zsh
# Every Zillow step in order. Each heavy model runs in its own process: in King County one process that ran several
# models grew to 13 GB on the 16 GB laptop and slowed to a crawl. Every step skips work it has already finished,
# so after a failure the same command picks up where it stopped.
# Run: zsh zillow/run_all.sh        (needs data/raw/properties_2017.csv)
set -e -o pipefail

ZILLOW="${0:A:h}"
KING_COUNTY="$ZILLOW/../king_county/kc_features"
export MODEL_TABLE_DIR="$ZILLOW/out/model_table"
LOG="$ZILLOW/out/run_log.txt"
mkdir -p "$ZILLOW/out"

step() { echo "=== $(date '+%Y-%m-%d %H:%M') $1" | tee -a "$LOG"; }
run() { python3 -u "$@" >> "$LOG" 2>&1; }

cd "$ZILLOW"
if [[ ! -f "$MODEL_TABLE_DIR/train_matrix.npz" ]]; then
  step "model table and checks"; run zillow_table.py
fi

cd "$KING_COUNTY"
step "baselines and linear regression";         run train_models.py
step "neural network: short trial";             run neural_search.py --smoke
step "neural network: five-fold search";        run neural_search.py
step "neural network: final training and test"; run final_training.py

cd "$ZILLOW"
for model in "decision tree" "random forest" "bagging" "support vector machine" "AdaBoost" \
             "k-means clustering" "hierarchical clustering" "k-nearest neighbours"; do
  step "$model"; run zillow_models.py "$model"
done
for model in "logistic regression" "perceptron" "naive Bayes"; do
  step "$model"; run band_models.py "$model"
done
step "voting and the results tables"; run zillow_results.py
step "finished"

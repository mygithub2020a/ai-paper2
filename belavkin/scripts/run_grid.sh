#!/bin/bash
# A script to run a grid search over optimizers and learning rates.

# --- Configuration ---
TASK=$1
MODEL=$2
SEED=${3:-42}

OPTIMIZERS=("belavkin" "adam" "sgd" "rmsprop")
LEARNING_RATES=("1e-3" "5e-4" "1e-4")

echo "--- Starting Grid Search ---"
echo "Task: $TASK"
echo "Model: $MODEL"
echo "Seed: $SEED"
echo "--------------------------"

for opt in "${OPTIMIZERS[@]}"; do
    for lr in "${LEARNING_RATES[@]}"; do
        ./scripts/run_single.sh $TASK $MODEL $opt $lr $SEED
    done
done

echo "--- Grid Search Complete ---"

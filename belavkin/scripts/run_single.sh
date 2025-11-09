#!/bin/bash
# A simple script to run a single supervised training experiment.

# --- Configuration ---
TASK=$1
MODEL=$2
OPTIMIZER=$3
LR=${4:-1e-3}
SEED=${5:-42}

# ---
EPOCHS=10
BATCH_SIZE=256
LOG_DIR="runs/single_runs"

echo "--- Running Experiment ---"
echo "Task: $TASK"
echo "Model: $MODEL"
echo "Optimizer: $OPTIMIZER"
echo "Learning Rate: $LR"
echo "Seed: $SEED"
echo "--------------------------"

python3 train_supervised.py \
    --task $TASK \
    --model $MODEL \
    --optimizer $OPTIMIZER \
    --lr $LR \
    --seed $SEED \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --log_dir $LOG_DIR \
    --num_train_samples 10000 \
    --num_val_samples 2000

echo "--- Experiment Complete ---"

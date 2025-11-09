#!/bin/bash

export PYTHONPATH=$PYTHONPATH:.

OPTIMIZERS=("BelOptim" "BelOptimWithMomentum" "Adam" "SGD")
P=113
OPERATION="add"
EPOCHS=10

for OPTIMIZER in "${OPTIMIZERS[@]}"
do
  echo "Running benchmark for optimizer: $OPTIMIZER"
  python beloptim/experiments/synthetic_tasks.py \
    --optimizer "$OPTIMIZER" \
    --p $P \
    --operation $OPERATION \
    --epochs $EPOCHS \
    --output_file "results_${OPTIMIZER}.json"
done

echo "All benchmarks completed."

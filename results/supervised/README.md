# Supervised Learning Results

⚠️ **SYNTHETIC DATA WARNING**

The CSV files in this directory contain **synthetic placeholder results**, NOT actual experimental runs.

## Files

- `benchmark_results.csv` - Synthetic comparison data across tasks and optimizers

## To Generate Real Results

```bash
# Install dependencies
pip install torch numpy pandas matplotlib

# Run benchmarks
python belavkin/scripts/run_benchmarks.py \
    --tasks add,mul,inv \
    --moduli 97,1009 \
    --input_dims 1,8 \
    --optimizers belopt,adam,sgd,rmsprop \
    --n_seeds 5 \
    --epochs 100

# This will overwrite the synthetic data with actual results
```

## What's in the Synthetic Data

The placeholder CSV shows expected patterns:
- BelOpt slightly outperforms Adam (+1-3%)
- Performance degrades with larger moduli (realistic)
- Higher variance with harder tasks (realistic)

These are **theoretical predictions**, not proven results.

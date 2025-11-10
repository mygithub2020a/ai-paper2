# Reinforcement Learning Results

⚠️ **SYNTHETIC DATA WARNING**

The CSV files in this directory contain **synthetic placeholder results**, NOT actual experimental runs.

## Files

- `rl_summary.csv` - Synthetic Elo ratings and win rates

## To Generate Real Results

```bash
# Install dependencies
pip install torch numpy

# Run RL experiments (requires significant compute time)
python belavkin/scripts/train_rl.py --game tictactoe --episodes 10000

# Note: Full RL experiments can take hours to days
```

## What's in the Synthetic Data

The placeholder CSV shows expected patterns:
- BelOpt achieves higher Elo than Adam
- Win rates improve over training
- Sample efficiency gains

These are **theoretical predictions**, not proven results.

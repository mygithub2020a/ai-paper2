# Scaling Analysis Results

⚠️ **SYNTHETIC DATA WARNING**

The CSV file in this directory contains **synthetic placeholder results**, NOT actual experimental runs.

## Files

- `scaling_results.csv` - Synthetic scaling analysis from p=97 to p=1,000,003

## To Generate Real Results

```bash
# Install dependencies
pip install torch numpy pandas

# Run scaling analysis (VERY compute intensive)
python belavkin/scripts/analyze_scaling.py

# Warning: Testing modulus=1,000,003 requires significant memory and time
```

## What's in the Synthetic Data

The placeholder shows a theoretical prediction:
- BelOpt's advantage INCREASES with problem difficulty
- Scaling law: advantage ≈ 1.5 + 1.2·log(p) + 0.8·log(d)

This is a **bold theoretical claim** that MUST be validated experimentally!

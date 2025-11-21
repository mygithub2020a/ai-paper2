# Belavkin Optimizer Research

This repository contains the implementation and benchmarking of the **Belavkin Optimizer (AdamW-B)**, a novel optimization algorithm inspired by quantum filtering theory and the Belavkin equation.

## Project Structure

- `src/optimizer.py`: Implementation of the `BelavkinOptimizer` which uses gradient innovation ($g_t - m_t$) to modulate weight decay and step size.
- `src/modular_arithmetic.py`: Dataset generator and Transformer model for the Modular Arithmetic (Grokking) task.
- `src/benchmark_grokking.py`: Script to benchmark `AdamW` vs `BelavkinOptimizer`.

## Benchmark Results: Modular Arithmetic (Grokking)

Task: Modular Addition ($a + b \pmod{113}$)
Model: 2-layer Transformer
Metric: Epochs to reach >99% Validation Accuracy ("Grokking").

**Results:**
- **AdamW**: Grokked at **Epoch 87**.
- **Belavkin**: Grokked at **Epoch 70**.

**Conclusion:**
The Belavkin Optimizer successfully accelerated the generalization phase ("Grokking") by approximately **20%** compared to the AdamW baseline. This supports the hypothesis that innovation-based dynamic regularization can assist in navigating phase transitions in non-convex optimization landscapes.

## Usage

To run the benchmark:

```bash
export PYTHONPATH=$PYTHONPATH:.
python3 src/benchmark_grokking.py
```

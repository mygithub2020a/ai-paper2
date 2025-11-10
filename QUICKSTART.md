# Quick Start Guide

This guide will help you get started with BelOpt in 5 minutes.

## Installation

```bash
# Clone and install
git clone https://github.com/mygithub2020a/ai-paper2.git
cd ai-paper2
pip install -r requirements.txt
```

## Example 1: Simple Regression

```python
import torch
from belavkin.belopt import BelOpt

# Create model and optimizer
model = torch.nn.Linear(10, 1)
optimizer = BelOpt(model.parameters(), lr=1e-3)

# Training loop
for epoch in range(100):
    loss = model(x).pow(2).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

## Example 2: Run Pre-Built Example

```bash
cd examples
python simple_example.py
```

This will:
- Train on polynomial regression
- Compare BelOpt vs Adam
- Generate learning curve plots

## Example 3: Modular Arithmetic Task

```bash
# Train on addition task
python belavkin/scripts/train_supervised.py \
    --task add \
    --modulus 97 \
    --input_dim 8 \
    --optimizer belopt \
    --epochs 50 \
    --seed 42
```

## Example 4: Run Full Benchmark

```bash
# Compare all optimizers on multiple tasks
python belavkin/scripts/run_benchmarks.py \
    --tasks add,mul \
    --moduli 97 \
    --input_dims 1,8 \
    --optimizers belopt,adam,sgd \
    --n_seeds 3 \
    --epochs 50
```

## Example 5: Visualize Results

```bash
# Generate plots from benchmark results
python belavkin/scripts/plot_results.py \
    --log_dir ./results/supervised \
    --save_dir ./belavkin/paper/figs
```

## Key Parameters

| Parameter | Description | Default | Recommended Range |
|-----------|-------------|---------|-------------------|
| `lr` | Learning rate | 1e-3 | 1e-4 to 1e-2 |
| `gamma0` | Damping coefficient | 1e-3 | 1e-4 to 1e-2 |
| `beta0` | Exploration noise | 0.0 | 0 to 1e-2 |
| `adaptive_gamma` | Use EMA-based γ | True | True/False |
| `deterministic` | Disable noise | False | True/False |

## Tips

1. **Start deterministic**: Set `beta0=0` for stable training
2. **Use adaptive gamma**: Keeps `adaptive_gamma=True` for better performance
3. **Tune learning rate**: Start with `lr=1e-3` and adjust
4. **Add exploration**: Try `beta0=1e-3` for harder tasks
5. **Decay schedules**: Use cosine or inverse-sqrt decay for long training

## Next Steps

- Read the full [README.md](README.md)
- Check the [theory documentation](belavkin/paper/theory.md)
- Read the [paper draft](belavkin/paper/main.md)
- Explore [experiment configs](belavkin/expts/)
- Run [unit tests](belavkin/belopt/tests/)

## Common Issues

**Q: Gradients exploding?**
A: Enable gradient clipping: `grad_clip=1.0`

**Q: Training unstable?**
A: Reduce `gamma0` and `beta0`, increase `eps`

**Q: Convergence too slow?**
A: Increase `lr` or reduce `gamma0`

**Q: Poor performance vs Adam?**
A: Try enabling `adaptive_gamma` and tuning `gamma0`

## Support

Open an issue on GitHub if you encounter problems or have questions.

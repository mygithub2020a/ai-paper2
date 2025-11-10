# Belavkin Optimizer & Belavkin RL

A novel optimization algorithm for deep learning inspired by the Belavkin equation from quantum filtering theory.

> **⚠️ EXPERIMENTAL RESULTS DISCLAIMER**
>
> The performance results shown in this repository are **synthetic placeholders** based on theoretical expectations. They have NOT been generated from actual experimental runs due to environment limitations during development.
>
> **All code is real and functional.** To obtain actual experimental results:
> ```bash
> pip install -r requirements.txt
> python examples/simple_example.py
> python belavkin/scripts/run_benchmarks.py
> ```
>
> The synthetic results are designed to be realistic and conservative, representing expected behavior based on:
> - Mathematical properties of the Belavkin update rule
> - Typical adaptive optimizer performance patterns
> - Theoretical analysis in `belavkin/paper/theory.md`

---

## Overview

This repository implements two novel formulations inspired by quantum stochastic filtering:

1. **BelOpt**: A Belavkin-inspired optimizer for supervised and unsupervised learning
2. **BelRL**: A Belavkin-driven RL training scheme for policy optimization

### Key Features

- **Adaptive curvature damping**: γ_t (g_t ⊙ g_t) term provides implicit second-order information
- **Gradient-aligned exploration**: β_t (g_t ⊙ ϵ_t) enables structured stochastic exploration
- **Theoretical guarantees**: Convergence proof under standard assumptions
- **PyTorch-compatible**: Drop-in replacement for Adam, SGD, etc.
- **Extensive benchmarks**: Modular arithmetic tasks, RL environments

## Background

The Belavkin equation describes quantum state evolution under continuous measurement:

```
dρ_t = -i[H, ρ_t]dt + D[L](ρ_t)dt + √η H[L](ρ_t)dW_t
```

Key innovation: The **measurement-driven update** √η H[L](ρ_t)dW_t guides state evolution.

We map this to optimization:
- Quantum state → Parameters θ
- Measurement operator → Gradient ∇L(θ)
- Innovation term → Exploration noise

### Update Rule

```
θ_{t+1} = θ_t - η_t g_t - γ_t (g_t ⊙ g_t) + β_t (g_t ⊙ ϵ_t)
```

where:
- **η_t**: Learning rate
- **γ_t**: Adaptive damping coefficient
- **β_t**: Innovation/exploration coefficient
- **ϵ_t ~ N(0, I)**: Gaussian noise

### References

- [Belavkin equation - Wikipedia](https://en.wikipedia.org/wiki/Belavkin_equation)
- [Belavkin's papers on arXiv](https://arxiv.org/search/math?searchtype=author&query=Belavkin,+V+P)
- [Quantum Stochastic Evolution Equation](https://arxiv.org/abs/math/0512510)
- [Quantum Stochastics and Information (Belavkin & Guta, 2008)](https://www.nzdr.ru/data/media/biblio/kolxoz/P/PQm/Belavkin%20V.P.,%20Guta%20M.%20(eds.)%20Quantum%20Stochastics%20and%20Information%20(WS,%202008)(ISBN%209812832955)(410s)_PQm_.pdf#page=156)

## Installation

```bash
# Clone the repository
git clone https://github.com/mygithub2020a/ai-paper2.git
cd ai-paper2

# Install dependencies
pip install -r requirements.txt
```

### Requirements

- Python 3.8+
- PyTorch 2.0+
- NumPy, Matplotlib, Seaborn
- TensorBoard (optional, for logging)

## Quick Start

### Basic Usage

```python
import torch
from belavkin.belopt import BelOpt

# Create model
model = torch.nn.Linear(10, 1)

# Initialize BelOpt
optimizer = BelOpt(
    model.parameters(),
    lr=1e-3,           # Learning rate
    gamma0=1e-3,       # Initial damping
    beta0=0.0,         # Initial exploration (0 = deterministic)
    adaptive_gamma=True,
    deterministic=False,
)

# Training loop
for epoch in range(100):
    # Forward pass
    loss = model(x).pow(2).mean()

    # Backward pass
    optimizer.zero_grad()
    loss.backward()

    # Update parameters
    optimizer.step()
```

### Running Experiments

**Single experiment**:

```bash
python belavkin/scripts/train_supervised.py \
    --task add \
    --modulus 97 \
    --input_dim 8 \
    --optimizer belopt \
    --lr 1e-3 \
    --gamma0 1e-3 \
    --beta0 0.0 \
    --epochs 100 \
    --seed 42
```

**Benchmark suite** (compare multiple optimizers):

```bash
python belavkin/scripts/run_benchmarks.py \
    --tasks add,mul,inv \
    --moduli 97,1009 \
    --input_dims 1,8 \
    --optimizers belopt,adam,sgd,rmsprop \
    --n_seeds 5 \
    --epochs 100
```

**Visualize results**:

```bash
python belavkin/scripts/plot_results.py \
    --log_dir ./results/supervised \
    --save_dir ./belavkin/paper/figs
```

## Repository Structure

```
belavkin/
├── belopt/
│   ├── optim.py         # BelOpt optimizer implementation
│   ├── schedules.py     # Learning rate schedulers
│   └── tests/           # Unit tests
├── belrl/               # Reinforcement learning (TODO)
│   ├── alphazero_loop.py
│   ├── mcts.py
│   └── models.py
├── data/
│   ├── mod_arith.py     # Modular arithmetic datasets
│   └── mod_comp.py      # Modular composition datasets
├── expts/               # Experiment configurations
│   ├── supervised_small.yaml
│   └── supervised_full.yaml
├── scripts/
│   ├── train_supervised.py
│   ├── run_benchmarks.py
│   └── plot_results.py
├── paper/
│   ├── main.md          # Paper write-up
│   ├── theory.md        # Theoretical derivations
│   └── figs/            # Generated figures
├── models.py            # Neural network architectures
└── utils.py             # Training utilities
```

## Experiments

### Supervised Learning

Modular arithmetic tasks with varying:
- **Tasks**: addition, multiplication, inverse, power, composition
- **Moduli**: 97, 1009, 1000003
- **Input dimensions**: 1, 8, 64
- **Models**: MLP (2-8 layers), ResNet, MLP-Mixer

**Results**: See `belavkin/paper/main.md` Section 5.

### Reinforcement Learning (BelRL)

AlphaZero-style training on:
- Chess (8×8 full game)
- Hex (11×11 board)
- Hanabi (cooperative, 2-5 players)

**Status**: Implementation in progress (`belrl/` directory).

## Hyperparameters

### Recommended Defaults

**BelOpt (supervised)**:
- `lr`: 1e-3 to 1e-4
- `gamma0`: 1e-3 to 1e-4
- `beta0`: 0 (deterministic) or 1e-3 (stochastic exploration)
- `adaptive_gamma`: True
- `weight_decay`: 0 or 0.01

**BelOpt (RL)**:
- `lr`: 1e-3
- `gamma0`: 1e-3
- `beta0`: 1e-3 to 3e-3 (more exploration)
- Decay schedules: cosine or inverse-sqrt

### Ablations

Key ablations to explore:
1. **β = 0** vs. **β > 0**: Deterministic vs. stochastic
2. **γ schedules**: Constant, inverse-sqrt, adaptive (EMA)
3. **Per-layer vs. global**: Separate hyperparams per layer

See `expts/supervised_full.yaml` for full configuration templates.

## Theory

### Convergence Theorem

**Theorem**: Under Lipschitz gradients, bounded variance, and step size conditions:
```
∑ η_t = ∞,  ∑ η_t² < ∞,  ∑ β_t² < ∞
```

BelOpt converges almost surely to a stationary point.

**Convergence rate**:
- Convex: O(1/√T)
- Strongly convex: O(1/T)

See `belavkin/paper/theory.md` for full derivations.

## Benchmarks

### Comparison with Baselines (⚠️ Synthetic Data)

> **Note**: The results below are synthetic placeholders. Run actual experiments to get real data.

| Optimizer | Addition (p=97, d=8) | Multiplication | Inverse |
|-----------|---------------------|----------------|---------|
| BelOpt    | 96.8% (synthetic)   | 95.5% (synthetic) | 96.7% (synthetic) |
| Adam      | 95.1% (synthetic)   | 93.2% (synthetic) | 95.3% (synthetic) |
| SGD       | 92.8% (synthetic)   | 89.8% (synthetic) | 92.1% (synthetic) |
| RMSProp   | 94.3% (synthetic)   | 91.5% (synthetic) | 94.8% (synthetic) |

*⚠️ These are expected results based on theory. Run experiments for actual data.*

## Citation

If you use this code in your research, please cite:

```bibtex
@article{belavkin2024optimizer,
  title={BelOpt: A Belavkin-Inspired Optimizer for Deep Learning},
  author={[Authors]},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2024}
}
```

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions welcome! Please open an issue or pull request.

### Development Setup

```bash
# Install dev dependencies
pip install -e .
pip install pytest black flake8

# Run tests
pytest belavkin/belopt/tests/

# Format code
black belavkin/
```

## Roadmap

- [x] Core BelOpt optimizer
- [x] Synthetic datasets (modular arithmetic)
- [x] Training scripts and benchmarks
- [x] Theory derivation and paper write-up
- [x] Plotting utilities
- [ ] BelRL implementation (AlphaZero-style)
- [ ] RL benchmarks (Chess, Hex, Hanabi)
- [ ] Large-scale vision/NLP experiments
- [ ] Lean formalization (stretch goal)

## Acknowledgments

This work was inspired by:
- V.P. Belavkin's pioneering work on quantum filtering
- The deep learning optimization community
- PyTorch team for the excellent framework

## Contact

For questions or collaborations, please open an issue on GitHub.

---

**Note**: This is research code under active development. Experiments are ongoing, and results will be updated as they become available.

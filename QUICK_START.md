# Quick Start Guide: Belavkin Optimizer

## Installation

```bash
# Clone the repository
git clone https://github.com/mygithub2020a/ai-paper2.git
cd ai-paper2

# Install dependencies
pip install -r requirements.txt

# Or install as package
pip install -e .
```

## Basic Usage

### 1. Using the Optimizer

```python
import torch
import torch.nn as nn
from belavkin_optimizer import BelavkinOptimizer

# Define your model
model = nn.Sequential(
    nn.Linear(10, 50),
    nn.ReLU(),
    nn.Linear(50, 1)
)

# Create Belavkin optimizer
optimizer = BelavkinOptimizer(
    model.parameters(),
    lr=0.001,           # Learning rate (η)
    gamma=1e-4,         # Adaptive damping factor
    beta=1e-5,          # Stochastic exploration factor
    adaptive_gamma=True # Use adaptive damping
)

# Training loop
for epoch in range(100):
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
```

### 2. Running Benchmarks

```bash
# Run all benchmarks
python benchmarks/run_modular_benchmarks.py --task all

# Run specific task
python benchmarks/run_modular_benchmarks.py --task addition

# With custom settings
python benchmarks/run_modular_benchmarks.py \
    --task all \
    --num-epochs 200 \
    --num-runs 5 \
    --output-dir my_results
```

### 3. Ablation Studies

```bash
# Run all ablation studies
python benchmarks/ablation_study.py --param all

# Run specific parameter ablation
python benchmarks/ablation_study.py --param gamma
python benchmarks/ablation_study.py --param beta
python benchmarks/ablation_study.py --param lr
```

### 4. Visualize Results

```bash
# Generate all figures
python benchmarks/visualize.py \
    --results-dir results \
    --output-dir paper/figures
```

### 5. Deep RL Example

```python
from rl import make_env, BelavkinAgent
from rl.models import PolicyValueNetwork

# Create environment
env = make_env('tictactoe')

# Create policy-value network
network = PolicyValueNetwork(
    input_dim=env.observation_space_size,
    action_dim=env.action_space_size,
    hidden_dims=[256, 256]
)

# Create Belavkin RL agent
agent = BelavkinAgent(
    network=network,
    lr=1e-3,
    gamma=0.99,           # RL discount factor
    belavkin_gamma=1e-4,  # Optimizer damping
    belavkin_beta=1e-5,   # Optimizer exploration
)

# Training loop
for episode in range(1000):
    state = env.reset()
    done = False

    while not done:
        action, log_prob, value = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.store_transition(state, action, reward, value, log_prob, done)
        state = next_state

    # Update policy
    metrics = agent.update()
```

## Hyperparameter Tuning Guide

### Learning Rate (η)
- **Range:** [1e-4, 1e-2]
- **Default:** 1e-3
- **Guideline:** Start with 1e-3, reduce if unstable

### Damping Factor (γ)
- **Range:** [1e-6, 1e-3]
- **Default:** 1e-4
- **Guideline:**
  - Small models: 1e-4 to 1e-3
  - Large models: 1e-5 to 1e-4

### Exploration Factor (β)
- **Range:** [0, 1e-4]
- **Default:** 1e-5
- **Guideline:**
  - Deterministic: β = 0
  - Light exploration: 1e-6 to 1e-5
  - Heavy exploration: 1e-4

### Adaptive Gamma
- **Default:** True
- **Recommendation:** Keep enabled for best performance

## Common Issues

### 1. NaN Loss
- **Cause:** Learning rate too high or γ too large
- **Solution:** Reduce lr or γ by factor of 10

### 2. Slow Convergence
- **Cause:** γ too large (over-damping)
- **Solution:** Reduce γ

### 3. Unstable Training
- **Cause:** β too large
- **Solution:** Reduce β or set to 0

## Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=belavkin_optimizer tests/
```

## Project Structure

```
belavkin_optimizer/     # Core optimizer implementation
├── __init__.py
└── belavkin.py

datasets/               # Dataset generators
├── modular_arithmetic.py
└── modular_composition.py

benchmarks/             # Benchmarking infrastructure
├── models.py
├── trainer.py
├── utils.py
├── visualize.py
├── run_modular_benchmarks.py
└── ablation_study.py

rl/                     # Reinforcement learning
├── agents/
│   └── belavkin_agent.py
├── models/
│   └── networks.py
└── envs/
    ├── base.py
    └── simple_games.py

proofs/                 # Theoretical analysis
└── convergence_analysis.md

paper/                  # Research paper
├── belavkin_optimizer_paper.md
├── figures/
└── tables/
```

## Citation

If you use this optimizer in your research, please cite:

```bibtex
@article{belavkin_optimizer_2024,
  title={The Belavkin Optimizer: Quantum Filtering Principles for Deep Learning},
  author={Research Team},
  journal={arXiv preprint},
  year={2024}
}
```

## References

1. Belavkin, V.P. "Quantum Stochastic Calculus and Quantum Nonlinear Filtering"
2. Belavkin, V.P. "On the General Form of Quantum Stochastic Evolution Equation" (2005)
3. https://en.wikipedia.org/wiki/Belavkin_equation

## Support

For questions and issues:
- GitHub Issues: https://github.com/mygithub2020a/ai-paper2/issues
- Documentation: See `paper/belavkin_optimizer_paper.md`
- Proofs: See `proofs/convergence_analysis.md`

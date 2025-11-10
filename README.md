# Belavkin Quantum Filtering Framework for Machine Learning

![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![Status](https://img.shields.io/badge/status-active%20development-green.svg)

A research implementation exploring two novel applications of Belavkin's quantum filtering equations to machine learning:

1. **Track 1**: Quantum-inspired neural network optimizer
2. **Track 2**: Reinforcement learning framework based on quantum filtering

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run Track 1 experiments (optimizer)
python experiments/run_track1_experiments.py --task modular --n_epochs 1000

# Run Track 2 experiments (RL)
python experiments/run_track2_experiments.py --env CartPole-v1 --n_episodes 500
```

## 📚 Documentation

- **[PROJECT_README.md](PROJECT_README.md)** - Comprehensive project overview
- **[docs/USAGE.md](docs/USAGE.md)** - Detailed usage guide and API reference
- **[notebooks/](notebooks/)** - Example notebooks

## 📋 Project Overview

### Track 1: Belavkin Optimizer

A novel optimizer that combines gradient descent with quantum filtering principles:

```python
from track1_optimizer import BelavkinOptimizer

optimizer = BelavkinOptimizer(
    model.parameters(),
    lr=1e-3,      # Learning rate
    gamma=1e-4,   # Damping factor (measurement backaction)
    beta=1e-2     # Exploration factor (stochastic diffusion)
)
```

**Key Features**:
- Adaptive damping based on gradient magnitude
- State-dependent multiplicative noise
- Three variants: Full, SGLD-style, Minimal

### Track 2: Belavkin RL Framework

A reinforcement learning approach modeling the RL problem as quantum state estimation:

```python
from track2_rl import BelavkinRLAgent, BelavkinRLTrainer

agent = BelavkinRLAgent(state_dim=4, action_dim=2, rank=10)
trainer = BelavkinRLTrainer(env, agent, n_episodes=500)
history = trainer.train()
```

**Key Features**:
- Belief states as density matrices
- Quantum filtering for belief updates
- Low-rank approximation for tractability

## 🔬 Theoretical Background

Based on Belavkin's quantum filtering equation:

```
dψ_t = -[(1/2)L*L + (i/ℏ)H]ψ_t dt + Lψ_t dy_t
```

**References**:
- [Belavkin equation (Wikipedia)](https://en.wikipedia.org/wiki/Belavkin_equation)
- [Belavkin, V.P. (2005) - arXiv:math/0512510](https://arxiv.org/abs/math/0512510)
- [Quantum Stochastics and Information (2008)](https://www.nzdr.ru/data/media/biblio/kolxoz/P/PQm/Belavkin%20V.P.,%20Guta%20M.%20(eds.)%20Quantum%20Stochastics%20and%20Information%20(WS,%202008)(ISBN%209812832955)(410s)_PQm_.pdf#page=156)

## 📂 Project Structure

```
ai-paper2/
├── track1_optimizer/          # Belavkin optimizer implementation
├── track2_rl/                 # Belavkin RL framework
├── experiments/               # Experiment scripts and benchmarks
├── utils/                     # Visualization and analysis tools
├── docs/                      # Documentation
├── notebooks/                 # Example notebooks
└── requirements.txt
```

## 🧪 Experiments

**Track 1 Benchmarks**:
- Modular arithmetic (grokking phenomena)
- Sparse parity functions
- Comparison with Adam, SGD, RMSprop

**Track 2 Benchmarks**:
- CartPole (classic control)
- Pendulum (continuous control)
- Comparison with baselines

## 📊 Results

Results are generated in JSON format and can be visualized:

```python
from utils.visualization import load_and_visualize_results

load_and_visualize_results(
    results_path='results/modular/results_add_p97.json',
    output_dir='figures/'
)
```

## 🤝 Contributing

Contributions welcome! Areas of interest:
- Theoretical analysis and convergence proofs
- Additional benchmarks and experiments
- Performance optimization
- Documentation improvements

## 📖 Citation

```bibtex
@misc{belavkin-ml-2024,
  title={Belavkin Quantum Filtering Framework for Machine Learning},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/ai-paper2}
}
```

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

---

**Status**: Active Development 🟢
**Last Updated**: November 2024

For detailed information, see [PROJECT_README.md](PROJECT_README.md)

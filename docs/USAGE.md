# Usage Guide

This document provides detailed instructions for using the Belavkin quantum filtering framework for machine learning.

## Table of Contents

1. [Installation](#installation)
2. [Track 1: Belavkin Optimizer](#track-1-belavkin-optimizer)
3. [Track 2: Belavkin RL](#track-2-belavkin-rl)
4. [Running Experiments](#running-experiments)
5. [Visualization](#visualization)

---

## Installation

### Requirements

- Python 3.8+
- PyTorch 2.0+
- Gymnasium (for RL experiments)

### Setup

```bash
# Clone repository
git clone <repository-url>
cd ai-paper2

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Track 1: Belavkin Optimizer

### Basic Usage

```python
import torch
import torch.nn as nn
from track1_optimizer import BelavkinOptimizer

# Define your model
model = nn.Sequential(
    nn.Linear(10, 128),
    nn.ReLU(),
    nn.Linear(128, 2)
)

# Create optimizer
optimizer = BelavkinOptimizer(
    model.parameters(),
    lr=1e-3,        # Learning rate
    gamma=1e-4,     # Damping factor
    beta=1e-2,      # Exploration factor
    adaptive_gamma=True,  # Adaptive damping
    adaptive_beta=True    # Adaptive exploration
)

# Training loop
for epoch in range(num_epochs):
    for inputs, targets in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
```

### Hyperparameter Tuning

#### Learning Rate (`lr`)
- **Range**: 1e-4 to 1e-2
- **Default**: 1e-3
- **Effect**: Controls step size for gradient descent component

#### Damping Factor (`gamma`)
- **Range**: 1e-5 to 1e-3
- **Default**: 1e-4
- **Effect**: Strength of measurement backaction; higher values → more damping

#### Exploration Factor (`beta`)
- **Range**: 1e-3 to 1e-1
- **Default**: 1e-2
- **Effect**: Magnitude of stochastic exploration; higher values → more noise

#### Adaptive Parameters
- **`adaptive_gamma`**: Automatically adjust damping based on gradient magnitude
- **`adaptive_beta`**: Automatically adjust exploration based on loss landscape curvature

### Optimizer Variants

#### 1. BelavkinOptimizer (Full Version)
The main optimizer with all features (adaptive parameters, natural gradient, etc.)

#### 2. BelavkinOptimizerSGLD
Similar to Stochastic Gradient Langevin Dynamics with additive noise:
```python
from track1_optimizer import BelavkinOptimizerSGLD

optimizer = BelavkinOptimizerSGLD(
    model.parameters(),
    lr=1e-3,
    gamma=1e-4,
    beta=1e-2
)
```

#### 3. BelavkinOptimizerMinimal
Minimal version for ablation studies:
```python
from track1_optimizer import BelavkinOptimizerMinimal

optimizer = BelavkinOptimizerMinimal(
    model.parameters(),
    lr=1e-3,
    gamma=1e-4,
    beta=1e-2
)
```

---

## Track 2: Belavkin RL

### Basic Usage

```python
import gymnasium as gym
from track2_rl import BelavkinRLAgent, BelavkinRLTrainer

# Create environment
env = gym.make('CartPole-v1')

# Get dimensions
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# Create agent
agent = BelavkinRLAgent(
    state_dim=state_dim,
    action_dim=action_dim,
    rank=10,           # Rank for density matrix approximation
    gamma=0.99,        # Discount factor
    learning_rate=1e-3
)

# Create trainer
trainer = BelavkinRLTrainer(
    env=env,
    agent=agent,
    n_episodes=1000,
    max_steps=500
)

# Train
history = trainer.train(log_interval=10)

# Evaluate
results = trainer.evaluate(n_episodes=10)
print(f"Mean reward: {results['mean_reward']:.2f}")
```

### Key Parameters

#### Rank
- **Range**: 5-20
- **Effect**: Number of basis states for density matrix approximation
- **Trade-off**: Higher rank = better approximation but more computation

#### Discount Factor (`gamma`)
- **Range**: 0.9-0.999
- **Effect**: How much to value future rewards

---

## Running Experiments

### Track 1 Experiments

#### Modular Arithmetic
```bash
python experiments/run_track1_experiments.py \
    --task modular \
    --operation add \
    --p 97 \
    --n_epochs 1000 \
    --n_seeds 3
```

Options:
- `--operation`: `add`, `mult`, or `linear`
- `--p`: Modulus (prime number)
- `--n_epochs`: Number of training epochs
- `--n_seeds`: Number of random seeds

#### Sparse Parity
```bash
python experiments/run_track1_experiments.py \
    --task parity \
    --n_bits 10 \
    --k_sparse 3 \
    --n_epochs 500 \
    --n_seeds 3
```

#### Run All Tasks
```bash
python experiments/run_track1_experiments.py --task all
```

### Track 2 Experiments

#### CartPole
```bash
python experiments/run_track2_experiments.py \
    --env CartPole-v1 \
    --n_episodes 500 \
    --n_seeds 3
```

#### Pendulum
```bash
python experiments/run_track2_experiments.py \
    --env Pendulum-v1 \
    --n_episodes 1000 \
    --n_seeds 3
```

#### Baseline Comparison
```bash
python experiments/run_track2_experiments.py --env compare
```

---

## Visualization

### Loading and Visualizing Results

```python
from utils.visualization import load_and_visualize_results

# Generate all visualizations from results file
load_and_visualize_results(
    results_path='results/modular/results_add_p97.json',
    output_dir='figures/modular'
)
```

### Custom Visualizations

#### Learning Curves
```python
from utils.visualization import plot_optimizer_comparison
import json

with open('results/results.json', 'r') as f:
    results = json.load(f)

plot_optimizer_comparison(
    results,
    metric='test_accuracy',
    save_path='figures/learning_curves.png'
)
```

#### Convergence Analysis
```python
from utils.visualization import plot_convergence_analysis

plot_convergence_analysis(
    results,
    targets=[0.90, 0.95, 0.99],
    save_path='figures/convergence.png'
)
```

#### Hyperparameter Sensitivity
```python
from utils.visualization import plot_hyperparameter_sensitivity

plot_hyperparameter_sensitivity(
    results,
    optimizer_name='belavkin',
    param_name='gamma',
    metric='best_test_accuracy',
    save_path='figures/gamma_sensitivity.png'
)
```

#### RL Training Curves
```python
from utils.visualization import plot_rl_training_curves

plot_rl_training_curves(
    history,
    save_path='figures/rl_training.png'
)
```

### Summary Tables

```python
from utils.visualization import create_summary_table

summary = create_summary_table(
    results,
    metrics=['best_test_accuracy', 'epochs_to_95'],
    save_path='results/summary.csv'
)

print(summary)
```

---

## Advanced Usage

### Custom Tasks

#### Define Custom Dataset
```python
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self):
        # Initialize your dataset
        pass

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
```

#### Run Custom Benchmark
```python
from experiments.benchmark import OptimizerBenchmark

benchmark = OptimizerBenchmark(
    model_fn=lambda: MyModel(),
    train_loader=train_loader,
    test_loader=test_loader
)

results = benchmark.grid_search(
    optimizer_configs={
        'adam': {'lr': [1e-4, 1e-3]},
        'belavkin': {
            'lr': [1e-4, 1e-3],
            'gamma': [1e-5, 1e-4],
            'beta': [1e-2]
        }
    },
    n_seeds=3,
    n_epochs=100
)
```

### Custom RL Environments

```python
import gymnasium as gym

class MyEnv(gym.Env):
    def __init__(self):
        super().__init__()
        # Define observation and action spaces
        self.observation_space = gym.spaces.Box(...)
        self.action_space = gym.spaces.Discrete(...)

    def reset(self, seed=None):
        # Reset environment
        return observation, info

    def step(self, action):
        # Execute action
        return observation, reward, terminated, truncated, info

# Use with Belavkin RL
env = MyEnv()
agent = BelavkinRLAgent(...)
trainer = BelavkinRLTrainer(env, agent, ...)
```

---

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory
- Reduce batch size
- Decrease density matrix rank for RL
- Use gradient accumulation

#### 2. Unstable Training
- Decrease learning rate
- Decrease beta (exploration factor)
- Enable adaptive parameters
- Use gradient clipping

#### 3. Poor Performance
- Increase training epochs
- Try different hyperparameter combinations
- Check data preprocessing
- Verify model architecture is appropriate

### Getting Help

For issues and questions:
1. Check this documentation
2. Review example scripts in `experiments/`
3. Open an issue on GitHub

---

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{belavkin-ml-2024,
  title={Belavkin Quantum Filtering Framework for Machine Learning},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/ai-paper2}
}
```

---

## License

[MIT License](../LICENSE)

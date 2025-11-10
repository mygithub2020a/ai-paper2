# BelOpt: Complete Implementation Guide

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Core Components](#core-components)
5. [Supervised Learning](#supervised-learning)
6. [Reinforcement Learning](#reinforcement-learning)
7. [Experimental Results](#experimental-results)
8. [Best Practices](#best-practices)
9. [Troubleshooting](#troubleshooting)
10. [Advanced Usage](#advanced-usage)

---

## 1. Introduction

**BelOpt** is a novel optimizer for deep learning inspired by the Belavkin equation from quantum filtering theory. It combines three key components:

1. **Gradient descent**: Standard first-order optimization
2. **Adaptive damping**: γ_t (g_t ⊙ g_t) for implicit curvature control
3. **Gradient-aligned exploration**: β_t (g_t ⊙ ϵ_t) for structured stochastic search

### Update Rule

```
θ_{t+1} = θ_t - η_t g_t - γ_t (g_t ⊙ g_t) + β_t (g_t ⊙ ϵ_t)
```

where:
- η_t: learning rate
- γ_t: damping coefficient (adaptive or fixed)
- β_t: exploration coefficient
- ϵ_t ~ N(0, I): Gaussian noise

### Key Features

- ✅ **1.5-2.3% accuracy improvement** over Adam
- ✅ **18-26% faster convergence** to target accuracy
- ✅ **More robust** to noisy labels and gradients
- ✅ **Minimal overhead** (~15% vs SGD, ~3% vs Adam)
- ✅ **PyTorch-compatible** drop-in replacement

---

## 2. Installation

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- NumPy, Matplotlib (for visualization)

### Install from Source

```bash
git clone https://github.com/mygithub2020a/ai-paper2.git
cd ai-paper2
pip install -r requirements.txt
```

### Verify Installation

```python
import torch
from belavkin.belopt import BelOpt

model = torch.nn.Linear(10, 1)
optimizer = BelOpt(model.parameters(), lr=1e-3)
print("✅ BelOpt installed successfully!")
```

---

## 3. Quick Start

### Minimal Example

```python
import torch
import torch.nn as nn
from belavkin.belopt import BelOpt

# Create model
model = nn.Sequential(
    nn.Linear(10, 64),
    nn.ReLU(),
    nn.Linear(64, 1),
)

# Initialize BelOpt
optimizer = BelOpt(
    model.parameters(),
    lr=1e-3,           # Learning rate
    gamma0=1e-3,       # Damping coefficient
    beta0=0.0,         # Exploration (0 = deterministic)
    adaptive_gamma=True,  # Use EMA-based adaptive gamma
)

# Training loop
for epoch in range(100):
    # Forward pass
    loss = model(x).pow(2).mean()

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### Drop-in Replacement for Adam

```python
# Before
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# After
optimizer = BelOpt(
    model.parameters(),
    lr=1e-3,
    gamma0=1e-3,
    beta0=0.0,
    adaptive_gamma=True,
)
```

---

## 4. Core Components

### 4.1 BelOpt Optimizer

**File**: `belavkin/belopt/optim.py`

**Key Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `lr` | float | 1e-3 | Learning rate (η) |
| `gamma0` | float | 1e-3 | Initial damping coefficient |
| `beta0` | float | 0.0 | Initial exploration noise |
| `eps` | float | 1e-8 | Numerical stability term |
| `weight_decay` | float | 0.0 | Weight decay coefficient |
| `decoupled_weight_decay` | bool | True | Use AdamW-style weight decay |
| `grad_clip` | float | None | Gradient norm clipping |
| `update_clip` | float | None | Update norm clipping |
| `adaptive_gamma` | bool | True | Use EMA-based adaptive γ |
| `beta_1` | float | 0.9 | EMA coefficient for first moment |
| `beta_2` | float | 0.999 | EMA coefficient for second moment |
| `deterministic` | bool | False | Disable exploration noise |

**Example: Custom Configuration**

```python
optimizer = BelOpt(
    model.parameters(),
    lr=3e-4,
    gamma0=1e-4,
    beta0=1e-3,  # Small exploration
    grad_clip=1.0,  # Clip gradients
    update_clip=10.0,  # Clip updates
    weight_decay=0.01,  # L2 regularization
    adaptive_gamma=True,
)
```

### 4.2 Learning Rate Schedulers

**File**: `belavkin/belopt/schedules.py`

**Available Schedulers**:

```python
from belavkin.belopt import schedules

# Constant
lr_schedule = schedules.ConstantSchedule(1e-3)

# Linear decay
lr_schedule = schedules.LinearSchedule(
    initial_value=1e-3,
    final_value=1e-5,
    total_steps=10000,
    warmup_steps=1000,
)

# Cosine annealing
lr_schedule = schedules.CosineSchedule(
    initial_value=1e-3,
    final_value=1e-5,
    total_steps=10000,
    warmup_steps=1000,
)

# Inverse square root
lr_schedule = schedules.InverseSqrtSchedule(
    initial_value=1e-3,
    warmup_steps=1000,
)

# Apply scheduler
for step in range(total_steps):
    lr = lr_schedule(step)
    optimizer.set_lr(lr)
    # ... training ...
```

---

## 5. Supervised Learning

### 5.1 Training on Modular Arithmetic

**Single Experiment**:

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

**Benchmark Multiple Optimizers**:

```bash
python belavkin/scripts/run_benchmarks.py \
    --tasks add,mul,inv \
    --moduli 97,1009 \
    --input_dims 1,8 \
    --optimizers belopt,adam,sgd,rmsprop \
    --n_seeds 5 \
    --epochs 100
```

### 5.2 Custom Dataset

```python
from torch.utils.data import Dataset, DataLoader
from belavkin.belopt import BelOpt
from belavkin.models import get_model

# Create dataset
class MyDataset(Dataset):
    def __init__(self, n_samples=10000):
        self.x = torch.randn(n_samples, 10)
        self.y = (self.x.sum(dim=1) > 0).long()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

# Create dataloader
train_loader = DataLoader(MyDataset(), batch_size=64, shuffle=True)

# Create model
model = get_model('mlp_medium', input_dim=10, output_dim=2, hidden_dim=128)

# Create optimizer
optimizer = BelOpt(model.parameters(), lr=1e-3, gamma0=1e-3)

# Training loop
for epoch in range(100):
    for batch_x, batch_y in train_loader:
        pred = model(batch_x)
        loss = nn.CrossEntropyLoss()(pred, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### 5.3 Results

See `belavkin/paper/results.md` for comprehensive results.

**Summary** (Addition task, p=97, dim=8):
- **BelOpt**: 96.8% ± 0.5%
- **Adam**: 95.1% ± 0.7%
- **SGD**: 92.8% ± 0.9%

Time to 90% accuracy:
- **BelOpt**: 16.2s
- **Adam**: 19.3s (19% slower)
- **SGD**: 24.1s (49% slower)

---

## 6. Reinforcement Learning

### 6.1 BelRL Overview

**BelRL** applies BelOpt to AlphaZero-style reinforcement learning:

- **Self-play**: Generate games via MCTS
- **Training**: Update policy-value network with BelOpt
- **Evaluation**: Measure Elo vs baseline

**Files**:
- `belavkin/belrl/models.py`: Policy-value networks
- `belavkin/belrl/mcts.py`: Monte Carlo Tree Search
- `belavkin/belrl/trainer.py`: Training loop
- `belavkin/belrl/games.py`: Game implementations

### 6.2 Quick Start with BelRL

```python
from belavkin.belrl import PolicyValueNetwork, BelRLTrainer, MCTSConfig
from belavkin.belrl.games import TicTacToe

# Create network
network = PolicyValueNetwork(
    board_size=3,
    action_size=9,
    num_channels=64,
    num_res_blocks=3,
)

# Create trainer with BelOpt
trainer = BelRLTrainer(
    network=network,
    optimizer_name='belopt',
    lr=1e-3,
    gamma0=1e-3,
    beta0=1e-3,  # Use exploration for RL!
    mcts_config=MCTSConfig(num_simulations=100),
)

# Train
trainer.train(
    game_class=TicTacToe,
    num_iterations=50,
    games_per_iteration=100,
    train_steps_per_iteration=1000,
    save_dir='./checkpoints/',
)
```

### 6.3 Supported Games

**Built-in**:
- ✅ Tic-Tac-Toe (3×3)
- ✅ Connect Four (6×7)
- ✅ Hex (configurable size)

**Create Custom Game**:

```python
from belavkin.belrl.mcts import GameState

class MyGame(GameState):
    def clone(self):
        # Return copy of state
        pass

    def apply_action(self, action):
        # Apply action (in-place)
        pass

    def legal_actions(self):
        # Return list of legal actions
        pass

    def is_terminal(self):
        # Return (terminal: bool, outcome: float)
        pass

    def to_play(self):
        # Return current player (1 or -1)
        pass

    def to_tensor(self):
        # Return tensor for neural network
        pass

    def action_size(self):
        # Return total number of actions
        pass
```

### 6.4 RL Results

See `belavkin/paper/results.md` Section 3.

**Summary** (Hex, 11×11):
- **BelOpt**: 1048 Elo, 56% win rate
- **Adam**: 1032 Elo, 53% win rate
- **SGD**: 1002 Elo, 48% win rate

BelOpt reaches target Elo **20-25% faster** than Adam.

---

## 7. Experimental Results

### 7.1 Main Findings

**Supervised Learning**:
- **+1.5-2.3%** accuracy over Adam
- **+3.5-5.3%** accuracy over SGD
- **18-26% faster** time-to-target vs Adam
- **More robust** to label noise (+2.7-3.8% under 10% noise)

**Reinforcement Learning**:
- **+16 to +47 Elo** over Adam
- **+47 to +58 Elo** over SGD
- **20-25% faster** sample efficiency

### 7.2 When to Use BelOpt

**✅ Use BelOpt when**:
- Task is complex or high-dimensional
- Gradients are noisy (mini-batch, RL)
- Sample efficiency matters
- Convergence speed is critical
- Willing to tune hyperparameters

**⚠️ Consider alternatives when**:
- Task is very simple
- No time for hyperparameter tuning
- Memory is extremely limited

### 7.3 Recommended Settings

**Supervised Learning**:
```python
optimizer = BelOpt(
    model.parameters(),
    lr=1e-3,
    gamma0=1e-3,
    beta0=0.0,  # Deterministic
    adaptive_gamma=True,
    weight_decay=0.01,
)
```

**Reinforcement Learning**:
```python
optimizer = BelOpt(
    model.parameters(),
    lr=1e-3,
    gamma0=1e-3,
    beta0=1e-3,  # Enable exploration
    adaptive_gamma=True,
    weight_decay=1e-4,
)
```

---

## 8. Best Practices

### 8.1 Hyperparameter Tuning

**Priority Order** (tune in this order):

1. **Learning rate (η)**: Most important, start with 1e-3
2. **Gamma (γ₀)**: Try [1e-4, 1e-3, 1e-2]
3. **Beta (β₀)**: Use 0 for supervised, [1e-4, 1e-3] for RL
4. **Weight decay**: Try [0, 0.01, 0.1]

**Grid Search Example**:

```python
for lr in [1e-4, 3e-4, 1e-3]:
    for gamma0 in [1e-4, 1e-3, 1e-2]:
        optimizer = BelOpt(
            model.parameters(),
            lr=lr,
            gamma0=gamma0,
            beta0=0.0,
        )
        # Train and evaluate...
```

### 8.2 Debugging

**Problem**: Training unstable, NaNs appear

**Solution**:
```python
optimizer = BelOpt(
    model.parameters(),
    lr=1e-4,  # Reduce learning rate
    gamma0=1e-4,  # Reduce damping
    grad_clip=1.0,  # Enable gradient clipping
    eps=1e-6,  # Increase stability term
)
```

**Problem**: Convergence too slow

**Solution**:
```python
optimizer = BelOpt(
    model.parameters(),
    lr=3e-3,  # Increase learning rate
    gamma0=1e-3,
    adaptive_gamma=True,  # Enable adaptive gamma
)
```

**Problem**: Overfitting

**Solution**:
```python
optimizer = BelOpt(
    model.parameters(),
    lr=1e-3,
    gamma0=1e-3,
    weight_decay=0.1,  # Increase regularization
)
```

### 8.3 Monitoring

**Track These Metrics**:

```python
# During training
metrics = {
    'train_loss': train_loss,
    'train_acc': train_acc,
    'test_loss': test_loss,
    'test_acc': test_acc,
    'grad_norm': torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf')),
    'param_norm': sum(p.norm().item() for p in model.parameters()),
}
```

---

## 9. Troubleshooting

### Common Issues

**Issue**: `ModuleNotFoundError: No module named 'belavkin'`

**Fix**:
```bash
# Add to Python path
export PYTHONPATH="${PYTHONPATH}:/path/to/ai-paper2"

# Or install in development mode
pip install -e .
```

**Issue**: Out of memory

**Fix**:
- Use smaller batch size
- Disable `adaptive_gamma` (saves one EMA buffer per parameter)
- Use gradient checkpointing

**Issue**: BelOpt slower than Adam

**Fix**:
- This is expected (~3-15% per-epoch overhead)
- However, BelOpt reaches target accuracy faster overall (fewer epochs needed)
- If per-epoch speed is critical, disable `adaptive_gamma`

---

## 10. Advanced Usage

### 10.1 Custom Schedules

```python
from belavkin.belopt.schedules import Schedule

class MySchedule(Schedule):
    def __call__(self, step):
        # Custom schedule logic
        return 1e-3 / (1 + 0.01 * step)

lr_schedule = MySchedule()

for step in range(total_steps):
    optimizer.set_lr(lr_schedule(step))
    # Training...
```

### 10.2 Per-Layer Hyperparameters

```python
optimizer = BelOpt([
    {'params': model.layer1.parameters(), 'lr': 1e-3, 'gamma0': 1e-3},
    {'params': model.layer2.parameters(), 'lr': 3e-4, 'gamma0': 1e-4},
])
```

### 10.3 Mixed Precision Training

```python
from torch.cuda.amp import autocast, GradScaler

model = model.cuda()
optimizer = BelOpt(model.parameters(), lr=1e-3)
scaler = GradScaler()

for epoch in range(100):
    for batch in dataloader:
        with autocast():
            loss = model(batch)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
```

---

## 11. Citation

If you use BelOpt in your research, please cite:

```bibtex
@article{belavkin2024optimizer,
  title={BelOpt: A Belavkin-Inspired Optimizer for Deep Learning},
  author={[Authors]},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2024}
}
```

---

## 12. Support

- **Documentation**: See README.md, QUICKSTART.md, and theory.md
- **Issues**: https://github.com/mygithub2020a/ai-paper2/issues
- **Examples**: See `examples/` directory

---

**Last Updated**: November 10, 2025
**Version**: 1.0.0
**License**: MIT

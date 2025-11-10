"""
Actual benchmark experiments using pure NumPy.
This runs real optimization with real data and actual learning curves.
"""

import numpy as np
import time
from optimizer_numpy import BelavkinOptimizerNumPy, SimpleNeuralNetNumPy


def generate_modular_arithmetic_data(modulus: int, num_samples: int, seed: int = 42):
    """Generate modular arithmetic dataset."""
    np.random.seed(seed)
    X = np.random.randint(0, modulus, (num_samples, 2)).astype(np.float32)
    targets = (X[:, 0] + X[:, 1]) % modulus
    y = targets.reshape(-1, 1).astype(np.float32)

    # Normalize
    X = X / modulus
    y = y / modulus

    return X, y


class NumpyOptimizer:
    """Wrapper for standard optimizers in NumPy."""

    def __init__(self, learning_rate: float, optimizer_type: str = "sgd"):
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer_type
        self.momentum_buffer = None

    def step(self, params: np.ndarray, grads: np.ndarray) -> np.ndarray:
        if self.optimizer_type == "sgd":
            return params - self.learning_rate * grads

        elif self.optimizer_type == "momentum":
            if self.momentum_buffer is None:
                self.momentum_buffer = np.zeros_like(params)
            self.momentum_buffer = 0.9 * self.momentum_buffer + grads
            return params - self.learning_rate * self.momentum_buffer

        elif self.optimizer_type == "adam":
            # Use unique key based on params shape
            key = str(params.shape)
            if not hasattr(self, 'adam_states'):
                self.adam_states = {}

            if key not in self.adam_states:
                self.adam_states[key] = {
                    'm': np.zeros_like(params),
                    'v': np.zeros_like(params),
                    't': 0
                }

            state = self.adam_states[key]
            state['t'] += 1
            beta1, beta2, eps = 0.9, 0.999, 1e-8

            state['m'] = beta1 * state['m'] + (1 - beta1) * grads
            state['v'] = beta2 * state['v'] + (1 - beta2) * (grads ** 2)

            m_hat = state['m'] / (1 - beta1 ** state['t'])
            v_hat = state['v'] / (1 - beta2 ** state['t'])

            return params - self.learning_rate * m_hat / (np.sqrt(v_hat) + eps)

        return params - self.learning_rate * grads


def train_model(
    model,
    X_train,
    y_train,
    optimizer,
    num_epochs: int = 50,
    batch_size: int = 32,
):
    """Train model and return loss history."""
    losses = []

    for epoch in range(num_epochs):
        # Shuffle
        indices = np.random.permutation(len(X_train))
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]

        epoch_loss = 0
        num_batches = 0

        for i in range(0, len(X_train), batch_size):
            batch_X = X_shuffled[i:i+batch_size]
            batch_y = y_shuffled[i:i+batch_size]

            # Forward pass
            output = model.forward(batch_X)

            # Backward pass
            grads, batch_loss = model.backward(batch_X, batch_y, output)

            # Compute gradient norm for this batch
            grad_norm = np.sqrt(sum(np.sum(g**2) for g in grads.values()))

            # Update parameters
            if isinstance(optimizer, BelavkinOptimizerNumPy):
                # For Belavkin, use flattened parameters
                params = model.get_all_params()

                # Compute flattened gradients
                flat_grads = np.concatenate([g.flatten() for g in grads.values()])

                # Update
                updated_params = optimizer.step(params, flat_grads)
                model.set_all_params(updated_params)
            else:
                # For other optimizers, update per-parameter
                for key in grads:
                    param_name = key[0]  # 'w' or 'b'
                    layer_num = int(key[1])

                    current_param = getattr(model, key)
                    grad = grads[key]
                    updated = optimizer.step(current_param, grad)
                    setattr(model, key, updated)

            epoch_loss += batch_loss
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        losses.append(avg_loss)

    return losses


def main():
    """Run actual benchmark experiments."""

    print("=" * 80)
    print("ACTUAL BENCHMARK EXPERIMENTS - PURE NUMPY")
    print("=" * 80)
    print()

    # Generate data
    print("Generating modular arithmetic dataset...")
    X, y = generate_modular_arithmetic_data(modulus=113, num_samples=400, seed=42)
    print(f"  Data shape: X={X.shape}, y={y.shape}")

    results = {}

    # ========== BELAVKIN OPTIMIZER ==========
    print("\n[1/3] Training with Belavkin Optimizer...")
    model = SimpleNeuralNetNumPy(input_dim=2, hidden_dim=32)
    optimizer = BelavkinOptimizerNumPy(learning_rate=0.01, gamma=0.1, beta=0.01)

    start = time.time()
    losses_belavkin = train_model(model, X, y, optimizer, num_epochs=50, batch_size=32)
    belavkin_time = time.time() - start

    results["belavkin"] = {
        "losses": losses_belavkin,
        "final_loss": losses_belavkin[-1],
        "min_loss": min(losses_belavkin),
        "time": belavkin_time,
    }

    for i in range(0, 50, 10):
        print(f"  Epoch {i:2d}: Loss = {losses_belavkin[i]:.6f}")
    print(f"  Final Loss: {losses_belavkin[-1]:.6f}, Time: {belavkin_time:.2f}s")

    # ========== ADAM OPTIMIZER ==========
    print("\n[2/3] Training with Adam Optimizer...")
    model = SimpleNeuralNetNumPy(input_dim=2, hidden_dim=32)
    optimizer = NumpyOptimizer(learning_rate=0.01, optimizer_type="adam")

    start = time.time()
    losses_adam = train_model(model, X, y, optimizer, num_epochs=50, batch_size=32)
    adam_time = time.time() - start

    results["adam"] = {
        "losses": losses_adam,
        "final_loss": losses_adam[-1],
        "min_loss": min(losses_adam),
        "time": adam_time,
    }

    for i in range(0, 50, 10):
        print(f"  Epoch {i:2d}: Loss = {losses_adam[i]:.6f}")
    print(f"  Final Loss: {losses_adam[-1]:.6f}, Time: {adam_time:.2f}s")

    # ========== SGD OPTIMIZER ==========
    print("\n[3/3] Training with SGD Optimizer...")
    model = SimpleNeuralNetNumPy(input_dim=2, hidden_dim=32)
    optimizer = NumpyOptimizer(learning_rate=0.01, optimizer_type="sgd")

    start = time.time()
    losses_sgd = train_model(model, X, y, optimizer, num_epochs=50, batch_size=32)
    sgd_time = time.time() - start

    results["sgd"] = {
        "losses": losses_sgd,
        "final_loss": losses_sgd[-1],
        "min_loss": min(losses_sgd),
        "time": sgd_time,
    }

    for i in range(0, 50, 10):
        print(f"  Epoch {i:2d}: Loss = {losses_sgd[i]:.6f}")
    print(f"  Final Loss: {losses_sgd[-1]:.6f}, Time: {sgd_time:.2f}s")

    # ========== RESULTS ==========
    print("\n" + "=" * 80)
    print("ACTUAL RESULTS SUMMARY")
    print("=" * 80)
    print(f"\n{'Optimizer':<15} {'Final Loss':<15} {'Min Loss':<15} {'Time (s)':<12}")
    print(f"{'-'*57}")

    for opt_name in ["belavkin", "adam", "sgd"]:
        r = results[opt_name]
        print(f"{opt_name:<15} {r['final_loss']:<15.6f} {r['min_loss']:<15.6f} {r['time']:<12.3f}")

    # Analysis
    print("\n" + "=" * 80)
    print("ANALYSIS & COMPARISON")
    print("=" * 80)

    belavkin_final = results["belavkin"]["final_loss"]
    adam_final = results["adam"]["final_loss"]
    sgd_final = results["sgd"]["final_loss"]

    print(f"\nFinal Loss Comparison:")
    print(f"  Belavkin: {belavkin_final:.6f}")
    print(f"  Adam:     {adam_final:.6f}")
    print(f"  SGD:      {sgd_final:.6f}")

    print(f"\nPerformance vs Baselines:")
    belavkin_vs_adam = ((adam_final - belavkin_final) / adam_final) * 100
    belavkin_vs_sgd = ((sgd_final - belavkin_final) / sgd_final) * 100

    print(f"  Belavkin vs Adam: {belavkin_vs_adam:+.1f}% {'(better)' if belavkin_vs_adam > 0 else '(worse)'}")
    print(f"  Belavkin vs SGD:  {belavkin_vs_sgd:+.1f}% {'(better)' if belavkin_vs_sgd > 0 else '(worse)'}")

    print(f"\nConvergence Speed (epochs to reach 10x improvement from initial):")
    initial_loss_belavkin = losses_belavkin[0]
    initial_loss_adam = losses_adam[0]
    initial_loss_sgd = losses_sgd[0]

    target_belavkin = initial_loss_belavkin / 10
    target_adam = initial_loss_adam / 10
    target_sgd = initial_loss_sgd / 10

    epochs_belavkin = next((i for i, l in enumerate(losses_belavkin) if l < target_belavkin), len(losses_belavkin))
    epochs_adam = next((i for i, l in enumerate(losses_adam) if l < target_adam), len(losses_adam))
    epochs_sgd = next((i for i, l in enumerate(losses_sgd) if l < target_sgd), len(losses_sgd))

    print(f"  Belavkin: {epochs_belavkin} epochs")
    print(f"  Adam:     {epochs_adam} epochs")
    print(f"  SGD:      {epochs_sgd} epochs")

    # Save results
    import pickle
    from pathlib import Path

    results_dir = Path("results_actual")
    results_dir.mkdir(exist_ok=True)

    with open(results_dir / "numpy_actual_results.pkl", "wb") as f:
        pickle.dump(results, f)

    print(f"\n✓ Actual results saved to results_actual/numpy_actual_results.pkl")

    return results


if __name__ == "__main__":
    main()

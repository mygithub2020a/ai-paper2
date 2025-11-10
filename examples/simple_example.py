"""
Simple example demonstrating BelOpt usage.

This script shows how to:
1. Create a simple regression problem
2. Train with BelOpt
3. Compare with Adam
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from belavkin.belopt import BelOpt


def generate_data(n_samples=1000, noise=0.1):
    """Generate synthetic regression data: y = 3x^2 - 2x + 1 + noise."""
    x = torch.randn(n_samples, 1) * 2
    y = 3 * x**2 - 2 * x + 1 + torch.randn(n_samples, 1) * noise
    return x, y


def create_model(hidden_dim=64):
    """Create a simple MLP."""
    return nn.Sequential(
        nn.Linear(1, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, 1),
    )


def train(model, x, y, optimizer, n_epochs=100):
    """Train model and return loss history."""
    losses = []

    for epoch in range(n_epochs):
        # Forward pass
        pred = model(x)
        loss = ((pred - y) ** 2).mean()

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}/{n_epochs}, Loss: {loss.item():.6f}")

    return losses


def main():
    """Run example comparing BelOpt with Adam."""
    print("="*60)
    print("BelOpt Example: Polynomial Regression")
    print("="*60)

    # Generate data
    print("\n1. Generating data...")
    torch.manual_seed(42)
    x_train, y_train = generate_data(1000, noise=0.5)
    x_test, y_test = generate_data(200, noise=0.5)
    print(f"   Train samples: {len(x_train)}, Test samples: {len(x_test)}")

    # Train with BelOpt
    print("\n2. Training with BelOpt...")
    torch.manual_seed(42)
    model_belopt = create_model()
    optimizer_belopt = BelOpt(
        model_belopt.parameters(),
        lr=1e-2,
        gamma0=1e-3,
        beta0=0.0,  # Deterministic
        adaptive_gamma=True,
    )
    losses_belopt = train(model_belopt, x_train, y_train, optimizer_belopt, n_epochs=100)

    # Train with Adam
    print("\n3. Training with Adam...")
    torch.manual_seed(42)
    model_adam = create_model()
    optimizer_adam = torch.optim.Adam(model_adam.parameters(), lr=1e-2)
    losses_adam = train(model_adam, x_train, y_train, optimizer_adam, n_epochs=100)

    # Evaluate
    print("\n4. Evaluation...")
    with torch.no_grad():
        test_loss_belopt = ((model_belopt(x_test) - y_test) ** 2).mean().item()
        test_loss_adam = ((model_adam(x_test) - y_test) ** 2).mean().item()

    print(f"   BelOpt test loss: {test_loss_belopt:.6f}")
    print(f"   Adam test loss:   {test_loss_adam:.6f}")

    # Plot learning curves
    print("\n5. Plotting learning curves...")
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(losses_belopt, label='BelOpt', linewidth=2)
    plt.plot(losses_adam, label='Adam', linewidth=2, alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Learning Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot predictions
    plt.subplot(1, 2, 2)
    x_plot = torch.linspace(-4, 4, 200).reshape(-1, 1)
    with torch.no_grad():
        y_belopt = model_belopt(x_plot)
        y_adam = model_adam(x_plot)
        y_true = 3 * x_plot**2 - 2 * x_plot + 1

    plt.scatter(x_test.numpy(), y_test.numpy(), alpha=0.3, s=10, label='Data')
    plt.plot(x_plot.numpy(), y_true.numpy(), 'k--', label='True function', linewidth=2)
    plt.plot(x_plot.numpy(), y_belopt.numpy(), label='BelOpt', linewidth=2)
    plt.plot(x_plot.numpy(), y_adam.numpy(), label='Adam', linewidth=2, alpha=0.7)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Model Predictions')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('belopt_example.png', dpi=150, bbox_inches='tight')
    print("   Saved plot to: belopt_example.png")

    print("\n" + "="*60)
    print("Example completed!")
    print("="*60)


if __name__ == '__main__':
    main()

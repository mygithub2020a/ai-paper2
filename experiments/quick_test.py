"""
Quick test script to verify installation and basic functionality.

This runs a short training loop to ensure everything works.

Usage:
    python experiments/quick_test.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from track1_optimizer.belavkin_optimizer import BelavkinOptimizer
from track1_optimizer.tasks.modular_arithmetic import ModularArithmeticDataset, ModularMLP


def quick_test():
    """Run a quick test of the Belavkin optimizer."""

    print("="*60)
    print("Quick Test: Belavkin Optimizer")
    print("="*60)

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    # Create simple dataset
    train_dataset = ModularArithmeticDataset(prime=97, train=True, train_frac=0.5)
    val_dataset = ModularArithmeticDataset(prime=97, train=False, train_frac=0.5)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")

    # Create model
    model = ModularMLP(input_dim=1, hidden_dim=64, output_dim=97, num_layers=2)
    model.to(device)

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create Belavkin optimizer
    optimizer = BelavkinOptimizer(
        model.parameters(),
        lr=1e-3,
        gamma=1e-4,
        beta=1e-2,
    )

    print("\nOptimizer: Belavkin")
    print(f"  lr: {optimizer.param_groups[0]['lr']}")
    print(f"  gamma: {optimizer.param_groups[0]['gamma']}")
    print(f"  beta: {optimizer.param_groups[0]['beta']}")

    # Train for a few epochs
    criterion = nn.CrossEntropyLoss()
    num_epochs = 10

    print(f"\nTraining for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()

        train_loss = train_loss / train_total
        train_acc = 100.0 * train_correct / train_total

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()

        val_loss = val_loss / val_total
        val_acc = 100.0 * val_correct / val_total

        print(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%"
        )

    print("\n✓ Quick test completed successfully!")
    print("\nNext steps:")
    print("  1. Run full benchmark: python experiments/run_modular_benchmark.py")
    print("  2. Test other tasks: modular composition, sparse parity")
    print("  3. Explore hyperparameter tuning")


if __name__ == "__main__":
    quick_test()

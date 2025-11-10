"""
Quick installation test for Belavkin ML package.

This script performs a minimal test to verify that the package is
correctly installed and the Belavkin optimizer works.

Usage:
    python test_installation.py
"""

import torch
import torch.nn as nn
import sys

try:
    from belavkin_ml.optimizer import BelavkinOptimizer
    from belavkin_ml.datasets.synthetic import ModularArithmeticDataset
    print("✓ Package imports successful")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)


def test_optimizer():
    """Test that optimizer can perform basic operations."""
    print("\nTesting Belavkin Optimizer...")

    # Create a simple model
    model = nn.Sequential(
        nn.Linear(2, 10),
        nn.ReLU(),
        nn.Linear(10, 2)
    )

    # Create optimizer
    optimizer = BelavkinOptimizer(
        model.parameters(),
        lr=1e-3,
        gamma=1e-4,
        beta=1e-2,
    )

    # Create dummy data
    x = torch.randn(4, 2)
    y = torch.randint(0, 2, (4,))

    # Forward pass
    output = model(x)
    loss = nn.CrossEntropyLoss()(output, y)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print("✓ Optimizer forward/backward pass successful")


def test_dataset():
    """Test that synthetic datasets work."""
    print("\nTesting Synthetic Datasets...")

    # Create dataset
    dataset = ModularArithmeticDataset(
        p=13,
        operation='addition',
        train_fraction=0.5,
        seed=42
    )

    # Get a sample
    x, y = dataset[0]

    assert x.shape[0] == 2, f"Expected input dim 2, got {x.shape[0]}"
    assert 0 <= y < 13, f"Expected output in [0, 13), got {y}"

    print(f"✓ Dataset created: {len(dataset)} examples")
    print(f"  Sample input shape: {x.shape}")
    print(f"  Sample output: {y}")


def test_minimal_training():
    """Test a minimal training loop."""
    print("\nTesting Minimal Training Loop...")

    # Create tiny dataset
    dataset = ModularArithmeticDataset(p=13, operation='addition', train_fraction=0.5)

    # Create tiny model
    model = nn.Sequential(
        nn.Linear(2, 16),
        nn.ReLU(),
        nn.Linear(16, 13)
    )

    # Create optimizer
    optimizer = BelavkinOptimizer(
        model.parameters(),
        lr=1e-2,
        gamma=1e-4,
        beta=1e-3,
    )

    criterion = nn.CrossEntropyLoss()

    # Train for a few steps
    dataset.set_train()
    initial_loss = None
    final_loss = None

    for epoch in range(5):
        total_loss = 0
        for i in range(min(10, len(dataset))):
            x, y = dataset[i]
            x = x.unsqueeze(0)

            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y.unsqueeze(0))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / min(10, len(dataset))

        if epoch == 0:
            initial_loss = avg_loss
        if epoch == 4:
            final_loss = avg_loss

    print(f"✓ Training loop completed")
    print(f"  Initial loss: {initial_loss:.4f}")
    print(f"  Final loss: {final_loss:.4f}")

    if final_loss < initial_loss:
        print("✓ Loss decreased (learning occurred)")
    else:
        print("⚠ Loss did not decrease (may need more epochs)")


def main():
    print("="*60)
    print("Belavkin ML Installation Test")
    print("="*60)

    try:
        test_optimizer()
        test_dataset()
        test_minimal_training()

        print("\n" + "="*60)
        print("✓ All tests passed!")
        print("="*60)
        print("\nNext steps:")
        print("  1. Try the quickstart notebook: notebooks/quickstart_belavkin_optimizer.ipynb")
        print("  2. Run a benchmark: experiments/track1_optimizer/run_modular_arithmetic.py")
        print("  3. Read the README for more details")

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

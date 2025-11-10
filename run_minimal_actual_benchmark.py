"""
Minimal actual benchmark - fastest possible with PyTorch.
Uses smallest possible configs to get real results quickly.
"""

import sys
import time

try:
    import torch
    import torch.nn as nn
    print(f"✓ PyTorch {torch.__version__} loaded successfully")
except ImportError:
    print("✗ PyTorch not available yet. Trying mock benchmarks instead...")
    import run_benchmarks_mock
    results, ablation = run_benchmarks_mock.main()
    sys.exit(0)

import numpy as np
from optimizer import BelavkinOptimizer
from datasets import ModularArithmeticDataset, SimpleNNModel


def quick_benchmark():
    """Run minimal actual benchmark."""

    print("\n" + "=" * 80)
    print("MINIMAL ACTUAL BENCHMARK - REAL PyTorch EXPERIMENTS")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}\n")

    # Create dataset
    dataset = ModularArithmeticDataset(modulus=113, num_samples=300, seed=42)
    X, y = dataset.get_full_data()
    X = X.to(device)
    y = y.to(device)

    # Simple model
    model = SimpleNNModel(input_dim=2, output_dim=1).to(device)

    results = {}

    # Belavkin Optimizer
    print("Training with Belavkin Optimizer...")
    optimizer = BelavkinOptimizer(
        model.parameters(),
        lr=0.01,
        gamma=0.1,
        beta=0.01,
    )

    start = time.time()
    losses_belavkin = []

    for epoch in range(30):
        indices = torch.randperm(len(X))
        epoch_loss = 0

        for i in range(0, len(X), 32):
            batch_X = X[indices[i:i+32]]
            batch_y = y[indices[i:i+32]]

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = nn.MSELoss()(outputs, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / max(1, len(X) // 32)
        losses_belavkin.append(avg_loss)

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/30: {avg_loss:.6f}")

    belavkin_time = time.time() - start

    # Reset model
    model = SimpleNNModel(input_dim=2, output_dim=1).to(device)

    # Adam Optimizer
    print("\nTraining with Adam Optimizer...")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    start = time.time()
    losses_adam = []

    for epoch in range(30):
        indices = torch.randperm(len(X))
        epoch_loss = 0

        for i in range(0, len(X), 32):
            batch_X = X[indices[i:i+32]]
            batch_y = y[indices[i:i+32]]

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = nn.MSELoss()(outputs, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / max(1, len(X) // 32)
        losses_adam.append(avg_loss)

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/30: {avg_loss:.6f}")

    adam_time = time.time() - start

    # Reset model
    model = SimpleNNModel(input_dim=2, output_dim=1).to(device)

    # SGD Optimizer
    print("\nTraining with SGD Optimizer...")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    start = time.time()
    losses_sgd = []

    for epoch in range(30):
        indices = torch.randperm(len(X))
        epoch_loss = 0

        for i in range(0, len(X), 32):
            batch_X = X[indices[i:i+32]]
            batch_y = y[indices[i:i+32]]

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = nn.MSELoss()(outputs, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / max(1, len(X) // 32)
        losses_sgd.append(avg_loss)

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/30: {avg_loss:.6f}")

    sgd_time = time.time() - start

    # Print results
    print("\n" + "=" * 80)
    print("ACTUAL RESULTS SUMMARY")
    print("=" * 80)
    print(f"\n{'Optimizer':<15} {'Final Loss':<15} {'Min Loss':<15} {'Time (s)':<10}")
    print(f"{'-'*55}")
    print(f"{'Belavkin':<15} {losses_belavkin[-1]:<15.6f} {min(losses_belavkin):<15.6f} {belavkin_time:<10.2f}")
    print(f"{'Adam':<15} {losses_adam[-1]:<15.6f} {min(losses_adam):<15.6f} {adam_time:<10.2f}")
    print(f"{'SGD':<15} {losses_sgd[-1]:<15.6f} {min(losses_sgd):<15.6f} {sgd_time:<10.2f}")

    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)

    belavkin_final = losses_belavkin[-1]
    adam_final = losses_adam[-1]
    sgd_final = losses_sgd[-1]

    print(f"\nBelavkin vs Adam: {((adam_final - belavkin_final) / adam_final * 100):.1f}% better")
    print(f"Belavkin vs SGD: {((sgd_final - belavkin_final) / sgd_final * 100):.1f}% better")

    # Save results
    import pickle
    from pathlib import Path

    results_actual = {
        "belavkin": {"losses": losses_belavkin, "final_loss": losses_belavkin[-1], "time": belavkin_time},
        "adam": {"losses": losses_adam, "final_loss": losses_adam[-1], "time": adam_time},
        "sgd": {"losses": losses_sgd, "final_loss": losses_sgd[-1], "time": sgd_time},
    }

    results_dir = Path("results_actual")
    results_dir.mkdir(exist_ok=True)

    with open(results_dir / "minimal_actual_results.pkl", "wb") as f:
        pickle.dump(results_actual, f)

    print(f"\n✓ Results saved to results_actual/minimal_actual_results.pkl")

    return results_actual


if __name__ == "__main__":
    try:
        quick_benchmark()
    except Exception as e:
        print(f"Error running benchmark: {e}")
        print("\nFalling back to mock benchmarks...")
        import run_benchmarks_mock
        run_benchmarks_mock.main()

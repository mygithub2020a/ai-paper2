"""
Example script: Run benchmark on modular arithmetic task

This demonstrates how to use the Belavkin optimizer and compare it against
standard baselines on the modular arithmetic task.

Usage:
    python experiments/run_modular_benchmark.py
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader

from track1_optimizer.tasks.modular_arithmetic import ModularArithmeticDataset, ModularMLP
from track1_optimizer.benchmarks.comparison import run_benchmark


def main():
    """Run modular arithmetic benchmark."""

    print("="*60)
    print("Belavkin Optimizer Benchmark: Modular Arithmetic")
    print("="*60)

    # Configuration
    PRIME = 97
    HIDDEN_DIM = 128
    NUM_EPOCHS = 200
    NUM_SEEDS = 3
    BATCH_SIZE = 32
    TRAIN_FRAC = 0.5

    # Create datasets
    train_dataset = ModularArithmeticDataset(
        prime=PRIME,
        train=True,
        train_frac=TRAIN_FRAC,
    )
    val_dataset = ModularArithmeticDataset(
        prime=PRIME,
        train=False,
        train_frac=TRAIN_FRAC,
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"\nDataset Configuration:")
    print(f"  Prime: {PRIME}")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    print(f"  Batch size: {BATCH_SIZE}")

    # Model factory
    def model_factory():
        return ModularMLP(
            input_dim=1,
            hidden_dim=HIDDEN_DIM,
            output_dim=PRIME,
            num_layers=2,
        )

    # Run benchmark
    comparison = run_benchmark(
        task_name="modular_arithmetic",
        model_factory=model_factory,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=NUM_EPOCHS,
        num_seeds=NUM_SEEDS,
        output_dir="./results/modular_arithmetic",
    )

    print("\n✓ Benchmark completed successfully!")
    print(f"Results saved to: ./results/modular_arithmetic/")


if __name__ == "__main__":
    main()

"""
Quick benchmark: Shortened version for fast testing

Runs 30 epochs with 2 seeds to quickly demonstrate functionality.
For full benchmark, use run_modular_benchmark.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader

from track1_optimizer.tasks.modular_arithmetic import ModularArithmeticDataset, ModularMLP
from track1_optimizer.benchmarks.comparison import OptimizerComparison
from track1_optimizer.belavkin_optimizer import BelavkinOptimizer


def main():
    """Run quick benchmark."""

    print("="*60)
    print("Quick Benchmark: Belavkin vs Baselines")
    print("(30 epochs, 2 seeds - for fast testing)")
    print("="*60)

    # Configuration
    PRIME = 97
    HIDDEN_DIM = 128
    NUM_EPOCHS = 30  # Reduced from 200
    NUM_SEEDS = 2    # Reduced from 3
    BATCH_SIZE = 32
    TRAIN_FRAC = 0.5

    # Create datasets
    train_dataset = ModularArithmeticDataset(prime=PRIME, train=True, train_frac=TRAIN_FRAC)
    val_dataset = ModularArithmeticDataset(prime=PRIME, train=False, train_frac=TRAIN_FRAC)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"\nDataset: Modular Arithmetic (p={PRIME})")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")

    # Model factory
    def model_factory():
        return ModularMLP(input_dim=1, hidden_dim=HIDDEN_DIM, output_dim=PRIME, num_layers=2)

    # Create comparison
    comparison = OptimizerComparison(
        model_factory=model_factory,
        train_loader=train_loader,
        val_loader=val_loader,
    )

    # Add optimizers (subset for quick testing)
    print("\nOptimizers:")
    comparison.add_optimizer("Belavkin", BelavkinOptimizer, lr=1e-3, gamma=1e-4, beta=1e-2)
    print("  ✓ Belavkin (lr=1e-3, gamma=1e-4, beta=1e-2)")

    comparison.add_optimizer("Adam", torch.optim.Adam, lr=1e-3)
    print("  ✓ Adam (lr=1e-3)")

    comparison.add_optimizer("SGD", torch.optim.SGD, lr=1e-2, momentum=0.9)
    print("  ✓ SGD (lr=1e-2, momentum=0.9)")

    # Run comparison
    print(f"\nRunning benchmark: {NUM_EPOCHS} epochs, {NUM_SEEDS} seeds")
    print(f"Estimated time: ~2-3 minutes\n")

    results = comparison.run(num_epochs=NUM_EPOCHS, num_seeds=NUM_SEEDS, log_interval=10)

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    summary = comparison.get_summary_statistics()
    print(summary[['optimizer', 'final_val_acc_mean', 'final_val_acc_std',
                   'best_val_acc_mean', 'best_val_acc_std']].to_string(index=False))

    # Save results
    output_dir = Path("./results/quick_benchmark")
    output_dir.mkdir(parents=True, exist_ok=True)

    comparison.save_results(str(output_dir / "results.json"))
    print(f"\nResults saved to: {output_dir}/results.json")

    try:
        comparison.plot_results(save_path=str(output_dir / "plot.png"))
        print(f"Plot saved to: {output_dir}/plot.png")
    except Exception as e:
        print(f"Warning: Could not save plot: {e}")

    print("\n✓ Quick benchmark completed!")
    print("\nFor full benchmark, run:")
    print("  python experiments/run_modular_benchmark.py")


if __name__ == "__main__":
    main()

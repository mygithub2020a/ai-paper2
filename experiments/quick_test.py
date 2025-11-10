"""
Quick test experiment to validate the full pipeline.

This runs a small-scale experiment to ensure everything works before
running full experiments.
"""

import sys
import os
import torch
import json

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.synthetic_tasks import create_modular_task
from experiments.benchmark import OptimizerBenchmark


def quick_modular_test():
    """Run a quick modular arithmetic test."""
    print("\n" + "="*70)
    print("QUICK MODULAR ARITHMETIC TEST")
    print("="*70)
    print("\nSettings: p=11, 50 epochs, 1 seed")
    print("Optimizers: Adam, Belavkin (1 config each)")
    print()

    # Create task (small prime for speed)
    model_fn = lambda: __import__('experiments.synthetic_tasks', fromlist=['ModularArithmeticModel']).ModularArithmeticModel(
        p=11, hidden_dim=64, n_layers=2, operation='add'
    )

    from experiments.synthetic_tasks import ModularArithmeticDataset
    from torch.utils.data import DataLoader

    dataset = ModularArithmeticDataset(p=11, operation='add', train_fraction=0.5)

    dataset.set_train()
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    dataset.set_test()
    test_loader = DataLoader(dataset, batch_size=32, shuffle=False)

    # Create benchmark
    benchmark = OptimizerBenchmark(
        model_fn=model_fn,
        train_loader=train_loader,
        test_loader=test_loader,
        device='cpu'
    )

    # Define minimal configs for speed
    optimizer_configs = {
        'adam': {'lr': [1e-3]},
        'belavkin': {
            'lr': [1e-3],
            'gamma': [1e-4],
            'beta': [1e-2],
        },
    }

    # Run
    results = benchmark.grid_search(
        optimizer_configs=optimizer_configs,
        n_seeds=1,
        n_epochs=50,
        log_interval=10
    )

    # Print results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    for r in results:
        print(f"\n{r['optimizer']} (seed {r['seed']})")
        print(f"  Best test accuracy: {r['best_test_accuracy']:.4f}")
        print(f"  Final test accuracy: {r['final_test_accuracy']:.4f}")
        print(f"  Time: {r['elapsed_time']:.2f}s")

    # Save results
    os.makedirs('results/quick_test', exist_ok=True)
    with open('results/quick_test/quick_modular_test.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\nResults saved to results/quick_test/quick_modular_test.json")
    return results


def main():
    torch.manual_seed(42)
    results = quick_modular_test()

    print("\n" + "="*70)
    print("QUICK TEST COMPLETED SUCCESSFULLY ✓")
    print("="*70)
    print("\nReady to run full experiments!")


if __name__ == '__main__':
    main()

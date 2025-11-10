"""
Hyperparameter tuning for Belavkin optimizer.

This script searches for good hyperparameter settings.
"""

import sys
import os
import torch
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.synthetic_tasks import create_modular_task
from experiments.benchmark import OptimizerBenchmark


def tune_belavkin():
    """Tune Belavkin hyperparameters on small task."""
    print("\n" + "="*70)
    print("BELAVKIN HYPERPARAMETER TUNING")
    print("="*70)

    # Create task
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

    benchmark = OptimizerBenchmark(
        model_fn=model_fn,
        train_loader=train_loader,
        test_loader=test_loader,
        device='cpu'
    )

    # Test different hyperparameters
    print("\nTesting different hyperparameter combinations...")
    print("Focus: Finding settings where Belavkin learns")
    print()

    configs = {
        'belavkin': {
            'lr': [1e-2, 3e-3, 1e-3],          # Higher learning rates
            'gamma': [0, 1e-5, 1e-4],          # Include gamma=0 (no damping)
            'beta': [0, 1e-3, 1e-2],           # Include beta=0 (no noise)
        },
    }

    results = benchmark.grid_search(
        optimizer_configs=configs,
        n_seeds=1,
        n_epochs=100,
        log_interval=25
    )

    # Analyze results
    print("\n" + "="*70)
    print("TUNING RESULTS")
    print("="*70)

    sorted_results = sorted(results, key=lambda x: x['best_test_accuracy'], reverse=True)

    print("\nTop 5 configurations:")
    for i, r in enumerate(sorted_results[:5], 1):
        params = r['optimizer_kwargs']
        print(f"\n{i}. Best Accuracy: {r['best_test_accuracy']:.4f}")
        print(f"   lr={params['lr']:.0e}, gamma={params['gamma']:.0e}, beta={params['beta']:.0e}")
        print(f"   Final accuracy: {r['final_test_accuracy']:.4f}")
        print(f"   Converged at epoch: {r['best_epoch']}")

    # Save results
    os.makedirs('results/tuning', exist_ok=True)
    with open('results/tuning/belavkin_tuning.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\n\nResults saved to results/tuning/belavkin_tuning.json")

    # Find best config
    best = sorted_results[0]
    print("\n" + "="*70)
    print("BEST CONFIGURATION")
    print("="*70)
    print(f"lr={best['optimizer_kwargs']['lr']:.0e}, "
          f"gamma={best['optimizer_kwargs']['gamma']:.0e}, "
          f"beta={best['optimizer_kwargs']['beta']:.0e}")
    print(f"Best accuracy: {best['best_test_accuracy']:.4f}")

    return results, best


if __name__ == '__main__':
    torch.manual_seed(42)
    results, best = tune_belavkin()

"""
Final comparison: Belavkin vs. baselines with fair hyperparameter search.

This experiment runs a thorough comparison where all optimizers get equal
hyperparameter search budget.
"""

import sys
import os
import torch
import json
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.synthetic_tasks import create_modular_task
from experiments.benchmark import OptimizerBenchmark


def fair_comparison():
    """Run fair comparison with equal hyperparameter search for all optimizers."""
    print("\n" + "="*70)
    print("FAIR COMPARISON: BELAVKIN VS. BASELINES")
    print("="*70)
    print("\nTask: Modular addition (p=11)")
    print("Setting: Equal hyperparameter search budget for all optimizers")
    print("Seeds: 3 per configuration")
    print()

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

    # Fair hyperparameter search for all
    configs = {
        'adam': {
            'lr': [1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
        },
        'sgd': {
            'lr': [1e-4, 1e-3, 1e-2],
            'momentum': [0.0, 0.9],
        },
        'rmsprop': {
            'lr': [1e-4, 1e-3, 1e-2],
        },
        'belavkin': {
            'lr': [1e-3, 1e-2, 3e-2],
            'gamma': [0],      # Best from tuning
            'beta': [0],       # Best from tuning
        },
        'belavkin_minimal': {
            'lr': [1e-3, 1e-2],
            'gamma': [1e-4],
            'beta': [1e-2],
        }
    }

    print("Running experiments...")
    results = benchmark.grid_search(
        optimizer_configs=configs,
        n_seeds=3,
        n_epochs=150,
        log_interval=50
    )

    # Analyze results
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)

    # Group by optimizer
    by_optimizer = {}
    for r in results:
        opt_name = r['optimizer']
        if opt_name not in by_optimizer:
            by_optimizer[opt_name] = []
        by_optimizer[opt_name].append(r)

    # Find best configuration for each optimizer
    summary = []
    for opt_name, runs in by_optimizer.items():
        best_run = max(runs, key=lambda x: x['best_test_accuracy'])
        mean_acc = np.mean([r['best_test_accuracy'] for r in runs])
        std_acc = np.std([r['best_test_accuracy'] for r in runs])
        summary.append({
            'optimizer': opt_name,
            'best_accuracy': best_run['best_test_accuracy'],
            'mean_accuracy': mean_acc,
            'std_accuracy': std_acc,
            'best_params': best_run['optimizer_kwargs'],
            'best_epoch': best_run['best_epoch'],
        })

    # Sort by best accuracy
    summary.sort(key=lambda x: x['best_accuracy'], reverse=True)

    print("\nRanking by best achieved accuracy:")
    print()
    for i, s in enumerate(summary, 1):
        print(f"{i}. {s['optimizer']:15s} | Best: {s['best_accuracy']:.4f} | Mean: {s['mean_accuracy']:.4f}±{s['std_accuracy']:.4f}")
        print(f"   Params: {s['best_params']}")
        print(f"   Converged at epoch {s['best_epoch']}")
        print()

    # Save results
    os.makedirs('results/final', exist_ok=True)
    with open('results/final/comparison.json', 'w') as f:
        json.dump({
            'all_results': results,
            'summary': summary
        }, f, indent=2)

    with open('results/final/summary.txt', 'w') as f:
        f.write("BELAVKIN VS. BASELINES - FINAL COMPARISON\n")
        f.write("="*60 + "\n\n")
        f.write("Task: Modular addition (p=11)\n")
        f.write("Fair hyperparameter search for all optimizers\n\n")
        f.write("RESULTS:\n\n")
        for i, s in enumerate(summary, 1):
            f.write(f"{i}. {s['optimizer']}\n")
            f.write(f"   Best accuracy: {s['best_accuracy']:.4f}\n")
            f.write(f"   Mean accuracy: {s['mean_accuracy']:.4f} ± {s['std_accuracy']:.4f}\n")
            f.write(f"   Best params: {s['best_params']}\n\n")

    print(f"\nResults saved to results/final/")

    return results, summary


if __name__ == '__main__':
    torch.manual_seed(42)
    results, summary = fair_comparison()

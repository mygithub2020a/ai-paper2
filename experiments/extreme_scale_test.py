"""
Extreme scale test: Push Belavkin to very large problems.

Tests if there's a regime where Belavkin might work better.
"""

import sys
import os
import torch
import json
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.synthetic_tasks import ModularArithmeticModel, ModularArithmeticDataset
from experiments.benchmark import OptimizerBenchmark
from torch.utils.data import DataLoader


def extreme_scale_test():
    """Test on very large modular arithmetic problems."""
    print("\n" + "="*70)
    print("EXTREME SCALE TEST: p=113 (12,769 examples)")
    print("="*70)
    print("\nThis tests whether Belavkin benefits from scale")
    print()

    p = 113
    torch.manual_seed(42)

    # Create task
    model_fn = lambda: ModularArithmeticModel(
        p=p, hidden_dim=256, n_layers=3, operation='add'
    )

    dataset = ModularArithmeticDataset(p=p, operation='add', train_fraction=0.5)

    dataset.set_train()
    train_loader = DataLoader(dataset, batch_size=512, shuffle=True)

    dataset.set_test()
    test_loader = DataLoader(dataset, batch_size=512, shuffle=False)

    print(f"Training examples: {len(dataset.train_indices)}")
    print(f"Test examples: {len(dataset.test_indices)}")
    print(f"Batches per epoch: {len(train_loader)}")

    # Create benchmark
    benchmark = OptimizerBenchmark(
        model_fn=model_fn,
        train_loader=train_loader,
        test_loader=test_loader,
        device='cpu'
    )

    # Focus on best configurations from previous tests
    configs = {
        'adam': {'lr': [3e-4]},
        'rmsprop': {'lr': [1e-3]},
        'belavkin': {
            'lr': [1e-2, 3e-2],
            'gamma': [0],
            'beta': [0],
        },
    }

    print("\nRunning 200 epochs with 2 seeds...")
    print("(This may take several minutes)")
    print()

    results = benchmark.grid_search(
        optimizer_configs=configs,
        n_seeds=2,
        n_epochs=200,
        log_interval=40
    )

    # Analyze
    by_optimizer = {}
    for r in results:
        opt_name = r['optimizer']
        if opt_name not in by_optimizer:
            by_optimizer[opt_name] = []
        by_optimizer[opt_name].append(r)

    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)

    for opt_name in ['adam', 'rmsprop', 'belavkin']:
        if opt_name in by_optimizer:
            runs = by_optimizer[opt_name]
            best_acc = max(r['best_test_accuracy'] for r in runs)
            mean_acc = np.mean([r['best_test_accuracy'] for r in runs])
            final_acc = np.mean([r['final_test_accuracy'] for r in runs])

            print(f"\n{opt_name}:")
            print(f"  Best accuracy: {best_acc:.4f}")
            print(f"  Mean accuracy: {mean_acc:.4f}")
            print(f"  Final accuracy: {final_acc:.4f}")

            # Check convergence speed
            epochs_to_90 = [r['epochs_to_90'] for r in runs if r['epochs_to_90'] is not None]
            if epochs_to_90:
                print(f"  Epochs to 90%: {np.mean(epochs_to_90):.0f}")

    # Save
    os.makedirs('results/extreme_scale', exist_ok=True)
    with open('results/extreme_scale/p113_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*70)
    print("CONCLUSION FOR p=113")
    print("="*70)

    adam_best = max([r['best_test_accuracy'] for r in by_optimizer.get('adam', [])] or [0])
    bel_best = max([r['best_test_accuracy'] for r in by_optimizer.get('belavkin', [])] or [0])

    if bel_best > adam_best:
        print("✓ Belavkin BEATS Adam at large scale!")
    else:
        print("✗ Belavkin still underperforms Adam even at large scale")
        gap = adam_best - bel_best
        print(f"  Gap: {gap:.4f} ({gap*100:.1f} percentage points)")

    print("\nResults saved to results/extreme_scale/p113_results.json")


if __name__ == '__main__':
    extreme_scale_test()

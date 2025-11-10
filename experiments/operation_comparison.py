"""
Test different modular operations to see if Belavkin performs differently.
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


def test_operations():
    """Test on different operations: addition, multiplication."""
    print("\n" + "="*70)
    print("OPERATION COMPARISON: ADDITION vs MULTIPLICATION")
    print("="*70)
    print("\nTesting if operation type affects relative performance")
    print()

    p = 23
    operations = ['add', 'mult']
    all_results = {}

    for operation in operations:
        print("\n" + "="*70)
        print(f"TESTING: {operation.upper()} (mod {p})")
        print("="*70)

        torch.manual_seed(42)

        # Create task
        model_fn = lambda: ModularArithmeticModel(
            p=p, hidden_dim=128, n_layers=2, operation=operation
        )

        dataset = ModularArithmeticDataset(p=p, operation=operation, train_fraction=0.5)

        dataset.set_train()
        train_loader = DataLoader(dataset, batch_size=256, shuffle=True)

        dataset.set_test()
        test_loader = DataLoader(dataset, batch_size=256, shuffle=False)

        # Benchmark
        benchmark = OptimizerBenchmark(
            model_fn=model_fn,
            train_loader=train_loader,
            test_loader=test_loader,
            device='cpu'
        )

        configs = {
            'adam': {'lr': [3e-4, 1e-3]},
            'belavkin': {
                'lr': [1e-2, 3e-2],
                'gamma': [0],
                'beta': [0],
            },
            'belavkin_quantum': {
                'lr': [1e-2],
                'gamma': [1e-4],
                'beta': [1e-2],
            }
        }

        print(f"\nRunning experiments...")
        results = benchmark.grid_search(
            optimizer_configs=configs,
            n_seeds=2,
            n_epochs=200,
            log_interval=50
        )

        # Analyze
        by_optimizer = {}
        for r in results:
            opt_name = r['optimizer']
            if opt_name not in by_optimizer:
                by_optimizer[opt_name] = []
            by_optimizer[opt_name].append(r)

        summary = {}
        for opt_name, runs in by_optimizer.items():
            best_run = max(runs, key=lambda x: x['best_test_accuracy'])
            summary[opt_name] = {
                'best_accuracy': best_run['best_test_accuracy'],
                'mean_accuracy': np.mean([r['best_test_accuracy'] for r in runs]),
                'std_accuracy': np.std([r['best_test_accuracy'] for r in runs]),
            }

        all_results[operation] = summary

        # Print
        print(f"\n{'='*70}")
        print(f"RESULTS FOR {operation.upper()}")
        print(f"{'='*70}")
        for opt_name in ['adam', 'belavkin', 'belavkin_quantum']:
            if opt_name in summary:
                s = summary[opt_name]
                print(f"{opt_name:20s}: {s['best_accuracy']:.4f} "
                      f"(mean: {s['mean_accuracy']:.4f}±{s['std_accuracy']:.4f})")

    # Compare across operations
    print("\n" + "="*70)
    print("OPERATION COMPARISON")
    print("="*70)

    print(f"\n{'Optimizer':20s}", end='')
    for op in operations:
        print(f" | {op:10s}", end='')
    print(f" | Difference")
    print("-" * 70)

    for opt in ['adam', 'belavkin', 'belavkin_quantum']:
        print(f"{opt:20s}", end='')
        accs = []
        for op in operations:
            if opt in all_results[op]:
                acc = all_results[op][opt]['best_accuracy']
                print(f" | {acc:10.4f}", end='')
                accs.append(acc)
            else:
                print(f" | {'N/A':10s}", end='')

        if len(accs) == 2:
            diff = accs[1] - accs[0]  # mult - add
            print(f" | {diff:+.4f}")
        else:
            print(f" | N/A")

    # Key findings
    print("\n" + "="*70)
    print("KEY FINDINGS")
    print("="*70)

    for op in operations:
        adam_acc = all_results[op].get('adam', {}).get('best_accuracy', 0)
        bel_acc = all_results[op].get('belavkin', {}).get('best_accuracy', 0)

        if adam_acc > 0 and bel_acc > 0:
            gap = adam_acc - bel_acc
            print(f"\n{op.upper()}:")
            print(f"  Adam: {adam_acc:.4f}")
            print(f"  Belavkin: {bel_acc:.4f}")
            print(f"  Gap: {gap:.4f} ({gap*100:.1f} percentage points)")

            if gap > 0.05:
                print(f"  → Adam wins by significant margin")
            elif gap < -0.05:
                print(f"  → Belavkin wins by significant margin")
            else:
                print(f"  → Roughly equal performance")

    # Quantum components
    print("\n" + "="*70)
    print("QUANTUM COMPONENTS EFFECT")
    print("="*70)

    for op in operations:
        if 'belavkin' in all_results[op] and 'belavkin_quantum' in all_results[op]:
            no_q = all_results[op]['belavkin']['best_accuracy']
            with_q = all_results[op]['belavkin_quantum']['best_accuracy']
            diff = with_q - no_q

            print(f"\n{op.upper()}:")
            print(f"  Without quantum: {no_q:.4f}")
            print(f"  With quantum: {with_q:.4f}")
            print(f"  Difference: {diff:+.4f}")

            if diff > 0.01:
                print(f"  → Quantum components HELP")
            elif diff < -0.01:
                print(f"  → Quantum components HURT")
            else:
                print(f"  → Quantum components NEUTRAL")

    # Save
    os.makedirs('results/operations', exist_ok=True)
    with open('results/operations/operation_comparison.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "="*70)
    print("Results saved to results/operations/operation_comparison.json")
    print("="*70)


if __name__ == '__main__':
    test_operations()

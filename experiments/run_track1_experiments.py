"""
Track 1 Experiment Runner

This script runs comprehensive experiments comparing the Belavkin optimizer
against standard baselines on synthetic tasks.

Usage:
    python run_track1_experiments.py --task modular --n_epochs 1000
    python run_track1_experiments.py --task parity --n_seeds 5
"""

import argparse
import torch
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.synthetic_tasks import (
    create_modular_task,
    create_sparse_parity_task,
)
from experiments.benchmark import OptimizerBenchmark


def run_modular_arithmetic_experiment(
    p: int = 97,
    operation: str = 'add',
    n_epochs: int = 1000,
    n_seeds: int = 3,
    output_dir: str = 'results/modular',
):
    """Run modular arithmetic experiments."""
    print(f"\n{'='*70}")
    print(f"MODULAR ARITHMETIC EXPERIMENT: {operation.upper()}, p={p}")
    print(f"{'='*70}\n")

    # Create task
    model_fn = lambda: __import__('experiments.synthetic_tasks', fromlist=['ModularArithmeticModel']).ModularArithmeticModel(
        p=p, hidden_dim=128, n_layers=2, operation=operation
    )

    from experiments.synthetic_tasks import ModularArithmeticDataset
    from torch.utils.data import DataLoader

    dataset = ModularArithmeticDataset(p=p, operation=operation, train_fraction=0.5)

    dataset.set_train()
    train_loader = DataLoader(dataset, batch_size=512, shuffle=True)

    dataset.set_test()
    test_loader = DataLoader(dataset, batch_size=512, shuffle=False)

    # Create benchmark
    benchmark = OptimizerBenchmark(
        model_fn=model_fn, train_loader=train_loader, test_loader=test_loader
    )

    # Define optimizer configurations for grid search
    optimizer_configs = {
        'sgd': {'lr': [1e-3, 3e-3, 1e-2], 'momentum': [0.9]},
        'adam': {'lr': [1e-4, 3e-4, 1e-3]},
        'belavkin': {
            'lr': [1e-4, 1e-3],
            'gamma': [1e-5, 1e-4, 1e-3],
            'beta': [1e-3, 1e-2],
        },
        'belavkin_minimal': {
            'lr': [1e-4, 1e-3],
            'gamma': [1e-5, 1e-4],
            'beta': [1e-3, 1e-2],
        },
    }

    # Run grid search
    results = benchmark.grid_search(
        optimizer_configs=optimizer_configs, n_seeds=n_seeds, n_epochs=n_epochs, log_interval=100
    )

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    benchmark.save_results(results, f'{output_dir}/results_{operation}_p{p}.json')

    # Print summary
    summary = benchmark.summarize_results(results)
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}\n")
    for (opt_name, params), stats in summary.items():
        print(f"{opt_name} | {params}")
        print(f"  Best Acc: {stats['best_accuracy_mean']:.4f} ± {stats['best_accuracy_std']:.4f}")
        print(f"  Final Acc: {stats['final_accuracy_mean']:.4f} ± {stats['final_accuracy_std']:.4f}")
        print(f"  Time: {stats['time_mean']:.2f}s ± {stats['time_std']:.2f}s")
        print()


def run_sparse_parity_experiment(
    n_bits: int = 10,
    k_sparse: int = 3,
    n_epochs: int = 500,
    n_seeds: int = 3,
    output_dir: str = 'results/parity',
):
    """Run sparse parity experiments."""
    print(f"\n{'='*70}")
    print(f"SPARSE PARITY EXPERIMENT: n={n_bits}, k={k_sparse}")
    print(f"{'='*70}\n")

    # Create task
    from experiments.synthetic_tasks import SimpleMLP

    model_fn = lambda: SimpleMLP(input_dim=n_bits, hidden_dims=[128, 128], output_dim=2)

    from experiments.synthetic_tasks import SparseParityDataset
    from torch.utils.data import DataLoader

    dataset = SparseParityDataset(
        n_bits=n_bits, k_sparse=k_sparse, n_samples=10000, train_fraction=0.8
    )

    dataset.set_train()
    train_loader = DataLoader(dataset, batch_size=128, shuffle=True)

    dataset.set_test()
    test_loader = DataLoader(dataset, batch_size=128, shuffle=False)

    # Create benchmark
    benchmark = OptimizerBenchmark(
        model_fn=model_fn, train_loader=train_loader, test_loader=test_loader
    )

    # Optimizer configurations
    optimizer_configs = {
        'adam': {'lr': [1e-4, 3e-4, 1e-3]},
        'sgd': {'lr': [1e-3, 1e-2], 'momentum': [0.9]},
        'belavkin': {
            'lr': [1e-4, 1e-3],
            'gamma': [1e-4, 1e-3],
            'beta': [1e-2, 1e-1],
        },
    }

    # Run grid search
    results = benchmark.grid_search(
        optimizer_configs=optimizer_configs, n_seeds=n_seeds, n_epochs=n_epochs, log_interval=50
    )

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    benchmark.save_results(results, f'{output_dir}/results_n{n_bits}_k{k_sparse}.json')

    # Print summary
    summary = benchmark.summarize_results(results)
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}\n")
    for (opt_name, params), stats in summary.items():
        print(f"{opt_name} | {params}")
        print(f"  Best Acc: {stats['best_accuracy_mean']:.4f} ± {stats['best_accuracy_std']:.4f}")
        print(f"  Final Acc: {stats['final_accuracy_mean']:.4f} ± {stats['final_accuracy_std']:.4f}")
        print()


def main():
    parser = argparse.ArgumentParser(description='Run Track 1 experiments')
    parser.add_argument(
        '--task',
        type=str,
        choices=['modular', 'parity', 'all'],
        default='modular',
        help='Task to run',
    )
    parser.add_argument('--n_epochs', type=int, default=1000, help='Number of epochs')
    parser.add_argument('--n_seeds', type=int, default=3, help='Number of random seeds')
    parser.add_argument('--output_dir', type=str, default='results', help='Output directory')

    # Modular arithmetic specific
    parser.add_argument('--p', type=int, default=97, help='Modulus for modular arithmetic')
    parser.add_argument(
        '--operation',
        type=str,
        choices=['add', 'mult', 'linear'],
        default='add',
        help='Operation type',
    )

    # Sparse parity specific
    parser.add_argument('--n_bits', type=int, default=10, help='Number of bits for parity')
    parser.add_argument('--k_sparse', type=int, default=3, help='Sparsity for parity')

    args = parser.parse_args()

    if args.task in ['modular', 'all']:
        run_modular_arithmetic_experiment(
            p=args.p,
            operation=args.operation,
            n_epochs=args.n_epochs,
            n_seeds=args.n_seeds,
            output_dir=f'{args.output_dir}/modular',
        )

    if args.task in ['parity', 'all']:
        run_sparse_parity_experiment(
            n_bits=args.n_bits,
            k_sparse=args.k_sparse,
            n_epochs=args.n_epochs,
            n_seeds=args.n_seeds,
            output_dir=f'{args.output_dir}/parity',
        )


if __name__ == '__main__':
    main()

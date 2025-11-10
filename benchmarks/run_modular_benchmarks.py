"""
Run comprehensive benchmarks on modular arithmetic and composition tasks.

This script compares Belavkin optimizer against Adam, SGD, and RMSprop.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
from pathlib import Path

from belavkin_optimizer import BelavkinOptimizer
from belavkin_optimizer.belavkin import BelavkinOptimizerV2
from datasets import (
    generate_modular_addition,
    generate_modular_multiplication,
    generate_composition_data,
)
from benchmarks.models import ModularArithmeticMLP, ModularCompositionMLP
from benchmarks.trainer import BenchmarkRunner
from benchmarks.utils import set_seed, save_results


def run_modular_addition_benchmark(
    modulus: int = 97,
    num_samples: int = 10000,
    batch_size: int = 128,
    num_epochs: int = 100,
    num_runs: int = 3,
    output_dir: str = 'results',
    seed: int = 42,
):
    """Run benchmark on modular addition task."""
    print("\n" + "="*80)
    print("MODULAR ADDITION BENCHMARK")
    print("="*80)

    set_seed(seed)

    # Generate datasets
    train_ds, test_ds = generate_modular_addition(
        modulus=modulus,
        num_samples=num_samples,
        seed=seed,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # Model factory
    def model_factory():
        return ModularArithmeticMLP(
            vocab_size=modulus,
            embedding_dim=64,
            hidden_dims=[128, 128],
            dropout=0.1,
        )

    # Optimizer configurations
    optimizer_configs = {
        'Belavkin': {
            'class': BelavkinOptimizer,
            'lr': 0.001,
            'gamma': 1e-4,
            'beta': 1e-5,
            'adaptive_gamma': True,
        },
        'BelavkinV2': {
            'class': BelavkinOptimizerV2,
            'lr': 0.001,
            'gamma': 1e-6,
            'beta': 1e-5,
        },
        'Adam': {
            'class': torch.optim.Adam,
            'lr': 0.001,
        },
        'SGD': {
            'class': torch.optim.SGD,
            'lr': 0.01,
            'momentum': 0.9,
        },
        'RMSprop': {
            'class': torch.optim.RMSprop,
            'lr': 0.001,
        },
    }

    # Run benchmark
    runner = BenchmarkRunner(
        model_factory=model_factory,
        optimizer_configs=optimizer_configs,
        criterion=nn.CrossEntropyLoss(),
        seed=seed,
    )

    results = runner.run_benchmark(
        train_loader=train_loader,
        val_loader=test_loader,
        num_epochs=num_epochs,
        num_runs=num_runs,
        verbose=True,
    )

    # Print comparison table
    print(runner.get_comparison_table())

    # Save results
    output_path = Path(output_dir) / 'modular_addition_results.pkl'
    save_results(results, str(output_path))
    print(f"\nResults saved to {output_path}")

    return results


def run_modular_multiplication_benchmark(
    modulus: int = 97,
    num_samples: int = 10000,
    batch_size: int = 128,
    num_epochs: int = 100,
    num_runs: int = 3,
    output_dir: str = 'results',
    seed: int = 42,
):
    """Run benchmark on modular multiplication task."""
    print("\n" + "="*80)
    print("MODULAR MULTIPLICATION BENCHMARK")
    print("="*80)

    set_seed(seed)

    # Generate datasets
    train_ds, test_ds = generate_modular_multiplication(
        modulus=modulus,
        num_samples=num_samples,
        seed=seed,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # Model factory
    def model_factory():
        return ModularArithmeticMLP(
            vocab_size=modulus,
            embedding_dim=64,
            hidden_dims=[128, 128],
            dropout=0.1,
        )

    # Optimizer configurations
    optimizer_configs = {
        'Belavkin': {
            'class': BelavkinOptimizer,
            'lr': 0.001,
            'gamma': 1e-4,
            'beta': 1e-5,
            'adaptive_gamma': True,
        },
        'Adam': {
            'class': torch.optim.Adam,
            'lr': 0.001,
        },
        'SGD': {
            'class': torch.optim.SGD,
            'lr': 0.01,
            'momentum': 0.9,
        },
        'RMSprop': {
            'class': torch.optim.RMSprop,
            'lr': 0.001,
        },
    }

    # Run benchmark
    runner = BenchmarkRunner(
        model_factory=model_factory,
        optimizer_configs=optimizer_configs,
        criterion=nn.CrossEntropyLoss(),
        seed=seed,
    )

    results = runner.run_benchmark(
        train_loader=train_loader,
        val_loader=test_loader,
        num_epochs=num_epochs,
        num_runs=num_runs,
        verbose=True,
    )

    # Print comparison table
    print(runner.get_comparison_table())

    # Save results
    output_path = Path(output_dir) / 'modular_multiplication_results.pkl'
    save_results(results, str(output_path))
    print(f"\nResults saved to {output_path}")

    return results


def run_composition_benchmark(
    composition_type: str = 'two_layer',
    modulus: int = 97,
    num_samples: int = 10000,
    batch_size: int = 128,
    num_epochs: int = 100,
    num_runs: int = 3,
    output_dir: str = 'results',
    seed: int = 42,
):
    """Run benchmark on modular composition task."""
    print("\n" + "="*80)
    print(f"MODULAR COMPOSITION BENCHMARK ({composition_type})")
    print("="*80)

    set_seed(seed)

    # Generate datasets
    train_ds, test_ds = generate_composition_data(
        composition_type=composition_type,
        modulus=modulus,
        num_samples=num_samples,
        seed=seed,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # Determine number of inputs
    sample_data, _ = train_ds[0]
    num_inputs = sample_data.shape[0] if len(sample_data.shape) > 0 else 1

    # Model factory
    def model_factory():
        return ModularCompositionMLP(
            vocab_size=modulus,
            num_inputs=num_inputs,
            embedding_dim=64,
            hidden_dims=[128, 128, 128],
            dropout=0.1,
        )

    # Optimizer configurations
    optimizer_configs = {
        'Belavkin': {
            'class': BelavkinOptimizer,
            'lr': 0.001,
            'gamma': 1e-4,
            'beta': 1e-5,
            'adaptive_gamma': True,
        },
        'Adam': {
            'class': torch.optim.Adam,
            'lr': 0.001,
        },
        'SGD': {
            'class': torch.optim.SGD,
            'lr': 0.01,
            'momentum': 0.9,
        },
        'RMSprop': {
            'class': torch.optim.RMSprop,
            'lr': 0.001,
        },
    }

    # Run benchmark
    runner = BenchmarkRunner(
        model_factory=model_factory,
        optimizer_configs=optimizer_configs,
        criterion=nn.CrossEntropyLoss(),
        seed=seed,
    )

    results = runner.run_benchmark(
        train_loader=train_loader,
        val_loader=test_loader,
        num_epochs=num_epochs,
        num_runs=num_runs,
        verbose=True,
    )

    # Print comparison table
    print(runner.get_comparison_table())

    # Save results
    output_path = Path(output_dir) / f'composition_{composition_type}_results.pkl'
    save_results(results, str(output_path))
    print(f"\nResults saved to {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Run modular benchmarks')
    parser.add_argument('--task', type=str, default='all',
                        choices=['all', 'addition', 'multiplication', 'composition'],
                        help='Which benchmark to run')
    parser.add_argument('--modulus', type=int, default=97,
                        help='Modulus for modular arithmetic')
    parser.add_argument('--num-samples', type=int, default=10000,
                        help='Number of samples')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size')
    parser.add_argument('--num-epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--num-runs', type=int, default=3,
                        help='Number of runs per optimizer')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Output directory for results')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    args = parser.parse_args()

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Run benchmarks
    all_results = {}

    if args.task in ['all', 'addition']:
        results = run_modular_addition_benchmark(
            modulus=args.modulus,
            num_samples=args.num_samples,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            num_runs=args.num_runs,
            output_dir=args.output_dir,
            seed=args.seed,
        )
        all_results['addition'] = results

    if args.task in ['all', 'multiplication']:
        results = run_modular_multiplication_benchmark(
            modulus=args.modulus,
            num_samples=args.num_samples,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            num_runs=args.num_runs,
            output_dir=args.output_dir,
            seed=args.seed + 100,
        )
        all_results['multiplication'] = results

    if args.task in ['all', 'composition']:
        for comp_type in ['two_layer', 'three_layer', 'mixed', 'polynomial']:
            results = run_composition_benchmark(
                composition_type=comp_type,
                modulus=args.modulus,
                num_samples=args.num_samples,
                batch_size=args.batch_size,
                num_epochs=args.num_epochs,
                num_runs=args.num_runs,
                output_dir=args.output_dir,
                seed=args.seed + 200,
            )
            all_results[f'composition_{comp_type}'] = results

    print("\n" + "="*80)
    print("ALL BENCHMARKS COMPLETED")
    print("="*80)


if __name__ == '__main__':
    main()

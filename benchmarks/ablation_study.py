"""
Ablation studies for Belavkin optimizer hyperparameters.

Tests the effect of varying γ, η (lr), and β on performance.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
from pathlib import Path
import numpy as np

from belavkin_optimizer import BelavkinOptimizer
from datasets import generate_modular_addition
from benchmarks.models import ModularArithmeticMLP
from benchmarks.trainer import Trainer
from benchmarks.utils import set_seed, save_results
from benchmarks.visualize import plot_ablation_results


def run_ablation_gamma(
    gamma_values: list = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
    modulus: int = 97,
    num_samples: int = 10000,
    batch_size: int = 128,
    num_epochs: int = 100,
    num_runs: int = 3,
    output_dir: str = 'results/ablation',
    seed: int = 42,
):
    """
    Ablation study on gamma (adaptive damping factor).
    """
    print("\n" + "="*80)
    print("ABLATION STUDY: GAMMA (Adaptive Damping Factor)")
    print("="*80)

    results = {}

    for gamma in gamma_values:
        print(f"\n{'='*60}")
        print(f"Testing gamma = {gamma}")
        print(f"{'='*60}")

        run_results = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
        }

        for run in range(num_runs):
            print(f"Run {run+1}/{num_runs}")

            set_seed(seed + run)

            # Generate data
            train_ds, test_ds = generate_modular_addition(
                modulus=modulus,
                num_samples=num_samples,
                seed=seed + run,
            )

            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

            # Create model
            model = ModularArithmeticMLP(
                vocab_size=modulus,
                embedding_dim=64,
                hidden_dims=[128, 128],
                dropout=0.1,
            )

            # Create optimizer with specific gamma
            optimizer = BelavkinOptimizer(
                model.parameters(),
                lr=0.001,
                gamma=gamma,
                beta=1e-5,
                adaptive_gamma=True,
            )

            # Train
            trainer = Trainer(
                model=model,
                optimizer=optimizer,
                criterion=nn.CrossEntropyLoss(),
            )

            history = trainer.train(
                train_loader=train_loader,
                val_loader=test_loader,
                num_epochs=num_epochs,
                verbose=False,
            )

            # Store results
            run_results['train_loss'].append(history['train']['loss'])
            run_results['train_accuracy'].append(history['train']['accuracy'])
            run_results['val_loss'].append(history['val']['loss'])
            run_results['val_accuracy'].append(history['val']['accuracy'])

        # Compute statistics
        results[gamma] = {
            'train_loss_mean': np.array(run_results['train_loss']).mean(axis=0),
            'train_loss_std': np.array(run_results['train_loss']).std(axis=0),
            'train_accuracy_mean': np.array(run_results['train_accuracy']).mean(axis=0),
            'train_accuracy_std': np.array(run_results['train_accuracy']).std(axis=0),
            'val_loss_mean': np.array(run_results['val_loss']).mean(axis=0),
            'val_loss_std': np.array(run_results['val_loss']).std(axis=0),
            'val_accuracy_mean': np.array(run_results['val_accuracy']).mean(axis=0),
            'val_accuracy_std': np.array(run_results['val_accuracy']).std(axis=0),
        }

        print(f"Final Val Acc: {results[gamma]['val_accuracy_mean'][-1]:.4f} ± {results[gamma]['val_accuracy_std'][-1]:.4f}")

    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    save_results(results, str(output_path / 'ablation_gamma.pkl'))

    # Visualize
    plot_ablation_results(results, 'gamma', str(output_path / 'ablation_gamma.png'))

    return results


def run_ablation_beta(
    beta_values: list = [0.0, 1e-6, 1e-5, 1e-4, 1e-3],
    modulus: int = 97,
    num_samples: int = 10000,
    batch_size: int = 128,
    num_epochs: int = 100,
    num_runs: int = 3,
    output_dir: str = 'results/ablation',
    seed: int = 42,
):
    """
    Ablation study on beta (stochastic exploration factor).
    """
    print("\n" + "="*80)
    print("ABLATION STUDY: BETA (Stochastic Exploration Factor)")
    print("="*80)

    results = {}

    for beta in beta_values:
        print(f"\n{'='*60}")
        print(f"Testing beta = {beta}")
        print(f"{'='*60}")

        run_results = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
        }

        for run in range(num_runs):
            print(f"Run {run+1}/{num_runs}")

            set_seed(seed + run)

            # Generate data
            train_ds, test_ds = generate_modular_addition(
                modulus=modulus,
                num_samples=num_samples,
                seed=seed + run,
            )

            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

            # Create model
            model = ModularArithmeticMLP(
                vocab_size=modulus,
                embedding_dim=64,
                hidden_dims=[128, 128],
                dropout=0.1,
            )

            # Create optimizer with specific beta
            optimizer = BelavkinOptimizer(
                model.parameters(),
                lr=0.001,
                gamma=1e-4,
                beta=beta,
                adaptive_gamma=True,
            )

            # Train
            trainer = Trainer(
                model=model,
                optimizer=optimizer,
                criterion=nn.CrossEntropyLoss(),
            )

            history = trainer.train(
                train_loader=train_loader,
                val_loader=test_loader,
                num_epochs=num_epochs,
                verbose=False,
            )

            # Store results
            run_results['train_loss'].append(history['train']['loss'])
            run_results['train_accuracy'].append(history['train']['accuracy'])
            run_results['val_loss'].append(history['val']['loss'])
            run_results['val_accuracy'].append(history['val']['accuracy'])

        # Compute statistics
        results[beta] = {
            'train_loss_mean': np.array(run_results['train_loss']).mean(axis=0),
            'train_loss_std': np.array(run_results['train_loss']).std(axis=0),
            'train_accuracy_mean': np.array(run_results['train_accuracy']).mean(axis=0),
            'train_accuracy_std': np.array(run_results['train_accuracy']).std(axis=0),
            'val_loss_mean': np.array(run_results['val_loss']).mean(axis=0),
            'val_loss_std': np.array(run_results['val_loss']).std(axis=0),
            'val_accuracy_mean': np.array(run_results['val_accuracy']).mean(axis=0),
            'val_accuracy_std': np.array(run_results['val_accuracy']).std(axis=0),
        }

        print(f"Final Val Acc: {results[beta]['val_accuracy_mean'][-1]:.4f} ± {results[beta]['val_accuracy_std'][-1]:.4f}")

    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    save_results(results, str(output_path / 'ablation_beta.pkl'))

    # Visualize
    plot_ablation_results(results, 'beta', str(output_path / 'ablation_beta.png'))

    return results


def run_ablation_lr(
    lr_values: list = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
    modulus: int = 97,
    num_samples: int = 10000,
    batch_size: int = 128,
    num_epochs: int = 100,
    num_runs: int = 3,
    output_dir: str = 'results/ablation',
    seed: int = 42,
):
    """
    Ablation study on learning rate (η).
    """
    print("\n" + "="*80)
    print("ABLATION STUDY: LEARNING RATE (η)")
    print("="*80)

    results = {}

    for lr in lr_values:
        print(f"\n{'='*60}")
        print(f"Testing lr = {lr}")
        print(f"{'='*60}")

        run_results = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
        }

        for run in range(num_runs):
            print(f"Run {run+1}/{num_runs}")

            set_seed(seed + run)

            # Generate data
            train_ds, test_ds = generate_modular_addition(
                modulus=modulus,
                num_samples=num_samples,
                seed=seed + run,
            )

            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

            # Create model
            model = ModularArithmeticMLP(
                vocab_size=modulus,
                embedding_dim=64,
                hidden_dims=[128, 128],
                dropout=0.1,
            )

            # Create optimizer with specific lr
            optimizer = BelavkinOptimizer(
                model.parameters(),
                lr=lr,
                gamma=1e-4,
                beta=1e-5,
                adaptive_gamma=True,
            )

            # Train
            trainer = Trainer(
                model=model,
                optimizer=optimizer,
                criterion=nn.CrossEntropyLoss(),
            )

            history = trainer.train(
                train_loader=train_loader,
                val_loader=test_loader,
                num_epochs=num_epochs,
                verbose=False,
            )

            # Store results
            run_results['train_loss'].append(history['train']['loss'])
            run_results['train_accuracy'].append(history['train']['accuracy'])
            run_results['val_loss'].append(history['val']['loss'])
            run_results['val_accuracy'].append(history['val']['accuracy'])

        # Compute statistics
        results[lr] = {
            'train_loss_mean': np.array(run_results['train_loss']).mean(axis=0),
            'train_loss_std': np.array(run_results['train_loss']).std(axis=0),
            'train_accuracy_mean': np.array(run_results['train_accuracy']).mean(axis=0),
            'train_accuracy_std': np.array(run_results['train_accuracy']).std(axis=0),
            'val_loss_mean': np.array(run_results['val_loss']).mean(axis=0),
            'val_loss_std': np.array(run_results['val_loss']).std(axis=0),
            'val_accuracy_mean': np.array(run_results['val_accuracy']).mean(axis=0),
            'val_accuracy_std': np.array(run_results['val_accuracy']).std(axis=0),
        }

        print(f"Final Val Acc: {results[lr]['val_accuracy_mean'][-1]:.4f} ± {results[lr]['val_accuracy_std'][-1]:.4f}")

    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    save_results(results, str(output_path / 'ablation_lr.pkl'))

    # Visualize
    plot_ablation_results(results, 'learning_rate', str(output_path / 'ablation_lr.png'))

    return results


def main():
    parser = argparse.ArgumentParser(description='Run ablation studies')
    parser.add_argument('--param', type=str, default='all',
                        choices=['all', 'gamma', 'beta', 'lr'],
                        help='Which parameter to ablate')
    parser.add_argument('--output-dir', type=str, default='results/ablation',
                        help='Output directory')
    parser.add_argument('--num-epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--num-runs', type=int, default=3,
                        help='Number of runs per configuration')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    args = parser.parse_args()

    if args.param in ['all', 'gamma']:
        run_ablation_gamma(
            num_epochs=args.num_epochs,
            num_runs=args.num_runs,
            output_dir=args.output_dir,
            seed=args.seed,
        )

    if args.param in ['all', 'beta']:
        run_ablation_beta(
            num_epochs=args.num_epochs,
            num_runs=args.num_runs,
            output_dir=args.output_dir,
            seed=args.seed + 100,
        )

    if args.param in ['all', 'lr']:
        run_ablation_lr(
            num_epochs=args.num_epochs,
            num_runs=args.num_runs,
            output_dir=args.output_dir,
            seed=args.seed + 200,
        )

    print("\n" + "="*80)
    print("ALL ABLATION STUDIES COMPLETED")
    print("="*80)


if __name__ == '__main__':
    main()

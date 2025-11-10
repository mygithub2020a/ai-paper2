"""
Example experiment: Belavkin optimizer on sparse parity task.

This script tests the optimizer's ability to discover sparse combinatorial structure.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
from belavkin_ml.tasks.sparse_parity import create_sparse_parity_task, ParityMLP
from belavkin_ml.benchmarks.optimizer_bench import OptimizerBenchmark, run_optimizer_comparison
from belavkin_ml.utils.visualization import plot_optimizer_comparison, plot_convergence_analysis
import matplotlib.pyplot as plt


def main():
    """Run sparse parity task experiment."""
    print("="*60)
    print("Belavkin Optimizer - Sparse Parity Task")
    print("="*60)

    # Configuration
    n_bits = 10
    k_sparse = 3
    n_samples = 2000
    hidden_dims = [128]
    n_seeds = 3
    max_epochs = 2000
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"\nConfiguration:")
    print(f"  Input bits: {n_bits}")
    print(f"  Sparsity: {k_sparse}")
    print(f"  Samples: {n_samples}")
    print(f"  Hidden dims: {hidden_dims}")
    print(f"  Max epochs: {max_epochs}")
    print(f"  Device: {device}")

    # Create task
    print(f"\nCreating sparse parity task...")
    model, train_loader, test_loader, dataset = create_sparse_parity_task(
        n_bits=n_bits,
        k_sparse=k_sparse,
        n_samples=n_samples,
        hidden_dims=hidden_dims,
        train_fraction=0.8,
        batch_size=64,
        seed=42,
    )

    print(f"True parity bits: {dataset.get_true_parity_bits()}")

    # Model factory
    def model_factory():
        return ParityMLP(
            n_bits=n_bits,
            hidden_dims=hidden_dims,
        )

    # Create benchmark
    benchmark = OptimizerBenchmark(
        model_factory=model_factory,
        train_loader=train_loader,
        test_loader=test_loader,
        criterion=nn.CrossEntropyLoss(),
        device=device,
        max_epochs=max_epochs,
        eval_every=20,
        early_stop_patience=200,
        target_accuracy=0.95,
    )

    # Define optimizers
    optimizers = {
        'Adam': {'lr': 1e-3},
        'SGD': {'lr': 1e-2, 'momentum': 0.9},
        'Belavkin': {'lr': 1e-3, 'gamma': 1e-4, 'beta': 1e-2, 'adaptive_gamma': True},
        'Belavkin-Fixed': {'lr': 1e-3, 'gamma': 1e-4, 'beta': 1e-2, 'adaptive_gamma': False},
    }

    # Run comparison
    results_dir = Path(__file__).parent / 'results' / 'sparse_parity'
    results_dir.mkdir(parents=True, exist_ok=True)

    all_results = run_optimizer_comparison(
        benchmark=benchmark,
        optimizers=optimizers,
        n_seeds=n_seeds,
        save_dir=results_dir,
        verbose=True,
    )

    # Visualize
    fig = plot_optimizer_comparison(
        all_results=all_results,
        metric='test_accs',
        save_path=results_dir / 'training_curves.png',
        title=f'{k_sparse}-Sparse Parity on {n_bits} bits',
    )
    plt.close(fig)

    fig = plot_convergence_analysis(
        all_results=all_results,
        save_path=results_dir / 'convergence_analysis.png',
    )
    plt.close(fig)

    print(f"\nResults saved to: {results_dir}")


if __name__ == '__main__':
    main()

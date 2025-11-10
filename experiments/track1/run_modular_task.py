"""
Example experiment: Belavkin optimizer on modular arithmetic task.

This script demonstrates:
1. Creating a modular arithmetic task
2. Comparing Belavkin optimizer against baselines (Adam, SGD, SGLD)
3. Visualizing training curves and convergence analysis
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
from belavkin_ml.tasks.modular import create_modular_task, ModularMLP
from belavkin_ml.benchmarks.optimizer_bench import OptimizerBenchmark, run_optimizer_comparison
from belavkin_ml.utils.visualization import plot_optimizer_comparison, plot_convergence_analysis
import matplotlib.pyplot as plt


def main():
    """Run modular arithmetic task experiment."""
    print("="*60)
    print("Belavkin Optimizer - Modular Arithmetic Task")
    print("="*60)

    # Configuration
    p = 97  # Prime modulus
    hidden_dims = [512]
    n_seeds = 3
    max_epochs = 5000
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"\nConfiguration:")
    print(f"  Prime modulus: {p}")
    print(f"  Hidden dims: {hidden_dims}")
    print(f"  Max epochs: {max_epochs}")
    print(f"  Device: {device}")
    print(f"  Seeds: {n_seeds}")

    # Create task
    print(f"\nCreating modular task...")
    model, train_loader, test_loader = create_modular_task(
        task_type='arithmetic',
        p=p,
        hidden_dims=hidden_dims,
        train_fraction=0.5,
        batch_size=256,
        seed=42,
    )

    # Model factory
    def model_factory():
        return ModularMLP(
            input_dim=p,
            output_dim=p,
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
        eval_every=50,
        early_stop_patience=500,
        target_accuracy=0.95,
    )

    # Define optimizers to compare
    optimizers = {
        'Adam': {'lr': 1e-3},
        'SGD': {'lr': 1e-2, 'momentum': 0.9},
        'Belavkin': {'lr': 1e-3, 'gamma': 1e-4, 'beta': 1e-2},
        'BelavkinSGD': {'lr': 1e-2, 'gamma': 1e-3, 'beta': 1e-4},
        'SGLD': {'lr': 1e-3, 'temperature': 1e-4},
    }

    # Run comparison
    print(f"\n{'='*60}")
    print("Running optimizer comparison...")
    print(f"{'='*60}")

    results_dir = Path(__file__).parent / 'results' / 'modular_arithmetic'
    results_dir.mkdir(parents=True, exist_ok=True)

    all_results = run_optimizer_comparison(
        benchmark=benchmark,
        optimizers=optimizers,
        n_seeds=n_seeds,
        save_dir=results_dir,
        verbose=True,
    )

    # Visualize results
    print(f"\nGenerating visualizations...")

    # Training curves comparison
    fig = plot_optimizer_comparison(
        all_results=all_results,
        metric='test_accs',
        save_path=results_dir / 'training_curves.png',
        title=f'Modular Arithmetic (p={p}) - Test Accuracy',
    )
    plt.close(fig)

    # Convergence analysis
    fig = plot_convergence_analysis(
        all_results=all_results,
        save_path=results_dir / 'convergence_analysis.png',
    )
    plt.close(fig)

    print(f"\nResults saved to: {results_dir}")

    # Print summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")

    for opt_name, opt_results in all_results.items():
        final_accs = [r['final_test_acc'] for r in opt_results]
        conv_epochs = [r['convergence_epoch'] for r in opt_results
                       if r['convergence_epoch'] is not None]

        print(f"\n{opt_name}:")
        print(f"  Final accuracy: {sum(final_accs)/len(final_accs):.4f} ± {torch.tensor(final_accs).std():.4f}")

        if len(conv_epochs) > 0:
            print(f"  Convergence epoch: {sum(conv_epochs)/len(conv_epochs):.1f} ± {torch.tensor(conv_epochs).std():.1f}")
        else:
            print(f"  Convergence epoch: Did not converge")


if __name__ == '__main__':
    main()

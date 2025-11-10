"""
Experiment: Belavkin optimizer on Sparse Parity task.

Tests sample efficiency on learning k-sparse XOR functions.

Usage:
    python run_sparse_parity.py --n_bits 10 --k_sparse 3
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
import argparse
import matplotlib.pyplot as plt
import seaborn as sns

from belavkin_ml.datasets.synthetic import SparseParityDataset, create_dataloaders
from belavkin_ml.experiments.benchmark import OptimizerBenchmark, BenchmarkConfig
from belavkin_ml.utils.visualization import create_analysis_report

sns.set_style('whitegrid')


class ParityMLP(nn.Module):
    """MLP for sparse parity task."""

    def __init__(self, n_bits: int, hidden_dims=[128, 128]):
        super().__init__()

        layers = []
        prev_dim = n_bits

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 2))  # Binary classification

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def main(args):
    print("="*70)
    print("Belavkin Optimizer: Sparse Parity Task")
    print("="*70)

    # Create dataset
    print(f"\nCreating dataset:")
    print(f"  - Number of bits: {args.n_bits}")
    print(f"  - k-sparse: {args.k_sparse}")
    print(f"  - Training examples: {args.n_examples}")

    dataset = SparseParityDataset(
        n_bits=args.n_bits,
        k_sparse=args.k_sparse,
        n_examples=args.n_examples,
        train_fraction=args.train_fraction,
        seed=args.seed,
    )

    info = dataset.get_info()
    print(f"  - Relevant bits: {info['relevant_bits']}")
    print(f"  - Train examples: {info['train_examples']}")
    print(f"  - Test examples: {info['test_examples']}")

    # Create dataloaders
    train_loader, test_loader = create_dataloaders(
        dataset,
        batch_size=args.batch_size,
        num_workers=0,
    )

    # Model factory
    def model_fn():
        return ParityMLP(n_bits=args.n_bits, hidden_dims=args.hidden_dims)

    # Configure benchmark
    config = BenchmarkConfig(
        optimizers=args.optimizers,
        learning_rates=args.learning_rates,
        gammas=args.gammas,
        betas=args.betas,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        n_seeds=args.n_seeds,
        early_stopping_patience=args.patience,
        target_accuracies=[0.90, 0.95, 0.99],
        save_dir=Path(args.save_dir) / f"sparse_parity_n{args.n_bits}_k{args.k_sparse}",
    )

    print(f"\nBenchmark configuration:")
    print(f"  - Optimizers: {config.optimizers}")
    print(f"  - Epochs: {config.n_epochs}")
    print(f"  - Seeds: {config.n_seeds}")

    # Run benchmark
    benchmark = OptimizerBenchmark(config)
    results = benchmark.run(model_fn, train_loader, test_loader)

    # Save results
    benchmark.save_results(results)

    # Generate analysis
    create_analysis_report(
        results,
        save_dir=config.save_dir,
        experiment_name="sparse_parity"
    )

    # Print summary
    print("\n" + "="*70)
    print("BENCHMARK SUMMARY")
    print("="*70)

    for opt_name, opt_results in results.items():
        print(f"\n{opt_name.upper()}:")

        best_result = max(opt_results, key=lambda x: x['best_test_acc'])

        print(f"  Best test accuracy: {best_result['best_test_acc']:.4f}")
        print(f"  Best hyperparameters:")
        print(f"    - lr: {best_result['lr']:.0e}")
        if best_result['gamma'] is not None:
            print(f"    - gamma: {best_result['gamma']:.0e}")
        if best_result['beta'] is not None:
            print(f"    - beta: {best_result['beta']:.0e}")

    print("\n" + "="*70)
    print(f"Results saved to: {config.save_dir}")
    print("="*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sparse parity benchmark")

    # Dataset
    parser.add_argument("--n_bits", type=int, default=10, help="Number of input bits")
    parser.add_argument("--k_sparse", type=int, default=3, help="Number of relevant bits")
    parser.add_argument("--n_examples", type=int, default=1000, help="Total examples")
    parser.add_argument("--train_fraction", type=float, default=0.8,
                        help="Fraction for training")

    # Model
    parser.add_argument("--hidden_dims", type=int, nargs="+", default=[128, 128],
                        help="Hidden layer dimensions")

    # Training
    parser.add_argument("--n_epochs", type=int, default=200, help="Max epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--patience", type=int, default=30, help="Early stopping patience")

    # Optimizers
    parser.add_argument("--optimizers", type=str, nargs="+",
                        default=["sgd", "adam", "belavkin", "adaptive_belavkin"],
                        help="Optimizers to compare")
    parser.add_argument("--learning_rates", type=float, nargs="+",
                        default=[1e-4, 3e-4, 1e-3, 3e-3],
                        help="Learning rates")
    parser.add_argument("--gammas", type=float, nargs="+",
                        default=[1e-5, 1e-4, 1e-3],
                        help="Gamma values")
    parser.add_argument("--betas", type=float, nargs="+",
                        default=[1e-3, 1e-2],
                        help="Beta values")

    # Other
    parser.add_argument("--n_seeds", type=int, default=3, help="Number of seeds")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--save_dir", type=str,
                        default="experiments/track1_optimizer/benchmarks",
                        help="Save directory")

    args = parser.parse_args()
    main(args)

"""
Example experiment: Benchmarking Belavkin optimizer on modular arithmetic.

This script demonstrates the full pipeline:
1. Create synthetic dataset (modular arithmetic)
2. Define a simple MLP model
3. Run benchmark comparing all optimizers
4. Analyze and visualize results

Usage:
    python run_modular_arithmetic.py --p 97 --operation addition --epochs 200
"""

import torch
import torch.nn as nn
import argparse
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from belavkin_ml.datasets.synthetic import ModularArithmeticDataset, create_dataloaders
from belavkin_ml.experiments.benchmark import OptimizerBenchmark, BenchmarkConfig


class ModularArithmeticMLP(nn.Module):
    """
    MLP for modular arithmetic tasks.

    Args:
        input_dim: Input dimension (1 for linear, 2 for binary operations)
        output_dim: Output dimension (equals modulus p)
        hidden_dims: List of hidden layer dimensions
    """

    def __init__(self, input_dim: int, output_dim: int, hidden_dims=[128, 128]):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def main(args):
    print("="*70)
    print("Belavkin Optimizer Benchmark: Modular Arithmetic")
    print("="*70)

    # Create dataset
    print(f"\nCreating dataset:")
    print(f"  - Modulus p = {args.p}")
    print(f"  - Operation = {args.operation}")
    print(f"  - Train fraction = {args.train_fraction}")

    dataset = ModularArithmeticDataset(
        p=args.p,
        operation=args.operation,
        train_fraction=args.train_fraction,
        seed=args.seed,
    )

    info = dataset.get_info()
    print(f"  - Train examples: {info['train_examples']}")
    print(f"  - Test examples: {info['test_examples']}")
    print(f"  - Input dim: {info['input_dim']}")
    print(f"  - Output dim: {info['output_dim']}")

    # Create dataloaders
    train_loader, test_loader = create_dataloaders(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # Define model factory
    def model_fn():
        return ModularArithmeticMLP(
            input_dim=info['input_dim'],
            output_dim=info['output_dim'],
            hidden_dims=args.hidden_dims,
        )

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
        save_dir=Path(args.save_dir) / f"modular_{args.operation}_p{args.p}",
        num_workers=args.num_workers,
    )

    print(f"\nBenchmark configuration:")
    print(f"  - Optimizers: {config.optimizers}")
    print(f"  - Learning rates: {config.learning_rates}")
    print(f"  - Gammas (Belavkin): {config.gammas}")
    print(f"  - Betas (Belavkin): {config.betas}")
    print(f"  - Epochs: {config.n_epochs}")
    print(f"  - Seeds: {config.n_seeds}")
    print(f"  - Device: {config.device}")

    # Run benchmark
    benchmark = OptimizerBenchmark(config)
    results = benchmark.run(model_fn, train_loader, test_loader)

    # Save results
    benchmark.save_results(results)

    # Print summary
    print("\n" + "="*70)
    print("BENCHMARK SUMMARY")
    print("="*70)

    for opt_name, opt_results in results.items():
        print(f"\n{opt_name.upper()}:")

        # Find best configuration
        best_result = max(opt_results, key=lambda x: x['best_test_acc'])

        print(f"  Best test accuracy: {best_result['best_test_acc']:.4f}")
        print(f"  Best hyperparameters:")
        print(f"    - lr: {best_result['lr']:.0e}")
        if best_result['gamma'] is not None:
            print(f"    - gamma: {best_result['gamma']:.0e}")
        if best_result['beta'] is not None:
            print(f"    - beta: {best_result['beta']:.0e}")

        # Average across seeds for best hyperparameters
        best_config_results = [
            r for r in opt_results
            if r['lr'] == best_result['lr']
            and r['gamma'] == best_result['gamma']
            and r['beta'] == best_result['beta']
        ]

        if len(best_config_results) > 1:
            avg_acc = sum(r['best_test_acc'] for r in best_config_results) / len(best_config_results)
            std_acc = (sum((r['best_test_acc'] - avg_acc)**2 for r in best_config_results) / len(best_config_results))**0.5
            print(f"  Across {len(best_config_results)} seeds:")
            print(f"    - Mean accuracy: {avg_acc:.4f} ± {std_acc:.4f}")

        # Steps to targets
        print(f"  Steps to target (best config):")
        for target, steps in best_result['steps_to_target'].items():
            if steps is not None:
                print(f"    - {target:.1%}: {steps} steps")
            else:
                print(f"    - {target:.1%}: Not reached")

        print(f"  Average epoch time: {best_result['avg_epoch_time']:.3f}s")
        print(f"  Total training time: {best_result['total_time']:.1f}s")

    print("\n" + "="*70)
    print("Benchmark complete! Results saved to:", config.save_dir)
    print("="*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Belavkin optimizer on modular arithmetic")

    # Dataset arguments
    parser.add_argument("--p", type=int, default=97, help="Prime modulus")
    parser.add_argument("--operation", type=str, default="addition",
                        choices=["linear", "addition", "multiplication", "division"],
                        help="Modular operation type")
    parser.add_argument("--train_fraction", type=float, default=0.5,
                        help="Fraction of data for training")

    # Model arguments
    parser.add_argument("--hidden_dims", type=int, nargs="+", default=[128, 128],
                        help="Hidden layer dimensions")

    # Training arguments
    parser.add_argument("--n_epochs", type=int, default=200,
                        help="Maximum number of epochs")
    parser.add_argument("--batch_size", type=int, default=512,
                        help="Batch size")
    parser.add_argument("--patience", type=int, default=30,
                        help="Early stopping patience")

    # Optimizer arguments
    parser.add_argument("--optimizers", type=str, nargs="+",
                        default=["sgd", "adam", "belavkin", "adaptive_belavkin"],
                        help="Optimizers to compare")
    parser.add_argument("--learning_rates", type=float, nargs="+",
                        default=[1e-4, 3e-4, 1e-3, 3e-3],
                        help="Learning rates to search")
    parser.add_argument("--gammas", type=float, nargs="+",
                        default=[1e-5, 1e-4, 1e-3],
                        help="Gamma values for Belavkin")
    parser.add_argument("--betas", type=float, nargs="+",
                        default=[1e-3, 1e-2],
                        help="Beta values for Belavkin")

    # Experiment arguments
    parser.add_argument("--n_seeds", type=int, default=3,
                        help="Number of random seeds")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base random seed for dataset")
    parser.add_argument("--save_dir", type=str,
                        default="experiments/track1_optimizer/benchmarks",
                        help="Directory to save results")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="Number of data loader workers")

    args = parser.parse_args()
    main(args)

"""
Run ablation study for Belavkin optimizer.

Tests contribution of each component:
- Gradient-dependent damping
- Multiplicative vs additive noise
- Adaptive mechanisms

Usage:
    python run_ablation_study.py --task modular --p 97
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
import argparse

from belavkin_ml.datasets.synthetic import ModularArithmeticDataset, create_dataloaders
from belavkin_ml.experiments.ablation import AblationStudy, AblationConfig


class SimpleMLP(nn.Module):
    """Simple MLP for ablation study."""

    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


def main(args):
    print("="*70)
    print("Belavkin Optimizer Ablation Study")
    print("="*70)

    # Set device
    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    print(f"Device: {device}")

    # Create dataset
    if args.task == 'modular':
        print(f"\nDataset: Modular Arithmetic (p={args.p})")
        dataset = ModularArithmeticDataset(
            p=args.p,
            operation='addition',
            train_fraction=0.5,
            seed=args.seed,
        )

        info = dataset.get_info()
        print(f"  Input dim: {info['input_dim']}")
        print(f"  Output dim: {info['output_dim']}")
        print(f"  Train examples: {info['train_examples']}")
        print(f"  Test examples: {info['test_examples']}")

        # Create dataloaders
        train_loader, test_loader = create_dataloaders(
            dataset, batch_size=512, num_workers=0
        )

        # Model factory
        def model_fn():
            return SimpleMLP(
                input_dim=info['input_dim'],
                output_dim=info['output_dim'],
                hidden_dim=128
            )

    else:
        raise ValueError(f"Unknown task: {args.task}")

    # Configure ablation study
    components_to_ablate = [
        'no_damping',           # Remove damping term
        'no_exploration',       # Remove stochastic term
        'additive_noise',       # Additive instead of multiplicative noise
        'no_adaptation',        # Disable all adaptation
        'no_adaptive_gamma',    # Disable gamma adaptation
        'no_adaptive_beta',     # Disable beta adaptation
        'only_damping',         # Only damping, no exploration
        'only_exploration',     # Only exploration, no damping
    ]

    config = AblationConfig(
        components_to_ablate=components_to_ablate,
        base_lr=args.lr,
        base_gamma=args.gamma,
        base_beta=args.beta,
        n_epochs=args.n_epochs,
        n_seeds=args.n_seeds,
        save_dir=Path(args.save_dir) / f"{args.task}_p{args.p}",
    )

    print(f"\nAblation Configuration:")
    print(f"  Components: {len(components_to_ablate)}")
    print(f"  Epochs: {config.n_epochs}")
    print(f"  Seeds: {config.n_seeds}")
    print(f"  Total runs: {(len(components_to_ablate) + 2) * config.n_seeds}")

    # Run ablation study
    study = AblationStudy(config)
    results = study.run(model_fn, train_loader, test_loader, device)

    # Analyze results
    analysis = study.analyze_results(results)

    # Print summary
    study.print_summary(analysis)

    # Save results
    study.save_results(results, analysis)

    print("\nAblation study complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ablation study for Belavkin optimizer")

    # Task
    parser.add_argument("--task", type=str, default="modular",
                        choices=["modular"],
                        help="Task to run ablation on")
    parser.add_argument("--p", type=int, default=97, help="Prime modulus for modular arithmetic")

    # Hyperparameters
    parser.add_argument("--lr", type=float, default=1e-3, help="Base learning rate")
    parser.add_argument("--gamma", type=float, default=1e-4, help="Base damping factor")
    parser.add_argument("--beta", type=float, default=1e-2, help="Base exploration factor")

    # Training
    parser.add_argument("--n_epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--n_seeds", type=int, default=3, help="Number of seeds")

    # Other
    parser.add_argument("--seed", type=int, default=42, help="Base random seed for dataset")
    parser.add_argument("--cpu", action="store_true", help="Force CPU usage")
    parser.add_argument("--save_dir", type=str,
                        default="experiments/track1_optimizer/ablations",
                        help="Save directory")

    args = parser.parse_args()
    main(args)

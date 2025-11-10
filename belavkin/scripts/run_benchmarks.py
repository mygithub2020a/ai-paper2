"""
Run comprehensive benchmarks comparing BelOpt with baseline optimizers.

This script runs multiple experiments across different:
- Tasks (add, mul, inv, composition)
- Moduli (97, 1003, etc.)
- Input dimensions (1, 8, 64)
- Optimizers (BelOpt, Adam, SGD, RMSProp)
- Seeds (for statistical significance)
"""

import argparse
import subprocess
import itertools
import json
import os
from pathlib import Path
import pandas as pd


def run_experiment(config, base_cmd='python belavkin/scripts/train_supervised.py'):
    """Run a single experiment."""
    cmd = [base_cmd.split()[0], base_cmd.split()[1]]

    for key, value in config.items():
        if isinstance(value, bool):
            if value:
                cmd.append(f'--{key}')
        else:
            cmd.extend([f'--{key}', str(value)])

    print(f"\nRunning: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return None

    return result.stdout


def parse_results(log_dir, exp_name):
    """Parse results from log file."""
    log_file = os.path.join(log_dir, f"{exp_name}.json")

    if not os.path.exists(log_file):
        return None

    with open(log_file, 'r') as f:
        logs = json.load(f)

    # Extract final metrics
    final = logs[-1]
    best_test_acc = max(log['test_acc'] for log in logs)

    # Time to target (e.g., 90% accuracy)
    target = 90.0
    time_to_target = None
    cumulative_time = 0
    for log in logs:
        cumulative_time += log['epoch_time']
        if log['test_acc'] >= target and time_to_target is None:
            time_to_target = cumulative_time

    return {
        'final_test_acc': final['test_acc'],
        'final_test_loss': final['test_loss'],
        'best_test_acc': best_test_acc,
        'time_to_target': time_to_target,
        'total_time': cumulative_time,
    }


def main(args):
    """Run benchmark suite."""
    # Define experiment grid
    tasks = args.tasks.split(',')
    moduli = [int(m) for m in args.moduli.split(',')]
    input_dims = [int(d) for d in args.input_dims.split(',')]
    optimizers = args.optimizers.split(',')
    seeds = list(range(args.start_seed, args.start_seed + args.n_seeds))

    # Base configuration
    base_config = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'hidden_dim': args.hidden_dim,
        'model': args.model,
        'n_train': args.n_train,
        'log_dir': args.log_dir,
    }

    # Optimizer-specific settings
    optimizer_configs = {
        'belopt': {'lr': 1e-3, 'gamma0': 1e-3, 'beta0': 0.0},
        'adam': {'lr': 1e-3},
        'sgd': {'lr': 1e-2, 'momentum': 0.9},
        'rmsprop': {'lr': 1e-3},
    }

    # Generate all experiment configurations
    all_configs = []
    for task, modulus, input_dim, optimizer, seed in itertools.product(
        tasks, moduli, input_dims, optimizers, seeds
    ):
        config = base_config.copy()
        config.update({
            'task': task,
            'modulus': modulus,
            'input_dim': input_dim,
            'optimizer': optimizer,
            'seed': seed,
        })
        config.update(optimizer_configs.get(optimizer, {}))

        all_configs.append(config)

    print(f"Total experiments to run: {len(all_configs)}")

    # Run experiments
    results = []
    for i, config in enumerate(all_configs):
        print(f"\n{'='*60}")
        print(f"Experiment {i+1}/{len(all_configs)}")
        print(f"{'='*60}")

        if not args.dry_run:
            run_experiment(config)

            # Parse results
            exp_name = f"{config['task']}_{config['optimizer']}_m{config['modulus']}_d{config['input_dim']}_s{config['seed']}"
            result = parse_results(config['log_dir'], exp_name)

            if result is not None:
                result.update({
                    'task': config['task'],
                    'modulus': config['modulus'],
                    'input_dim': config['input_dim'],
                    'optimizer': config['optimizer'],
                    'seed': config['seed'],
                })
                results.append(result)
        else:
            print(f"Config: {config}")

    if not args.dry_run and results:
        # Save aggregated results
        df = pd.DataFrame(results)
        output_file = os.path.join(args.log_dir, 'benchmark_results.csv')
        df.to_csv(output_file, index=False)
        print(f"\nResults saved to: {output_file}")

        # Print summary
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)

        summary = df.groupby(['task', 'optimizer']).agg({
            'best_test_acc': ['mean', 'std'],
            'time_to_target': ['mean', 'std'],
        }).round(2)

        print(summary)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run benchmark suite')

    # Experiment grid
    parser.add_argument('--tasks', type=str, default='add,mul,inv',
                        help='Comma-separated list of tasks')
    parser.add_argument('--moduli', type=str, default='97,1009',
                        help='Comma-separated list of moduli')
    parser.add_argument('--input_dims', type=str, default='1,8',
                        help='Comma-separated list of input dimensions')
    parser.add_argument('--optimizers', type=str, default='belopt,adam,sgd,rmsprop',
                        help='Comma-separated list of optimizers')
    parser.add_argument('--n_seeds', type=int, default=3,
                        help='Number of random seeds')
    parser.add_argument('--start_seed', type=int, default=42,
                        help='Starting seed value')

    # Shared experiment settings
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs per experiment')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='Hidden dimension')
    parser.add_argument('--model', type=str, default='mlp_medium',
                        help='Model architecture')
    parser.add_argument('--n_train', type=int, default=10000,
                        help='Number of training samples')

    # Other
    parser.add_argument('--log_dir', type=str, default='./results/supervised',
                        help='Log directory')
    parser.add_argument('--dry_run', action='store_true',
                        help='Print configs without running')

    args = parser.parse_args()
    main(args)

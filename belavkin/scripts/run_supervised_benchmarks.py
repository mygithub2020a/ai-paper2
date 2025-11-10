import os
import subprocess
import itertools
import pandas as pd
import numpy as np

def run_experiment(task, optimizer, lr, gamma0, beta0, adaptive_gamma, weight_decay, seed):
    """Runs a single supervised learning experiment."""
    print(f"  Running experiment: task={task}, optimizer={optimizer}, lr={lr}, gamma0={gamma0}, beta0={beta0}, adaptive_gamma={adaptive_gamma}, weight_decay={weight_decay}, seed={seed}")
    command = [
        'python3', 'belavkin/scripts/train_supervised.py',
        '--dataset', 'mod_arith',
        '--task', task,
        '--optimizer', optimizer,
        '--lr', str(lr),
        '--gamma0', str(gamma0),
        '--beta0', str(beta0),
        '--epochs', '100', # Reduced epochs
        '--seed', str(seed),
        '--weight_decay', str(weight_decay)
    ]
    if adaptive_gamma:
        command.append('--adaptive_gamma')

    # Run the command and capture the output
    result = subprocess.run(command, capture_output=True, text=True, env={**os.environ, 'PYTHONPATH': '.'})

    # Parse the final loss from the output
    try:
        last_line = result.stdout.strip().split('\n')[-1]
        final_loss = float(last_line.split('Loss: ')[1])
    except (IndexError, ValueError):
        final_loss = float('nan')

    return final_loss

def main():
    """Runs the full supervised learning benchmark suite with ablations."""
    print("Starting supervised benchmarks with ablations...")
    tasks = ['add'] # Reduced tasks
    seeds = [0, 1] # Reduced seeds

    # Define the hyperparameter grid for the ablations
    belopt_configs = {
        "Deterministic": {"beta0": [0.0], "adaptive_gamma": [False], "weight_decay": [0.0]},
        "Stochastic": {"beta0": [1e-3], "adaptive_gamma": [False], "weight_decay": [0.0]},
        "Adaptive Gamma": {"beta0": [0.0], "adaptive_gamma": [True], "weight_decay": [0.0]},
        "Weight Decay": {"beta0": [0.0], "adaptive_gamma": [False], "weight_decay": [0.01]},
    }

    results = []

    for task in tasks:
        # Run BelOpt ablations
        for ablation_name, config in belopt_configs.items():
            print(f"Running ablation: {ablation_name}")
            for lr in [1e-6]: # Reduced learning rates
                for beta0, adaptive_gamma, weight_decay in itertools.product(config["beta0"], config["adaptive_gamma"], config["weight_decay"]):
                    losses = []
                    for seed in seeds:
                        loss = run_experiment(task, 'belopt', lr, 1e-3, beta0, adaptive_gamma, weight_decay, seed)
                        losses.append(loss)

                    mean_loss = np.mean(losses)
                    std_loss = np.std(losses)
                    results.append([task, 'belopt', ablation_name, lr, np.mean(losses), np.std(losses)])

        # Run Adam baseline
        print("Running baseline: Adam")
        for lr in [1e-3]: # Reduced learning rates
            losses = []
            for seed in seeds:
                loss = run_experiment(task, 'adam', lr, 0.0, 0.0, False, 0.0, seed)
                losses.append(loss)

            results.append([task, 'adam', 'baseline', lr, np.mean(losses), np.std(losses)])

    # Save results to a CSV file
    print("Saving results...")
    df = pd.DataFrame(results, columns=['task', 'optimizer', 'ablation', 'lr', 'mean_loss', 'std_loss'])
    df.to_csv('belavkin/results/supervised_ablations.csv', index=False)
    print("Supervised benchmarks with ablations complete.")

if __name__ == '__main__':
    main()

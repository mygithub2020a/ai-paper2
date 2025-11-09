import os
import subprocess
import itertools
import pandas as pd

def run_experiment(task, optimizer, lr, gamma0, beta0):
    """Runs a single supervised learning experiment."""
    command = [
        'python3', 'belavkin/scripts/train_supervised.py',
        '--dataset', 'mod_arith',
        '--task', task,
        '--optimizer', optimizer,
        '--lr', str(lr),
        '--gamma0', str(gamma0),
        '--beta0', str(beta0),
        '--epochs', '100' # Reduced epochs
    ]

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
    """Runs a reduced supervised learning benchmark suite."""
    print("Starting reduced supervised benchmarks...")
    tasks = ['add'] # Reduced tasks
    optimizers = ['belopt', 'adam'] # Reduced optimizers
    learning_rates = [1e-3, 1e-6] # Reduced learning rates
    gamma0_values = [0.0, 1e-3] # Reduced gamma values
    beta0_values = [0.0, 1e-3] # Reduced beta values

    results = []

    for task in tasks:
        for optimizer in optimizers:
            if optimizer == 'belopt':
                # Hyperparameter sweep for BelOpt
                for lr, gamma0, beta0 in itertools.product(learning_rates, gamma0_values, beta0_values):
                    print(f"Running: task={task}, optimizer={optimizer}, lr={lr}, gamma0={gamma0}, beta0={beta0}")
                    final_loss = run_experiment(task, optimizer, lr, gamma0, beta0)
                    results.append([task, optimizer, lr, gamma0, beta0, final_loss])
            else:
                # Hyperparameter sweep for baselines
                for lr in learning_rates:
                    print(f"Running: task={task}, optimizer={optimizer}, lr={lr}")
                    final_loss = run_experiment(task, optimizer, lr, 0.0, 0.0)
                    results.append([task, optimizer, lr, 0.0, 0.0, final_loss])

    # Save results to a CSV file
    print("Saving results to CSV...")
    df = pd.DataFrame(results, columns=['task', 'optimizer', 'lr', 'gamma0', 'beta0', 'final_loss'])
    df.to_csv('belavkin/results/supervised_benchmarks.csv', index=False)
    print("Supervised benchmarks complete. Results saved to belavkin/results/supervised_benchmarks.csv")

if __name__ == '__main__':
    main()

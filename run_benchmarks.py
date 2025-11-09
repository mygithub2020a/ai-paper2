import torch
import torch.optim as optim
import pandas as pd
import itertools
import time
from torch.utils.data import DataLoader

from belavkin_optimizer.optimizer.optimizer import BelavkinOptimizer
from belavkin_optimizer.benchmarks.modular_arithmetic import ModularArithmeticModel, ModularArithmeticDataset, train, evaluate

def run_benchmarks():
    results = []

    # Define the parameter grid
    n_values = [10, 20]
    op = 'add'
    epochs = 5
    batch_size = 32

    optimizers = {
        'Belavkin': BelavkinOptimizer,
        'Adam': optim.Adam,
        'SGD': optim.SGD,
        'RMSprop': optim.RMSprop
    }

    param_grid = {
        'Belavkin': {'lr': [1e-2, 1e-3], 'gamma': [1e-3, 1e-4], 'beta': [1e-3, 1e-4]},
        'Adam': {'lr': [1e-2, 1e-3]},
        'SGD': {'lr': [1e-2, 1e-3], 'momentum': [0.9]},
        'RMSprop': {'lr': [1e-2, 1e-3]}
    }

    for n in n_values:
        for optimizer_name, optimizer_class in optimizers.items():

            # Get the hyperparameters for the current optimizer
            params = param_grid[optimizer_name]
            keys, values = zip(*params.items())

            for v in itertools.product(*values):
                hyperparams = dict(zip(keys, v))

                print(f"Running benchmark: n={n}, optimizer={optimizer_name}, params={hyperparams}")

                # Create model, dataset, and dataloader
                model = ModularArithmeticModel(n)
                dataset = ModularArithmeticDataset(n, op=op)
                dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

                # Initialize optimizer
                optimizer = optimizer_class(model.parameters(), **hyperparams)

                # Train the model
                start_time = time.time()
                final_loss = train(model, optimizer, dataloader, epochs)
                end_time = time.time()
                training_time = end_time - start_time

                # Evaluate the model
                accuracy = evaluate(model, dataloaloader)

                # Store the results
                result = {
                    'n': n,
                    'optimizer': optimizer_name,
                    'op': op,
                    'epochs': epochs,
                    'accuracy': accuracy,
                    'final_loss': loss,
                    'training_time': training_time
                }
                result.update(hyperparams)
                results.append(result)

    # Save the results to a CSV file
    df = pd.DataFrame(results)
    df.to_csv('results/benchmark_results.csv', index=False)
    print("Benchmarking complete. Results saved to results/benchmark_results.csv")

if __name__ == '__main__':
    run_benchmarks()

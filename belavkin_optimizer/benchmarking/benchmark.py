import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import json
from .datasets import ModularArithmeticDataset, ModularCompositionDataset
from .train import SimpleModel, train
from ..optimizer.belavkin_optimizer import BelavkinOptimizer

def run_benchmarks():
    # Define datasets
    datasets = {
        "small_mod_arith": ModularArithmeticDataset(1000, 10),
        "medium_mod_arith": ModularArithmeticDataset(5000, 50),
        "large_mod_arith": ModularArithmeticDataset(10000, 100),
        "small_mod_comp": ModularCompositionDataset(1000, 10, 20),
        "medium_mod_comp": ModularCompositionDataset(5000, 50, 100),
        "large_mod_comp": ModularCompositionDataset(10000, 100, 200),
    }

    # Define models and optimizers
    results = {}

    for name, dataset in datasets.items():
        print(f"--- Benchmarking on {name} ---")
        input_dim = dataset.X.shape[1]
        output_dim = dataset.y.shape[1]

        models = {
            "Adam": SimpleModel(input_dim, output_dim),
            "SGD": SimpleModel(input_dim, output_dim),
            "RMSprop": SimpleModel(input_dim, output_dim),
            "Belavkin": SimpleModel(input_dim, output_dim),
        }

        optimizers = {
            "Adam": optim.Adam(models["Adam"].parameters()),
            "SGD": optim.SGD(models["SGD"].parameters(), lr=0.001),
            "RMSprop": optim.RMSprop(models["RMSprop"].parameters()),
            "Belavkin": BelavkinOptimizer(models["Belavkin"].parameters(), eta=1e-4),
        }

        criterion = nn.MSELoss()
        dataset_results = {}

        plt.figure(figsize=(10, 6))
        for opt_name, optimizer in optimizers.items():
            print(f"Training with {opt_name}...")
            model = models[opt_name]
            loss_history = train(model, optimizer, dataset, num_epochs=50, batch_size=32, criterion=criterion)
            dataset_results[opt_name] = loss_history
            plt.plot(loss_history, label=opt_name)

        plt.title(f"Loss Curves for {name}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(f"{name}_loss_curves.png")
        plt.close()

        results[name] = dataset_results

    return results

if __name__ == "__main__":
    benchmark_results = run_benchmarks()

    with open("benchmark_results.json", "w") as f:
        json.dump(benchmark_results, f, indent=4)

    for dataset_name, dataset_results in benchmark_results.items():
        print(f"\n--- Results for {dataset_name} ---")
        for opt_name, loss_history in dataset_results.items():
            print(f"{opt_name}: Final Loss = {loss_history[-1]:.4f}")

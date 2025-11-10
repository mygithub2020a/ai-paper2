"""
Fast benchmark suite for actual experiments with smaller configs.
Runs real experiments with PyTorch to generate actual results.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Any
import time
import numpy as np
from optimizer import create_optimizer
from datasets import create_dataset, SimpleNNModel, DeepNNModel


def run_single_experiment(
    optimizer_name: str,
    optimizer_kwargs: Dict[str, Any],
    dataset_name: str,
    dataset_kwargs: Dict[str, Any],
    model_class,
    model_kwargs: Dict[str, Any],
    num_epochs: int = 50,
    batch_size: int = 32,
) -> Dict[str, Any]:
    """Run a single training experiment and return results."""

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create instances
    dataset = create_dataset(dataset_name, **dataset_kwargs)
    model = model_class(**model_kwargs).to(device)
    optimizer = create_optimizer(
        model.parameters(),
        optimizer_name,
        **optimizer_kwargs,
    )
    loss_fn = nn.MSELoss()

    # Get data
    X, y = dataset.get_full_data()
    X = X.to(device)
    y = y.to(device)

    losses = []
    start_time = time.time()

    for epoch in range(num_epochs):
        # Shuffle and batch
        indices = torch.randperm(len(X))
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        epoch_loss = 0
        num_batches = 0

        for i in range(0, len(X), batch_size):
            batch_X = X_shuffled[i:i+batch_size]
            batch_y = y_shuffled[i:i+batch_size]

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = loss_fn(outputs, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        losses.append(avg_loss)

        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}")

    total_time = time.time() - start_time

    return {
        "optimizer": optimizer_name,
        "dataset": dataset_name,
        "losses": losses,
        "final_loss": losses[-1],
        "min_loss": min(losses),
        "total_time": total_time,
    }


def main():
    """Run actual benchmark experiments."""

    print("=" * 80)
    print("BELAVKIN OPTIMIZER - ACTUAL BENCHMARK EXPERIMENTS")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    # Minimal configurations for speed
    optimizers = {
        "belavkin": {"lr": 0.01, "gamma": 0.1, "beta": 0.01},
        "adam": {"lr": 0.01},
        "sgd": {"lr": 0.01},
    }

    datasets = {
        "modular_arithmetic_small": {"modulus": 113, "num_samples": 500, "seed": 42},
        "modular_composition_small": {"modulus": 113, "num_samples": 500, "seed": 42},
    }

    model_specs = {
        "simple": (SimpleNNModel, {}),
    }

    all_results = []

    print("\n" + "=" * 80)
    print("Running Experiments...")
    print("=" * 80)

    experiment_count = 0
    total_experiments = len(optimizers) * len(datasets) * len(model_specs)

    start_total = time.time()

    for dataset_name, dataset_kwargs in datasets.items():
        print(f"\n[Dataset: {dataset_name}]")

        # Determine input dim
        input_dim = 3 if "composition" in dataset_name else 2

        for model_name, (model_class, model_base_kwargs) in model_specs.items():
            print(f"  [Model: {model_name}]")

            model_kwargs = {
                "input_dim": input_dim,
                "output_dim": 1,
                **model_base_kwargs,
            }

            for opt_name, opt_kwargs in optimizers.items():
                experiment_count += 1
                print(f"    [{opt_name}] ({experiment_count}/{total_experiments})", end=" ... ")

                result = run_single_experiment(
                    optimizer_name=opt_name,
                    optimizer_kwargs=opt_kwargs,
                    dataset_name=dataset_name,
                    dataset_kwargs=dataset_kwargs,
                    model_class=model_class,
                    model_kwargs=model_kwargs,
                    num_epochs=50,
                    batch_size=32,
                )

                all_results.append(result)
                print(f"Loss: {result['final_loss']:.6f}, Time: {result['total_time']:.1f}s")

    total_time = time.time() - start_total

    # Print summary
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    for dataset_name in datasets.keys():
        print(f"\n{dataset_name}:")
        print(f"  {'Optimizer':<15} {'Final Loss':<15} {'Min Loss':<15} {'Time (s)':<10}")
        print(f"  {'-'*55}")

        dataset_results = [r for r in all_results if r["dataset"] == dataset_name]

        for opt_name in optimizers.keys():
            opt_results = [r for r in dataset_results if r["optimizer"] == opt_name]
            if opt_results:
                r = opt_results[0]
                print(f"  {opt_name:<15} {r['final_loss']:<15.6f} {r['min_loss']:<15.6f} {r['total_time']:<10.2f}")

    print("\n" + "=" * 80)
    print(f"Total execution time: {total_time:.2f} seconds")
    print(f"Experiments completed: {experiment_count}")

    # Save results
    import pickle
    results_dir = Path("results_actual")
    results_dir.mkdir(exist_ok=True)

    with open(results_dir / "actual_results.pkl", "wb") as f:
        pickle.dump(all_results, f)

    print(f"Results saved to {results_dir}/actual_results.pkl")

    return all_results


if __name__ == "__main__":
    from pathlib import Path
    main()

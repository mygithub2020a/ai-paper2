"""
Main script to run complete benchmark suite and generate results.
"""

import torch
import time
import json
from pathlib import Path
from benchmarks import BenchmarkRunner, run_ablation_study
from analysis import generate_analysis_report
from datasets import SimpleNNModel, DeepNNModel


def main():
    """Run the complete benchmarking suite."""
    print("=" * 80)
    print("BELAVKIN OPTIMIZER - COMPREHENSIVE BENCHMARK SUITE")
    print("=" * 80)

    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    # Create results directory
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    # Initialize benchmark runner
    runner = BenchmarkRunner(device=device, verbose=True)

    # Define optimizer configurations
    optimizers = {
        "belavkin": {
            "lr": 0.01,
            "gamma": 0.1,
            "beta": 0.01,
            "momentum": 0.0,
        },
        "adaptive_belavkin": {
            "lr": 0.01,
            "gamma": 0.1,
            "beta": 0.01,
            "momentum": 0.0,
            "adaptive_gamma": True,
            "adaptive_beta": True,
        },
        "adam": {
            "lr": 0.01,
            "betas": (0.9, 0.999),
        },
        "sgd": {
            "lr": 0.01,
            "momentum": 0.9,
        },
        "rmsprop": {
            "lr": 0.01,
            "alpha": 0.99,
        },
    }

    # Define datasets with varying difficulty
    datasets = {
        "modular_arithmetic_small": {
            "modulus": 113,
            "num_samples": 500,
            "seed": 42,
        },
        "modular_arithmetic_medium": {
            "modulus": 113,
            "num_samples": 2000,
            "seed": 42,
        },
        "modular_arithmetic_large": {
            "modulus": 113,
            "num_samples": 5000,
            "seed": 42,
        },
        "modular_composition_small": {
            "modulus": 113,
            "num_samples": 500,
            "seed": 42,
        },
        "modular_composition_medium": {
            "modulus": 113,
            "num_samples": 2000,
            "seed": 42,
        },
    }

    # Define model specifications
    model_specs = {
        "simple": (SimpleNNModel, {}),
        "deep": (DeepNNModel, {}),
    }

    # Run main benchmark suite
    print("\n" + "=" * 80)
    print("Running Main Benchmark Suite...")
    print("=" * 80)

    start_time = time.time()

    results = runner.run_suite(
        optimizers=optimizers,
        datasets=datasets,
        model_specs=model_specs,
        num_epochs=100,
        batch_size=32,
        num_runs=3,
    )

    main_benchmark_time = time.time() - start_time
    print(f"\nMain benchmark suite completed in {main_benchmark_time:.2f} seconds")

    # Print summary
    runner.print_summary(results)

    # Save results
    import pickle
    with open(results_dir / "main_results.pkl", "wb") as f:
        pickle.dump(results, f)
    print(f"\nMain results saved to {results_dir}/main_results.pkl")

    # Run ablation study on core Belavkin Optimizer
    print("\n" + "=" * 80)
    print("Running Ablation Study...")
    print("=" * 80)

    ablation_start = time.time()

    base_config = {
        "lr": 0.01,
        "gamma": 0.1,
        "beta": 0.01,
        "momentum": 0.0,
    }

    param_variations = {
        "gamma": [0.01, 0.05, 0.1, 0.2, 0.5],
        "beta": [0.001, 0.005, 0.01, 0.05, 0.1],
        "lr": [0.001, 0.005, 0.01, 0.05, 0.1],
    }

    ablation_results = run_ablation_study(
        base_config=base_config,
        param_variations=param_variations,
        num_epochs=100,
        batch_size=32,
    )

    ablation_time = time.time() - ablation_start
    print(f"\nAblation study completed in {ablation_time:.2f} seconds")

    # Save ablation results
    with open(results_dir / "ablation_results.pkl", "wb") as f:
        pickle.dump(ablation_results, f)
    print(f"Ablation results saved to {results_dir}/ablation_results.pkl")

    # Generate analysis plots and tables
    print("\n" + "=" * 80)
    print("Generating Analysis Reports...")
    print("=" * 80)

    analyzer = generate_analysis_report(results, ablation_results, output_dir="results")

    # Generate summary statistics
    summary_stats = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": device,
        "num_optimizers": len(optimizers),
        "num_datasets": len(datasets),
        "num_models": len(model_specs),
        "main_benchmark_time_seconds": main_benchmark_time,
        "ablation_study_time_seconds": ablation_time,
        "total_time_seconds": main_benchmark_time + ablation_time,
    }

    with open(results_dir / "summary_stats.json", "w") as f:
        json.dump(summary_stats, f, indent=2)

    print(f"\nSummary statistics saved to {results_dir}/summary_stats.json")

    print("\n" + "=" * 80)
    print("BENCHMARK SUITE COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(f"\nTotal execution time: {main_benchmark_time + ablation_time:.2f} seconds")
    print(f"Results saved to: {results_dir.absolute()}/")


if __name__ == "__main__":
    main()

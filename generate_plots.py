"""
Script to generate visualization plots from benchmark results.
"""

import pickle
from pathlib import Path
from analysis import BenchmarkAnalyzer


def main():
    """Generate all plots and analysis."""
    results_dir = Path("results")

    print("Loading benchmark results...")
    with open(results_dir / "main_results.pkl", "rb") as f:
        results = pickle.load(f)

    with open(results_dir / "ablation_results.pkl", "rb") as f:
        ablation_results = pickle.load(f)

    print("Generating analysis visualizations...")
    analyzer = BenchmarkAnalyzer(output_dir="results")

    # Generate all plots
    analyzer.plot_loss_curves(results)
    analyzer.plot_final_loss_comparison(results)
    analyzer.plot_convergence_speed(results)
    analyzer.create_results_table(results)
    analyzer.plot_ablation_study(ablation_results)

    print("\nVisualization generation complete!")
    print(f"Plots saved to: {results_dir.absolute()}/")

    # List all generated files
    print("\nGenerated files:")
    for f in sorted(results_dir.glob("*")):
        if f.is_file():
            size = f.stat().st_size
            print(f"  {f.name} ({size:,} bytes)")


if __name__ == "__main__":
    main()

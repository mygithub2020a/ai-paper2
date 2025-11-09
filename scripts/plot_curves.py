import pandas as pd
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

def plot_curves(results_file, metric='accuracy'):
    """Plots the learning curves for a given metric."""
    df = pd.read_csv(results_file)

    # Get the experiment name from the file name
    exp_name = Path(results_file).stem.replace('_results', '')

    # Create a directory for the figures if it doesn't exist
    figs_dir = Path("paper/figs")
    figs_dir.mkdir(exist_ok=True)

    plt.figure(figsize=(10, 6))

    optimizers = df['optimizer'].unique()

    for optimizer in optimizers:
        opt_df = df[df['optimizer'] == optimizer]

        # Group by epoch and calculate the mean and std of the metric
        mean_metric = opt_df.groupby('epoch')[metric].mean()
        std_metric = opt_df.groupby('epoch')[metric].std()

        epochs = mean_metric.index

        plt.plot(epochs, mean_metric, label=optimizer)
        plt.fill_between(epochs, mean_metric - std_metric, mean_metric + std_metric, alpha=0.2)

    plt.title(f"{metric.capitalize()} vs. Epoch for {exp_name}")
    plt.xlabel("Epoch")
    plt.ylabel(metric.capitalize())
    plt.legend()
    plt.grid(True)

    # Save the plot
    plot_path = figs_dir / f"{exp_name}_{metric}_curves.png"
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Plot learning curves from benchmark results.")
    parser.add_argument('results_files', nargs='+', type=str, help='Path to the CSV results file(s).')
    args = parser.parse_args()

    for results_file in args.results_files:
        plot_curves(results_file, 'accuracy')
        plot_curves(results_file, 'loss')

if __name__ == '__main__':
    main()

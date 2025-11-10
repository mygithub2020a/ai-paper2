import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import argparse
import ast

def plot_curves(results_df, output_dir, file_prefix):
    """
    Plots learning curves for loss and accuracy.
    """
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(12, 6))
    sns.lineplot(data=results_df, x='epoch', y='avg_loss', hue='optimizer_name', style='data_task', errorbar='sd')
    plt.title('Average Loss vs. Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f"{file_prefix}_loss_curves.png"))
    plt.close()

    plt.figure(figsize=(12, 6))
    sns.lineplot(data=results_df, x='epoch', y='accuracy', hue='optimizer_name', style='data_task', errorbar='sd')
    plt.title('Accuracy vs. Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f"{file_prefix}_accuracy_curves.png"))
    plt.close()

def plot_ablation_curves(results_df, output_dir):
    """
    Plots learning curves for the ablation studies.
    """
    os.makedirs(output_dir, exist_ok=True)

    results_df['beta_value'] = results_df['optimizer_params'].apply(lambda x: ast.literal_eval(x)['beta_scheduler']['params']['initial_value'])
    results_df['gamma_schedule'] = results_df['optimizer_params'].apply(lambda x: ast.literal_eval(x)['gamma_scheduler']['name'])

    plt.figure(figsize=(12, 6))
    sns.lineplot(data=results_df, x='epoch', y='avg_loss', hue='beta_value', style='gamma_schedule', errorbar='sd')
    plt.title('Ablation Study: Average Loss vs. Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "ablations_loss_curves.png"))
    plt.close()

    plt.figure(figsize=(12, 6))
    sns.lineplot(data=results_df, x='epoch', y='accuracy', hue='beta_value', style='gamma_schedule', errorbar='sd')
    plt.title('Ablation Study: Accuracy vs. Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "ablations_accuracy_curves.png"))
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Plotting script for BelOpt benchmarks.")
    parser.add_argument('--results-dir', type=str, default='results', help="Directory containing the results CSV files.")
    parser.add_argument('--output-dir', type=str, default='paper/figs', help="Directory to save the plots.")
    args = parser.parse_args()

    all_files = os.listdir(args.results_dir)

    benchmark_files = [f for f in all_files if f.startswith('supervised_benchmarks')]
    ablation_files = [f for f in all_files if f.startswith('supervised_ablations')]

    if benchmark_files:
        latest_benchmark_file = sorted(benchmark_files)[-1]
        print(f"Plotting benchmarks from: {latest_benchmark_file}")
        benchmark_df = pd.read_csv(os.path.join(args.results_dir, latest_benchmark_file))
        plot_curves(benchmark_df, args.output_dir, 'benchmarks')

    if ablation_files:
        latest_ablation_file = sorted(ablation_files)[-1]
        print(f"Plotting ablations from: {latest_ablation_file}")
        ablation_df = pd.read_csv(os.path.join(args.results_dir, latest_ablation_file))
        plot_ablation_curves(ablation_df, args.output_dir)

    print(f"Plots saved to {args.output_dir}")

if __name__ == '__main__':
    main()

import os
import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt

def analyze_results(results_dir='results'):
    """
    Analyzes the benchmark results from the CSV files in the results directory.
    """
    results = []
    for filename in os.listdir(results_dir):
        if filename.endswith('.csv'):
            # Correctly parse filenames which may have variable parts
            parts = filename.replace('.csv', '').split('_')

            # The dataset can be one or two words
            if parts[1] in ['arithmetic', 'composition', 'parity']:
                dataset = '_'.join(parts[0:2])
                parts = parts[2:]
            else:
                dataset = parts.pop(0)

            optimizer_parts = []
            while parts and not parts[0].replace('.', '', 1).isdigit():
                optimizer_parts.append(parts.pop(0))
            optimizer = '_'.join(optimizer_parts)

            lr = float(parts.pop(0))
            seed = int(parts.pop(0))

            # Reconstruct a simplified optimizer name for grouping
            if 'belavkin' in optimizer:
                if 'adaptive_gamma' in filename and 'adaptive_beta' in filename:
                    optimizer_name = 'belavkin_adaptive_both'
                elif 'adaptive_gamma' in filename:
                    optimizer_name = 'belavkin_adaptive_gamma'
                elif 'adaptive_beta' in filename:
                    optimizer_name = 'belavkin_adaptive_beta'
                else:
                    optimizer_name = 'belavkin_default'
            else:
                optimizer_name = optimizer


            df = pd.read_csv(os.path.join(results_dir, filename))
            final_test_acc = df['test_acc'].iloc[-1]
            avg_time_per_epoch = df['time'].mean()

            results.append({
                'dataset': dataset,
                'optimizer': optimizer_name,
                'lr': lr,
                'seed': seed,
                'final_test_acc': final_test_acc,
                'avg_time_per_epoch': avg_time_per_epoch
            })

    if not results:
        print("No results found to analyze.")
        return

    df_results = pd.DataFrame(results)

    # --- Create Summary Table ---
    summary = df_results.pivot_table(index='optimizer', columns='dataset', values='final_test_acc', aggfunc='mean')
    print("--- Final Test Accuracy (%) ---")
    print(tabulate(summary, headers='keys', tablefmt='psql'))

    # --- Create Plots ---
    for dataset in df_results['dataset'].unique():
        plt.figure(figsize=(10, 6))

        for optimizer in df_results['optimizer'].unique():
            # Filter the DataFrame for the current dataset and optimizer
            subset = df_results[(df_results['dataset'] == dataset) & (df_results['optimizer'] == optimizer)]
            if not subset.empty:
                lr = subset['lr'].iloc[0]
                seed = subset['seed'].iloc[0]

                # Reconstruct the original optimizer name for the filename
                original_optimizer = optimizer.split('_')[0] if 'belavkin' in optimizer else optimizer

                # Find the correct filename, accounting for belavkin variants
                found_file = False
                for f in os.listdir(results_dir):
                    if f.startswith(f"{dataset}_{original_optimizer}_{lr}_{seed}"):
                        filepath = os.path.join(results_dir, f)
                        df_run = pd.read_csv(filepath)
                        plt.plot(df_run['epoch'], df_run['test_acc'], label=optimizer)
                        found_file = True
                        break

                if not found_file:
                    print(f"Warning: Could not find results file for {dataset}, {optimizer}")

        plt.title(f'Test Accuracy vs. Epochs on {dataset}')
        plt.xlabel('Epoch')
        plt.ylabel('Test Accuracy (%)')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'results/{dataset}_accuracy_curves.png')
        plt.close()

if __name__ == '__main__':
    analyze_results()

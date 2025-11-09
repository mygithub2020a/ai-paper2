import pandas as pd
import matplotlib.pyplot as plt
import os

def analyze_and_plot(benchmark_name, filepath):
    """
    Analyzes the benchmark data and generates plots and tables.
    """
    df = pd.read_csv(filepath)

    # --- Plotting ---
    plt.figure(figsize=(10, 6))
    for column in df.columns:
        plt.plot(df[column], label=column)
    plt.title(f'{benchmark_name} - Loss Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Create paper directory if it doesn't exist
    if not os.path.exists('paper'):
        os.makedirs('paper')

    plot_path = f'paper/{benchmark_name}_loss_curves.png'
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved loss curve plot to {plot_path}")

    # --- Performance Table ---
    summary = df.agg(['min', 'mean', 'std']).transpose()
    summary.columns = ['Min Loss', 'Mean Loss', 'Std Dev']
    table_path = f'paper/{benchmark_name}_performance_table.md'
    summary.to_markdown(table_path)
    print(f"Saved performance table to {table_path}")


def main():
    analyze_and_plot('Modular_Arithmetic', 'data/modular_arithmetic_benchmark.csv')
    analyze_and_plot('Modular_Composition', 'data/modular_composition_benchmark.csv')

if __name__ == '__main__':
    main()

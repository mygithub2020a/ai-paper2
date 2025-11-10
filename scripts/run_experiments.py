import yaml
import argparse
import pandas as pd
import os
from datetime import datetime
from itertools import product
from tqdm import tqdm
import copy

# Add the project root to the python path
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.train_harness import run_single_experiment

def generate_experiment_configs(base_config):
    """
    Generates a list of specific experiment configurations from a base config
    that may contain lists of options for certain parameters.
    """
    grid_params = base_config.pop('grid', {})

    keys, values = zip(*grid_params.items())

    experiment_configs = []

    for combo_values in product(*values):
        combo = dict(zip(keys, combo_values))

        exp_config = copy.deepcopy(base_config)

        # Update the config with the current combination
        for key_str, value in combo.items():
            keys_list = key_str.split('.')
            d = exp_config
            for k in keys_list[:-1]:
                d = d[k]
            d[keys_list[-1]] = value

        # Update the experiment name
        combo_name = "_".join([f"{k.split('.')[-1]}-{v}" for k, v in combo.items()])
        # Use a new 'name' key for the specific experiment instance
        exp_config['experiment']['name'] = f"{base_config['experiment']['base_name']}_{combo_name}"

        experiment_configs.append(exp_config)

    return experiment_configs


def main():
    parser = argparse.ArgumentParser(description="Experiment runner for BelOpt benchmarks.")
    parser.add_argument('--config', type=str, required=True,
                        help="Path to the main experiment config YAML file.")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        base_config = yaml.safe_load(f)

    experiment_configs = generate_experiment_configs(base_config)

    all_results = []

    print(f"--- Generated {len(experiment_configs)} experiments ---")

    for config in tqdm(experiment_configs, desc="Running experiments"):
        print(f"\n--- Running experiment: {config['experiment']['name']} ---")
        results_df = run_single_experiment(config)
        all_results.append(results_df)

    final_results = pd.concat(all_results)

    # Save results
    output_dir = base_config['experiment'].get('output_dir', 'results')
    os.makedirs(output_dir, exist_ok=True)

    base_name = base_config['experiment']['base_name']
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{base_name}_{timestamp}.csv"

    final_results.to_csv(os.path.join(output_dir, filename), index=False)
    print(f"\nAll experiments complete. Results saved to {os.path.join(output_dir, filename)}")


if __name__ == '__main__':
    main()

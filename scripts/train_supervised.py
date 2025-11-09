import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import argparse
import time
import pandas as pd
from pathlib import Path

from belopt.optim import BelOpt
from belopt.models import SimpleMLP
from data.mod_arith import get_dataloader
from belopt.schedules import CosineAnnealing

def get_optimizer(optimizer_name, model_params, optimizer_params):
    """Factory function to create an optimizer."""
    if optimizer_name.lower() == 'belopt':
        return BelOpt(model_params, **optimizer_params)
    elif optimizer_name.lower() == 'adam':
        return optim.Adam(model_params, **optimizer_params)
    elif optimizer_name.lower() == 'sgd':
        return optim.SGD(model_params, **optimizer_params)
    elif optimizer_name.lower() == 'rmsprop':
        return optim.RMSprop(model_params, **optimizer_params)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

def train(config):
    """Main training loop driven by the config."""
    results = []

    # Create results directory if it doesn't exist
    Path("results").mkdir(exist_ok=True)

    # Experiment settings
    exp_name = config['name']
    task_params = config['task']
    model_params = config['model']
    training_params = config['training']
    seeds = config['seeds']
    optimizers_config = config['optimizers']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for seed in seeds:
        torch.manual_seed(seed)

        for opt_config in optimizers_config:
            optimizer_name = opt_config['name']
            optimizer_params = opt_config.get('params', {})
            print(f"--- Running: Seed {seed}, Optimizer {optimizer_name} ---")

            # --- Data ---
            if task_params['op'] in ['add', 'multiply', 'power']:
                num_inputs = 2
            elif task_params['op'] == 'invert':
                num_inputs = 1
            else:
                raise ValueError(f"Unknown operation: {task_params['op']}")

            input_dim = num_inputs * task_params['p']
            train_loader = get_dataloader(p=task_params['p'], op=task_params['op'], batch_size=training_params['batch_size'])

            # --- Model ---
            model = SimpleMLP(input_dim, model_params['hidden_dim'], task_params['p'], model_params['num_layers']).to(device)

            # --- Optimizer ---
            optimizer = get_optimizer(optimizer_name, model.parameters(), optimizer_params)

            # --- Schedulers ---
            schedulers = []
            if 'lr' in optimizer_params:
                schedulers.append(CosineAnnealing(optimizer, 'lr', T_max=training_params['epochs']))
            if optimizer_name.lower() == 'belopt':
                 if 'gamma' in optimizer_params:
                    schedulers.append(CosineAnnealing(optimizer, 'gamma', T_max=training_params['epochs']))
                 if 'beta' in optimizer_params:
                    schedulers.append(CosineAnnealing(optimizer, 'beta', T_max=training_params['epochs']))

            # --- Loss Function ---
            criterion = nn.CrossEntropyLoss()

            # --- Training Loop ---
            start_time = time.time()
            for epoch in range(training_params['epochs']):
                model.train()
                total_loss = 0
                correct = 0
                total = 0

                for x, y in train_loader:
                    x, y = x.to(device), y.to(device)

                    optimizer.zero_grad()

                    x_one_hot = torch.nn.functional.one_hot(x.long(), num_classes=task_params['p']).view(x.size(0), -1).float()

                    outputs = model(x_one_hot)
                    loss = criterion(outputs, y)

                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += y.size(0)
                    correct += (predicted == y).sum().item()

                avg_loss = total_loss / len(train_loader)
                accuracy = 100 * correct / total
                epoch_time = time.time() - start_time

                print(f"Epoch [{epoch+1}/{training_params['epochs']}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

                results.append({
                    'seed': seed,
                    'optimizer': optimizer_name,
                    'task': task_params['op'],
                    'epoch': epoch + 1,
                    'loss': avg_loss,
                    'accuracy': accuracy,
                    'time': epoch_time
                })

                for scheduler in schedulers:
                    scheduler.step()

    # --- Save Results ---
    df = pd.DataFrame(results)
    results_path = f"results/{exp_name}_results.csv"
    df.to_csv(results_path, index=False)
    print(f"Results saved to {results_path}")

def main():
    parser = argparse.ArgumentParser(description="Run supervised learning benchmarks.")
    parser.add_argument('config', type=str, help='Path to the YAML configuration file.')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    for experiment in config['experiments']:
        train(experiment)

if __name__ == '__main__':
    # Add this to make sure PYTHONPATH is set correctly when running from root
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    main()

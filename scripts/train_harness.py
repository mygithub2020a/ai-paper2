import torch
import torch.nn as nn
import pandas as pd
import os
from datetime import datetime
import json

# Add the project root to the python path
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from belopt.optim import BelOpt
from belopt.models import MLP, MLPMixer
from data.mod_arith import (
    modular_addition_data,
    modular_multiplication_data,
    modular_inversion_data,
    modular_power_data
)
from data.mod_comp import modular_composition_data
from belopt.schedules import ConstantScheduler, CosineDecayScheduler, InverseSqrtScheduler


def get_model(config):
    model_name = config['model']['name']
    model_params = config['model']['params']
    if model_name == 'MLP':
        return MLP(**model_params)
    elif model_name == 'MLPMixer':
        return MLPMixer(**model_params)
    else:
        raise ValueError(f"Unknown model: {model_name}")

def get_data(config):
    task_name = config['data']['task']
    task_params = config['data']['params']
    if task_name == 'modular_addition':
        return modular_addition_data(**task_params)
    elif task_name == 'modular_multiplication':
        return modular_multiplication_data(**task_params)
    elif task_name == 'modular_inversion':
        return modular_inversion_data(**task_params)
    elif task_name == 'modular_power':
        return modular_power_data(**task_params)
    elif task_name == 'modular_composition':
        x, y, _ = modular_composition_data(**task_params)
        return x, y
    else:
        raise ValueError(f"Unknown task: {task_name}")

def get_optimizer(model, config):
    optimizer_name = config['optimizer']['name']
    optimizer_params = config['optimizer']['params']

    if optimizer_name == 'BelOpt':
        eta_scheduler_cfg = optimizer_params['eta_scheduler']
        gamma_scheduler_cfg = optimizer_params['gamma_scheduler']
        beta_scheduler_cfg = optimizer_params['beta_scheduler']

        eta_scheduler = get_scheduler(eta_scheduler_cfg)
        gamma_scheduler = get_scheduler(gamma_scheduler_cfg)
        beta_scheduler = get_scheduler(beta_scheduler_cfg)

        return BelOpt(model.parameters(),
                      eta_scheduler=eta_scheduler,
                      gamma_scheduler=gamma_scheduler,
                      beta_scheduler=beta_scheduler,
                      decoupled_weight_decay=optimizer_params.get('decoupled_weight_decay', 0.0),
                      update_clip=optimizer_params.get('update_clip', None))
    else:
        lr = float(optimizer_params.get('lr', 1e-3))
        if optimizer_name == 'Adam':
            return torch.optim.Adam(model.parameters(), lr=lr)
        elif optimizer_name == 'SGD':
            momentum = float(optimizer_params.get('momentum', 0))
            return torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
        elif optimizer_name == 'RMSProp':
            return torch.optim.RMSprop(model.parameters(), lr=lr)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

def get_scheduler(config):
    scheduler_name = config['name']
    params = config.get('params', {})
    initial_value = float(params['initial_value'])

    if scheduler_name == 'ConstantScheduler':
        return ConstantScheduler(initial_value)
    elif scheduler_name == 'CosineDecayScheduler':
        return CosineDecayScheduler(initial_value, params['max_steps'])
    elif scheduler_name == 'InverseSqrtScheduler':
        return InverseSqrtScheduler(initial_value, params.get('warmup_steps', 0))
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")


def train_single_seed(config, seed):
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x, y = get_data(config)
    x = x.float().to(device)
    y = y.long().to(device)
    dataset = torch.utils.data.TensorDataset(x, y)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=config['training']['batch_size'], shuffle=True)

    model = get_model(config).to(device)
    optimizer = get_optimizer(model, config)
    loss_fn = nn.CrossEntropyLoss()

    results = []
    start_time = datetime.now()

    for epoch in range(config['training']['epochs']):
        total_loss = 0
        correct_predictions = 0
        total_samples = 0

        for x_batch, y_batch in dataloader:
            optimizer.zero_grad()
            logits = model(x_batch)
            loss = loss_fn(logits, y_batch)
            loss.backward()
            if 'grad_clip' in config['training'] and config['training']['grad_clip'] is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['grad_clip'])
            optimizer.step()
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct_predictions += (preds == y_batch).sum().item()
            total_samples += y_batch.size(0)

        avg_loss = total_loss / len(dataloader)
        accuracy = correct_predictions / total_samples

        results.append({
            'seed': seed,
            'epoch': epoch + 1,
            'avg_loss': avg_loss,
            'accuracy': accuracy,
            'time_elapsed': (datetime.now() - start_time).total_seconds()
        })

        if (epoch + 1) % 10 == 0 or epoch == config['training']['epochs'] - 1:
            print(f"Seed {seed}, Epoch [{epoch+1}/{config['training']['epochs']}], Loss: {avg_loss:.4f}, Acc: {accuracy:.4f}")

    results_df = pd.DataFrame(results)
    results_df['optimizer_name'] = config['optimizer']['name']
    results_df['data_task'] = config['data']['task']
    results_df['optimizer_params'] = json.dumps(config['optimizer']['params'])
    return results_df

def run_single_experiment(config):
    all_results = []
    for seed in range(config['experiment']['num_seeds']):
        print(f"--- Running seed {seed}/{config['experiment']['num_seeds']-1} ---")
        results_df = train_single_seed(config, seed)
        all_results.append(results_df)

    return pd.concat(all_results)

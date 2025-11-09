import argparse
import json
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from src.data import (create_modular_arithmetic_dataset,
                    create_modular_composition_dataset,
                    create_sparse_parity_dataset)
from src.models import (create_modular_arithmetic_model,
                      create_modular_composition_model,
                      create_sparse_parity_model)
from src.belavkin_optimizer import BelavkinOptimizer

def get_optimizer(optimizer_name, model_params, lr, **kwargs):
    if optimizer_name.lower() == 'sgd':
        return optim.SGD(model_params, lr=lr, momentum=kwargs.get('momentum', 0.9))
    elif optimizer_name.lower() == 'adam':
        return optim.Adam(model_params, lr=lr, betas=kwargs.get('betas', (0.9, 0.999)))
    elif optimizer_name.lower() == 'rmsprop':
        return optim.RMSprop(model_params, lr=lr, alpha=kwargs.get('alpha', 0.99))
    elif optimizer_name.lower() == 'adamw':
        return optim.AdamW(model_params, lr=lr, weight_decay=kwargs.get('weight_decay', 0.01))
    elif optimizer_name.lower() == 'belavkin':
        return BelavkinOptimizer(model_params, lr=lr,
                                 gamma=kwargs.get('gamma', 1e-4),
                                 beta=kwargs.get('beta', 1e-2),
                                 adaptive_gamma=kwargs.get('adaptive_gamma', False),
                                 adaptive_beta=kwargs.get('adaptive_beta', False))
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

def train(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)

        # Squeeze target if it's shape [batch_size, 1]
        if len(target.shape) > 1 and target.shape[1] == 1:
            target = target.squeeze(1)

        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

    return total_loss / len(train_loader), 100. * correct / total

def test(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            # Squeeze target if it's shape [batch_size, 1]
            if len(target.shape) > 1 and target.shape[1] == 1:
                target = target.squeeze(1)

            loss = criterion(output, target)
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    return total_loss / len(test_loader), 100. * correct / total

def main():
    parser = argparse.ArgumentParser(description='Belavkin Optimizer Benchmarks')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['modular_arithmetic', 'modular_composition', 'sparse_parity'])
    parser.add_argument('--optimizer', type=str, required=True,
                        choices=['sgd', 'adam', 'rmsprop', 'adamw', 'belavkin'])
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--optimizer_params', type=json.loads, default='{}',
                        help='JSON string for optimizer specific parameters')
    parser.add_argument('--results_dir', type=str, default='results')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Data Loading ---
    if args.dataset == 'modular_arithmetic':
        dataset = create_modular_arithmetic_dataset()
        model = create_modular_arithmetic_model().to(device)
    elif args.dataset == 'modular_composition':
        dataset = create_modular_composition_dataset()
        model = create_modular_composition_model().to(device)
    elif args.dataset == 'sparse_parity':
        dataset = create_sparse_parity_dataset()
        model = create_sparse_parity_model().to(device)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # --- Optimizer and Loss ---
    optimizer = get_optimizer(args.optimizer, model.parameters(), args.lr, **args.optimizer_params)
    criterion = nn.CrossEntropyLoss()

    # --- Training and Logging ---
    results = []
    output_filename = f"{args.dataset}_{args.optimizer}_{args.lr}_{args.seed}.csv"
    output_path = os.path.join(args.results_dir, output_filename)

    with open(output_path, 'w') as f:
        f.write("epoch,train_loss,train_acc,test_loss,test_acc,time\n")

    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
        test_loss, test_acc = test(model, test_loader, criterion, device)

        epoch_time = time.time() - start_time
        start_time = time.time()

        print(f'Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
              f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')

        with open(output_path, 'a') as f:
            f.write(f"{epoch},{train_loss},{train_acc},{test_loss},{test_acc},{epoch_time}\n")

if __name__ == '__main__':
    main()

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from benchmarks.modular_arithmetic import ModularAdditionDataset, SimpleMLP
from belavkin_optimizer.optimizer import BelavkinOptimizer
import time

def run_benchmark(optimizer_class, optimizer_params, n, num_epochs):
    train_dataset = ModularAdditionDataset(n, train=True)
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    val_dataset = ModularAdditionDataset(n, train=False)
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    model = SimpleMLP(n)
    optimizer = optimizer_class(model.parameters(), **optimizer_params)
    criterion = nn.CrossEntropyLoss()

    start_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        for inputs, targets in train_dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += targets.size(0)
            train_correct += (predicted == targets).sum().item()

        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, targets in val_dataloader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += targets.size(0)
                val_correct += (predicted == targets).sum().item()

        print(f"Epoch {epoch+1} | "
              f"Train Loss: {train_loss/len(train_dataloader):.4f} | "
              f"Train Acc: {100 * train_correct / train_total:.2f}% | "
              f"Val Loss: {val_loss/len(val_dataloader):.4f} | "
              f"Val Acc: {100 * val_correct / val_total:.2f}%")

    end_time = time.time()
    return end_time - start_time

if __name__ == '__main__':
    n = 200
    num_epochs = 10

    optimizers = {
        "Belavkin (eta=0.01, gamma=0.1, beta=0.01, decay=0.1)": (BelavkinOptimizer, {'eta': 0.01, 'gamma': 0.1, 'beta': 0.01, 'gamma_decay': 0.1}),
        "Belavkin (eta=0.1, gamma=0.1, beta=0.01, decay=0.1)": (BelavkinOptimizer, {'eta': 0.1, 'gamma': 0.1, 'beta': 0.01, 'gamma_decay': 0.1}),
        "Belavkin (ablation, no gamma)": (BelavkinOptimizer, {'eta': 0.01, 'gamma': 0.0, 'beta': 0.01, 'gamma_decay': 0.0}),
        "Belavkin (ablation, no beta)": (BelavkinOptimizer, {'eta': 0.01, 'gamma': 0.1, 'beta': 0.0, 'gamma_decay': 0.1}),
        "Adam": (optim.Adam, {'lr': 0.001}),
    }

    results = {}
    for name, (optimizer_class, optimizer_params) in optimizers.items():
        print(f"Running {name} optimizer...")
        time_taken = run_benchmark(optimizer_class, optimizer_params, n, num_epochs)
        results[name] = time_taken
        print(f"{name}: {time_taken:.4f} seconds")
        print("-" * 30)

    print("\n--- Benchmark Summary ---")
    for name, time_taken in results.items():
        print(f"{name}: {time_taken:.4f} seconds")

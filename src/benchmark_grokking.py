import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.modular_arithmetic import ModularArithmeticDataset, TransformerModel
from src.optimizer import BelavkinOptimizer
import time
import numpy as np
import sys

def train(model, optimizer, train_loader, device):
    model.train()
    total_loss = 0
    criterion = nn.CrossEntropyLoss()

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()

        if isinstance(optimizer, BelavkinOptimizer):
            loss_val, gns, panic = optimizer.step()
        else:
            optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)

def evaluate(model, val_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    return 100 * correct / total

def run_benchmark(optimizer_name, epochs=2000, p=113, seed=42):
    print(f"Running benchmark for {optimizer_name} with seed {seed}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Re-seed everything for fairness
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    train_dataset = ModularArithmeticDataset(p=p, train_split=0.5, seed=seed, mode='train')
    val_dataset = ModularArithmeticDataset(p=p, train_split=0.5, seed=seed, mode='val')

    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)

    model = TransformerModel(vocab_size=p).to(device)

    if optimizer_name == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1.0)
    elif optimizer_name == "Belavkin":
        optimizer = BelavkinOptimizer(model.parameters(), lr=1e-3, weight_decay=1.0, adaptive_decay=True)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    start_time = time.time()
    grok_epoch = -1

    for epoch in range(epochs):
        train_loss = train(model, optimizer, train_loader, device)
        val_acc = evaluate(model, val_loader, device)

        # Check every epoch for granularity
        if val_acc >= 99.0 and grok_epoch == -1:
            grok_epoch = epoch
            print(f"GROK! Reached 99% accuracy at epoch {epoch}")
            break

    end_time = time.time()
    duration = end_time - start_time

    final_val_acc = evaluate(model, val_loader, device)
    # If didn't grok, record max epoch
    grok_epoch_res = grok_epoch if grok_epoch != -1 else epochs

    print(f"Seed {seed} Result: Grok Epoch = {grok_epoch}, Time = {duration:.2f}s")

    return grok_epoch_res, duration, final_val_acc

if __name__ == "__main__":
    epochs = 300
    p = 113
    seeds = [42, 123]

    print("--- Multi-Seed Benchmark Start ---")

    adam_res = []
    bel_res = []

    for seed in seeds:
        adam_res.append(run_benchmark("AdamW", epochs=epochs, p=p, seed=seed))
        bel_res.append(run_benchmark("Belavkin", epochs=epochs, p=p, seed=seed))

    adam_epochs = [r[0] for r in adam_res]
    bel_epochs = [r[0] for r in bel_res]

    adam_mean = np.mean(adam_epochs)
    bel_mean = np.mean(bel_epochs)

    print("\n--- Results Summary ---")
    print(f"AdamW Mean Grok Epoch: {adam_mean:.2f} (Raw: {adam_epochs})")
    print(f"Belavkin Mean Grok Epoch: {bel_mean:.2f} (Raw: {bel_epochs})")

    if bel_mean < adam_mean:
        speedup = (adam_mean - bel_mean) / adam_mean * 100
        print(f"CONCLUSION: AGREE. Belavkin optimizer accelerated Grokking by {speedup:.2f}%.")
    else:
        print("CONCLUSION: REFUTE. Belavkin optimizer failed to accelerate Grokking.")

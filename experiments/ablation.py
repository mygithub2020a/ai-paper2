
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from belavkin.data.modular import ModularDataset
from belavkin.optimizers.adam_b import AdamWB
import argparse
import json
import time

# Re-using the model and dataset from modular_arithmetic.py but focusing on ablation
# Copying ModularTransformer class here to be self-contained or importing?
# Let's import.
from experiments.modular_arithmetic import ModularTransformer

def run_ablation(signal_mode, epochs=200, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

    p = 97
    dataset = ModularDataset(p=p, operation='add', train_fraction=0.5, seed=seed)
    train_loader = DataLoader(dataset, batch_size=512, shuffle=True)
    dataset.eval()
    test_loader = DataLoader(dataset, batch_size=512, shuffle=False)

    model = ModularTransformer(p)
    criterion = nn.CrossEntropyLoss()

    # AdamWB with defaults but varying signal
    optimizer = AdamWB(model.parameters(), lr=1e-3, weight_decay=1.0,
                       gamma_damping=1.0, lambda_collapse=0.1,
                       signal_mode=signal_mode)

    history = {'train_loss': [], 'test_acc': []}

    dataset.train()
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        history['train_loss'].append(total_loss / len(train_loader))

        if epoch % 10 == 0:
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for x, y in test_loader:
                    logits = model(x)
                    pred = logits.argmax(dim=1)
                    correct += (pred == y).sum().item()
                    total += y.size(0)
            history['test_acc'].append(correct / total)

    return history

if __name__ == "__main__":
    modes = ['innovation', 'random', 'magnitude']
    results = {}

    for mode in modes:
        print(f"Running Ablation: {mode}")
        hist = run_ablation(mode, epochs=200)
        results[mode] = hist
        print(f"Final Acc: {hist['test_acc'][-1]:.4f}")

    with open('results_ablation.json', 'w') as f:
        json.dump(results, f)

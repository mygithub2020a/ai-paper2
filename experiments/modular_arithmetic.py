
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from belavkin.data.modular import ModularDataset
from belavkin.optimizers.adam_b import AdamWB
import argparse
import time
import json
import os

class ModularTransformer(nn.Module):
    def __init__(self, p, d_model=128, num_layers=1, nhead=4):
        super().__init__()
        self.embedding = nn.Embedding(p, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(2, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model*4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, p)

    def forward(self, x):
        # x: [batch, 2]
        emb = self.embedding(x) + self.pos_embedding # [batch, 2, d_model]
        out = self.transformer(emb) # [batch, 2, d_model]
        # Mean pooling
        out = out.mean(dim=1) # [batch, d_model]
        logits = self.fc(out) # [batch, p]
        return logits

def run_experiment(optimizer_name, p=97, epochs=1000, seed=42, device='cpu'):
    torch.manual_seed(seed)
    np.random.seed(seed)

    dataset = ModularDataset(p=p, operation='add', train_fraction=0.5, seed=seed)
    train_loader = DataLoader(dataset, batch_size=512, shuffle=True)

    # Test set
    dataset.eval()
    test_loader = DataLoader(dataset, batch_size=512, shuffle=False)

    model = ModularTransformer(p).to(device)
    criterion = nn.CrossEntropyLoss()

    if optimizer_name == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1.0)
    elif optimizer_name == 'AdamWB':
        # Tuned hyperparameters for Belavkin
        optimizer = AdamWB(model.parameters(), lr=1e-3, weight_decay=1.0,
                           gamma_damping=1.0, lambda_collapse=0.1)
    elif optimizer_name == 'AdamWB-NoDamp':
        optimizer = AdamWB(model.parameters(), lr=1e-3, weight_decay=1.0,
                           gamma_damping=0.0, lambda_collapse=0.1)
    elif optimizer_name == 'AdamWB-NoCollapse':
        optimizer = AdamWB(model.parameters(), lr=1e-3, weight_decay=1.0,
                           gamma_damping=1.0, lambda_collapse=0.0)
    else:
        raise ValueError("Unknown optimizer")

    history = {'train_loss': [], 'test_acc': [], 'time': []}
    start_time = time.time()

    dataset.train()
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        history['train_loss'].append(avg_loss)

        if epoch % 10 == 0:
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for x, y in test_loader:
                    x, y = x.to(device), y.to(device)
                    logits = model(x)
                    pred = logits.argmax(dim=1)
                    correct += (pred == y).sum().item()
                    total += y.size(0)
            acc = correct / total
            history['test_acc'].append(acc)

            # Grokking Check: If Acc > 99%, stop? No, let's run full duration to see stability.
            # print(f"Epoch {epoch}: Loss {avg_loss:.4f}, Acc {acc:.4f}")

    history['time'] = time.time() - start_time
    return history

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=str, default='results_modular.json')
    args = parser.parse_args()

    results = {}
    opts = ['AdamW', 'AdamWB', 'AdamWB-NoDamp', 'AdamWB-NoCollapse']

    for opt in opts:
        print(f"Running {opt}...")
        # Short run for testing, increase for real paper
        # Reducing to 200 epochs for speed in sandbox
        hist = run_experiment(opt, epochs=200, seed=42)
        results[opt] = hist

    with open(args.out, 'w') as f:
        json.dump(results, f)
    print("Done.")

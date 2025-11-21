import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.modular_arithmetic import ModularAdditionDataset, TransformerGrokkingModel
from src.optimizer import BelavkinOptimizer
import time
import sys

def train_and_evaluate(optimizer_name, epochs=1000, p=113, train_fraction=0.5, lr=1e-3, weight_decay=0.1):
    print(f"Starting training with {optimizer_name}...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataset = ModularAdditionDataset(p=p, split='train', train_fraction=train_fraction)
    val_dataset = ModularAdditionDataset(p=p, split='val', train_fraction=train_fraction)

    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)

    model = TransformerGrokkingModel(p=p, d_model=128, nhead=4, num_layers=2).to(device)

    criterion = nn.CrossEntropyLoss()

    if optimizer_name == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.98))
    elif optimizer_name == 'Belavkin':
        optimizer = BelavkinOptimizer(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.98), adaptive_decay=True)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    results = []
    grok_epoch = None

    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            if optimizer_name == 'Belavkin':
                optimizer.step()
            else:
                optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        train_loss /= total
        train_acc = 100. * correct / total

        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()

        val_loss /= val_total
        val_acc = 100. * val_correct / val_total

        if epoch % 50 == 0 or val_acc > 95:
             print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        if val_acc > 99.0 and grok_epoch is None:
            grok_epoch = epoch
            print(f"GROKKED at epoch {epoch}!")
            # Continue training a bit to confirm stability, but we found the metric.
            # We don't stop immediately as we want to see if it stays high.

        results.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc
        })

        # Stop condition: significantly past grokking or max epochs
        if grok_epoch and epoch > grok_epoch + 100:
            break

    end_time = time.time()
    print(f"Training finished in {end_time - start_time:.2f} seconds.")
    return grok_epoch, results

if __name__ == "__main__":
    # Small P for faster training/grokking potential within time limits
    P = 113
    EPOCHS = 2000

    print("--- Benchmarking AdamW ---")
    grok_adam, res_adam = train_and_evaluate('AdamW', epochs=EPOCHS, p=P)

    print("\n--- Benchmarking Belavkin ---")
    grok_belavkin, res_belavkin = train_and_evaluate('Belavkin', epochs=EPOCHS, p=P)

    print("\n=== RESULTS ===")
    print(f"AdamW Grok Epoch: {grok_adam if grok_adam else 'Not Grokked'}")
    print(f"Belavkin Grok Epoch: {grok_belavkin if grok_belavkin else 'Not Grokked'}")

    if grok_belavkin and grok_adam:
        if grok_belavkin < grok_adam:
            print("CONCLUSION: Belavkin was FASTER.")
        elif grok_belavkin > grok_adam:
            print("CONCLUSION: Belavkin was SLOWER.")
        else:
            print("CONCLUSION: TIED.")
    elif grok_belavkin and not grok_adam:
        print("CONCLUSION: Belavkin SUCCEEDED where AdamW FAILED (within epoch limit).")
    elif not grok_belavkin and grok_adam:
        print("CONCLUSION: AdamW SUCCEEDED where Belavkin FAILED.")
    else:
        print("CONCLUSION: BOTH FAILED to grok within epoch limit.")

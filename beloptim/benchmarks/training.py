import torch
import torch.nn as nn
from tqdm import tqdm
from beloptim.benchmarks.datasets import get_modular_arithmetic_dataloader
from beloptim.benchmarks.models import ModularArithmeticNet
from beloptim.optimizers.belavkin import BelOptim
from beloptim.optimizers.belavkin_variants import BelOptimWithMomentum, BelOptimAdaptive, BelOptimLayerwise
import torch.optim as optim

def train_modular_arithmetic(optimizer_name, p, operation, d_model, n_layers, n_head,
                             num_samples, batch_size, epochs, lr, gamma, beta,
                             device):

    # Dataloaders
    train_loader = get_modular_arithmetic_dataloader(p, operation, num_samples, batch_size)
    val_loader = get_modular_arithmetic_dataloader(p, operation, int(num_samples * 0.25), batch_size)

    # Model
    model = ModularArithmeticNet(p, d_model, n_layers, n_head).to(device)

    # Optimizer
    if optimizer_name == 'BelOptim':
        optimizer = BelOptim(model.parameters(), lr=lr, gamma=gamma, beta=beta)
    elif optimizer_name == 'BelOptimWithMomentum':
        optimizer = BelOptimWithMomentum(model.parameters(), lr=lr, gamma=gamma, beta=beta)
    elif optimizer_name == 'BelOptimAdaptive':
        optimizer = BelOptimAdaptive(model.parameters(), lr=lr, gamma0=gamma, beta=beta)
    elif optimizer_name == 'BelOptimLayerwise':
        optimizer = BelOptimLayerwise(model.parameters(), lr=lr, gamma_base=gamma, beta_base=beta)
    elif optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    # Loss function
    criterion = nn.CrossEntropyLoss()

    history = {'train_loss': [], 'val_accuracy': []}

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Training]"):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        history['train_loss'].append(train_loss)

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Validation]"):
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                _, predicted = torch.max(outputs.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()

        val_accuracy = 100 * correct / total
        history['val_accuracy'].append(val_accuracy)

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

    return history

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt

# Custom optimizer
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from optimizer.belavkin_optimizer import BelavkinOptimizer

# 1. Define a simple neural network model
class SimpleModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, output_size)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

# 2. Generate synthetic datasets
def generate_modular_arithmetic_data(p, num_samples=1000):
    """Generates data for a + b mod p."""
    a = torch.randint(0, p, (num_samples, 1))
    b = torch.randint(0, p, (num_samples, 1))
    x = torch.cat((a, b), dim=1).float()
    y = ((a + b) % p).float()
    return TensorDataset(x, y)

def generate_modular_composition_data(p, num_samples=1000):
    """Generates data for (a * x + b) mod p."""
    a = torch.randint(0, p, (num_samples, 1))
    b = torch.randint(0, p, (num_samples, 1))
    x_val = torch.randint(0, p, (num_samples, 1))
    x = torch.cat((a, b, x_val), dim=1).float()
    y = ((a * x_val + b) % p).float()
    return TensorDataset(x, y)

# 3. Training loop
def train(model, optimizer, criterion, data_loader, epochs=100):
    loss_history = []
    for epoch in range(epochs):
        for x_batch, y_batch in data_loader:
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
        loss_history.append(loss.item())
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
    return loss_history

# 4. Benchmarking setup
def run_benchmarks():
    p = 59  # A prime number
    num_samples = 2000
    batch_size = 64
    epochs = 50

    # Modular arithmetic dataset
    mod_arith_dataset = generate_modular_arithmetic_data(p, num_samples)
    mod_arith_loader = DataLoader(mod_arith_dataset, batch_size=batch_size, shuffle=True)

    # Modular composition dataset
    mod_comp_dataset = generate_modular_composition_data(p, num_samples)
    mod_comp_loader = DataLoader(mod_comp_dataset, batch_size=batch_size, shuffle=True)

    datasets = {
        "Modular Arithmetic": (mod_arith_loader, 2, 1),
        "Modular Composition": (mod_comp_loader, 3, 1)
    }

    optimizers = {
        "Adam": optim.Adam,
        "SGD": optim.SGD,
        "RMSprop": optim.RMSprop,
        "Belavkin": BelavkinOptimizer
    }

    results = {}

    for name, (data_loader, input_size, output_size) in datasets.items():
        print(f"--- Benchmarking on {name} ---")

        # Initialize the model and loss function
        model = SimpleModel(input_size, output_size)
        criterion = nn.MSELoss()

        loss_curves = {}
        for opt_name, opt_class in optimizers.items():
            print(f"Running with {opt_name}...")
            # Re-initialize model for a fair comparison
            model = SimpleModel(input_size, output_size)
            if opt_name == "Belavkin":
                optimizer = BelavkinOptimizer(model.parameters(), eta=1e-4, gamma=1e-5, beta=1e-3, clip_value=1.0)
            else:
                optimizer = opt_class(model.parameters(), lr=0.01)

            loss_history = train(model, optimizer, criterion, data_loader, epochs)
            loss_curves[opt_name] = loss_history

        results[name] = loss_curves

    return results

def plot_results(results):
    for name, loss_curves in results.items():
        plt.figure(figsize=(10, 6))
        for opt_name, loss_history in loss_curves.items():
            plt.plot(loss_history, label=opt_name)

        plt.title(f"Optimizer Performance on {name}")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{name.replace(' ', '_').lower()}_benchmark.png")
        plt.show()

if __name__ == '__main__':
    benchmark_results = run_benchmarks()
    plot_results(benchmark_results)
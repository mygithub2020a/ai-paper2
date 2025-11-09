import torch
import torch.nn as nn
import torch.optim as optim
from src.optimizer import BelavkinOptimizer
from src.dataset import generate_modular_arithmetic_dataset, generate_modular_composition_dataset
import pandas as pd
import os

# Define a simple MLP model
class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

def train(model, optimizer, X, y, epochs=1000):
    criterion = nn.CrossEntropyLoss()
    loss_history = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())
        if (epoch+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
    return loss_history

def main():
    # Parameters
    p_arithmetic = 113
    n_samples_arithmetic = p_arithmetic * p_arithmetic
    p_composition = 19
    n_functions_composition = 10
    n_samples_composition = 5000
    hidden_size = 128
    epochs = 2000

    # Create data directory if it doesn't exist
    if not os.path.exists('data'):
        os.makedirs('data')

    # --- Modular Arithmetic Benchmark ---
    print("--- Starting Modular Arithmetic Benchmark ---")
    X_arith, y_arith = generate_modular_arithmetic_dataset(p_arithmetic, n_samples_arithmetic)
    input_size_arith = X_arith.shape[1]
    output_size_arith = p_arithmetic

    optimizers_to_test = {
        "Belavkin": BelavkinOptimizer,
        "Adam": optim.Adam,
        "SGD": optim.SGD,
        "RMSprop": optim.RMSprop
    }

    results_arithmetic = {}

    for name, opt_class in optimizers_to_test.items():
        print(f"Training with {name} optimizer...")
        model_arith = SimpleMLP(input_size_arith, hidden_size, output_size_arith)
        optimizer = opt_class(model_arith.parameters(), lr=1e-3)
        loss_history = train(model_arith, optimizer, X_arith, y_arith, epochs)
        results_arithmetic[name] = loss_history

    df_arithmetic = pd.DataFrame(results_arithmetic)
    df_arithmetic.to_csv("data/modular_arithmetic_benchmark.csv", index=False)
    print("Modular Arithmetic Benchmark finished. Results saved to data/modular_arithmetic_benchmark.csv")


    # --- Modular Composition Benchmark ---
    print("\n--- Starting Modular Composition Benchmark ---")
    X_comp, y_comp = generate_modular_composition_dataset(p_composition, n_functions_composition, n_samples_composition)
    input_size_comp = X_comp.shape[1]
    output_size_comp = p_composition

    results_composition = {}

    for name, opt_class in optimizers_to_test.items():
        print(f"Training with {name} optimizer...")
        model_comp = SimpleMLP(input_size_comp, hidden_size, output_size_comp)
        optimizer = opt_class(model_comp.parameters(), lr=1e-3)
        loss_history = train(model_comp, optimizer, X_comp, y_comp, epochs)
        results_composition[name] = loss_history

    df_composition = pd.DataFrame(results_composition)
    df_composition.to_csv("data/modular_composition_benchmark.csv", index=False)
    print("Modular Composition Benchmark finished. Results saved to data/modular_composition_benchmark.csv")


if __name__ == "__main__":
    main()

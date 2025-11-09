import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time

from belavkin.optimizer import BelavkinOptimizer
from belavkin.dataset import ModularArithmeticDataset, ModularCompositionDataset
from belavkin.model import SimpleMLP

def train(optimizer_class, dataset, model, epochs=10, **optimizer_kwargs):
    criterion = nn.MSELoss()
    optimizer = optimizer_class(model.parameters(), **optimizer_kwargs)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    start_time = time.time()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for data, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs.squeeze(), labels.float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(dataloader):.4f}")
    end_time = time.time()
    return loss.item(), end_time - start_time

def main():
    datasets = {
        "modular_arithmetic": ModularArithmeticDataset(num_samples=1000),
        "modular_composition": ModularCompositionDataset(num_samples=1000)
    }

    optimizers = {
        "Adam": (torch.optim.Adam, {}),
        "SGD": (torch.optim.SGD, {}),
        "RMSprop": (torch.optim.RMSprop, {}),
    }

    belavkin_hyperparams = [
        {"eta": 1e-3, "gamma": 0.1},
        {"eta": 1e-4, "gamma": 0.05},
        {"eta": 1e-3, "gamma": 0.2},
        {"eta": 1e-4, "gamma": 0.1},
        {"eta": 1e-5, "gamma": 0.05},
    ]

    results = {}

    for d_name, dataset in datasets.items():
        results[d_name] = {}
        input_size = dataset.data.shape[1]

        for o_name, (optimizer, kwargs) in optimizers.items():
            model = SimpleMLP(input_size=input_size, hidden_size=64, output_size=1)
            loss, duration = train(optimizer, dataset, model, **kwargs)
            results[d_name][o_name] = {"loss": loss, "duration": duration}

        for i, kwargs in enumerate(belavkin_hyperparams):
            model = SimpleMLP(input_size=input_size, hidden_size=64, output_size=1)
            loss, duration = train(BelavkinOptimizer, dataset, model, **kwargs)
            o_name = f"Belavkin_{i+1}"
            results[d_name][o_name] = {"loss": loss, "duration": duration, "params": kwargs}

    for d_name, d_results in results.items():
        print(f"--- {d_name} ---")
        for o_name, o_results in d_results.items():
            if "params" in o_results:
                print(f"{o_name} ({o_results['params']}): Loss = {o_results['loss']:.4f}, Duration = {o_results['duration']:.2f}s")
            else:
                print(f"{o_name}: Loss = {o_results['loss']:.4f}, Duration = {o_results['duration']:.2f}s")

if __name__ == "__main__":
    main()

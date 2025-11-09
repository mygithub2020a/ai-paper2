import torch
import torch.nn as nn
import torch.optim as optim
from optimizer import BelavkinOptimizer
from dataset_generator import generate_modular_arithmetic_dataset, generate_modular_composition_dataset
import matplotlib.pyplot as plt

# Define a simple neural network model
class SimpleModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleModel, self).__init__()
        self.linear1 = nn.Linear(input_size, 128)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(128, output_size)

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        return out

def train(optimizer_class, X, y, model, epochs=100, **kwargs):
    optimizer = optimizer_class(model.parameters(), **kwargs)
    criterion = nn.CrossEntropyLoss()
    losses = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X.float())
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses

def main():
    num_samples = 1000
    modulus = 117
    epochs = 200

    # Modular Arithmetic Dataset
    X_arith, y_arith = generate_modular_arithmetic_dataset(num_samples, modulus)
    model_arith = SimpleModel(2, modulus)

    optimizers = {
        'Belavkin': (BelavkinOptimizer, {'gamma': 0.0001, 'eta': 0.001, 'beta': 0.0001}),
        'Adam': (optim.Adam, {'lr': 0.001}),
        'SGD': (optim.SGD, {'lr': 0.01}),
        'RMSprop': (optim.RMSprop, {'lr': 0.001})
    }

    results_arith = {}
    for name, (optimizer_class, kwargs) in optimizers.items():
        print(f"Training with {name} on modular arithmetic dataset...")
        model_arith = SimpleModel(2, modulus)
        losses = train(optimizer_class, X_arith, y_arith, model_arith, epochs, **kwargs)
        results_arith[name] = losses

    # Modular Composition Dataset
    X_comp, y_comp = generate_modular_composition_dataset(num_samples, modulus)
    model_comp = SimpleModel(3, modulus)

    results_comp = {}
    for name, (optimizer_class, kwargs) in optimizers.items():
        print(f"Training with {name} on modular composition dataset...")
        model_comp = SimpleModel(3, modulus)
        losses = train(optimizer_class, X_comp, y_comp, model_comp, epochs, **kwargs)
        results_comp[name] = losses


    # Plotting the results
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    for name, losses in results_arith.items():
        plt.plot(losses, label=name)
    plt.title('Modular Arithmetic')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    for name, losses in results_comp.items():
        plt.plot(losses, label=name)
    plt.title('Modular Composition')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig('benchmark_results.png')
    print("Benchmark results saved to benchmark_results.png")


if __name__ == '__main__':
    main()

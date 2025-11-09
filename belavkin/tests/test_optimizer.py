import torch
import torch.nn as nn
import numpy as np

from belavkin.optimizer.belavkin_optimizer import BelavkinOptimizer

def test_modular_arithmetic():
    """
    Tests the BelavkinOptimizer on a modular arithmetic task.
    """
    # 1. Define the task: f(x) = (ax + b) mod p
    p = 113
    a = 5
    b = 13

    # 2. Generate data
    X_unnormalized = torch.arange(p).view(-1, 1).float()
    X = X_unnormalized / (p - 1)
    y = ((a * X + b) % p).squeeze().long()

    # 3. Define the model
    class MLP(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(MLP, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            out = self.fc1(x)
            out = self.relu(out)
            out = self.fc2(out)
            return out

    model = MLP(input_size=1, hidden_size=256, output_size=p)

    # 4. Instantiate the optimizer and loss function
    optimizer = BelavkinOptimizer(model.parameters(), lr=1e-3, gamma=1e-4, beta=1e-2)
    criterion = nn.CrossEntropyLoss()

    # 5. Training loop
    epochs = 20000
    for epoch in range(epochs):
        # Forward pass
        outputs = model(X)
        loss = criterion(outputs, y)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

    # 6. Evaluate the model
    with torch.no_grad():
        outputs = model(X)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == y).sum().item()
        accuracy = correct / y.size(0)

    print(f'Final Accuracy: {accuracy * 100:.2f}%')

    # 7. Assert that the final accuracy is above a threshold
    assert accuracy > 0.95, f"Accuracy {accuracy} is below the 95% threshold."

if __name__ == '__main__':
    test_modular_arithmetic()

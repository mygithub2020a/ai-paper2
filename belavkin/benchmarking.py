import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from belavkin.optimizer import BelavkinOptimizer
from belavkin.dataset_generator import modular_arithmetic_dataset

# --- 1. Model Definition ---
class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleMLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.layers(x)

# --- 2. Training and Evaluation Functions ---
def train_model(model, optimizer, data_loader, epochs, device):
    model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        for x_batch, y_batch in data_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

def evaluate_model(model, data_loader, device):
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x_batch, y_batch in data_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            outputs = model(x_batch)
            _, predicted = torch.max(outputs.data, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
    return 100 * correct / total

# --- 3. Main Benchmarking Logic ---
if __name__ == "__main__":
    # Hyperparameters
    P = 97  # A prime number
    A = 3
    B = 13
    NUM_SAMPLES = 5000
    TRAIN_SPLIT = 0.8
    BATCH_SIZE = 64
    EPOCHS = 10
    HIDDEN_SIZE = 256
    BELAVKIN_LR = 1e-3
    ADAM_LR = 1e-3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Generate dataset
    full_dataset = modular_arithmetic_dataset(P, A, B, num_samples=NUM_SAMPLES)
    train_size = int(TRAIN_SPLIT * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # --- Belavkin Optimizer Benchmark ---
    print("\n--- Training with Belavkin Optimizer ---")
    model_belavkin = SimpleMLP(input_size=1, hidden_size=HIDDEN_SIZE, output_size=P)
    optimizer_belavkin = BelavkinOptimizer(model_belavkin.parameters(), lr=BELAVKIN_LR, adaptive_gamma=True)

    train_model(model_belavkin, optimizer_belavkin, train_loader, EPOCHS, device)
    accuracy_belavkin = evaluate_model(model_belavkin, test_loader, device)
    print(f"Final Accuracy (Belavkin): {accuracy_belavkin:.2f}%")

    # --- Adam Optimizer Benchmark ---
    print("\n--- Training with Adam Optimizer ---")
    model_adam = SimpleMLP(input_size=1, hidden_size=HIDDEN_SIZE, output_size=P)
    optimizer_adam = torch.optim.Adam(model_adam.parameters(), lr=ADAM_LR)

    train_model(model_adam, optimizer_adam, train_loader, EPOCHS, device)
    accuracy_adam = evaluate_model(model_adam, test_loader, device)
    print(f"Final Accuracy (Adam): {accuracy_adam:.2f}%")

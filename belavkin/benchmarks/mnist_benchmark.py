import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from belavkin.optimizer.optimizer import BelavkinOptimizer

# --- Model Definition ---
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

# --- Data Loading and Noise Injection ---
def get_mnist_loader(train=True, batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset = datasets.MNIST('./data', train=train, download=True, transform=transform)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

def inject_label_noise(labels, noise_level=0.3, num_classes=10):
    if torch.rand(1) < noise_level:
        shuffled_labels = labels[torch.randperm(labels.size(0))]
        return shuffled_labels
    return labels

# --- Training and Evaluation ---
def train(model, optimizer, train_loader, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        target = inject_label_noise(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}]\\tLoss: {loss.item():.6f}')

def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'\\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.0f}%)\\n')
    return accuracy

# --- Main ---
if __name__ == '__main__':
    train_loader = get_mnist_loader(train=True)
    test_loader = get_mnist_loader(train=False)

    # Belavkin Optimizer
    print("--- Training with Belavkin Optimizer ---")
    model_belavkin = SimpleCNN()
    optimizer_belavkin = BelavkinOptimizer(model_belavkin.parameters(), lr=0.01)
    for epoch in range(1, 3):
        train(model_belavkin, optimizer_belavkin, train_loader, epoch)
    belavkin_accuracy = test(model_belavkin, test_loader)

    # Adam Optimizer
    print("\\n--- Training with Adam Optimizer ---")
    model_adam = SimpleCNN()
    optimizer_adam = torch.optim.Adam(model_adam.parameters(), lr=0.001)
    for epoch in range(1, 3):
        train(model_adam, optimizer_adam, train_loader, epoch)
    adam_accuracy = test(model_adam, test_loader)

    print("\\n--- Final Results ---")
    print(f"Belavkin Optimizer Final Accuracy: {belavkin_accuracy:.2f}%")
    print(f"Adam Optimizer Final Accuracy: {adam_accuracy:.2f}%")

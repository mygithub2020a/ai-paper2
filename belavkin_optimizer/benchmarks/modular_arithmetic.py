import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class ModularArithmeticModel(nn.Module):
    def __init__(self, n, hidden_dim=128):
        super().__init__()
        self.n = n
        self.embed = nn.Embedding(n, hidden_dim)
        self.linear1 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, n)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.embed(x).view(x.shape[0], -1)
        x = self.relu(self.linear1(x))
        x = self.linear2(x)
        return x

class ModularArithmeticDataset(Dataset):
    def __init__(self, n, op='add'):
        self.n = n
        self.op = op
        self.x1 = torch.arange(n)
        self.x2 = torch.arange(n)

    def __len__(self):
        return self.n * self.n

    def __getitem__(self, idx):
        x1 = self.x1[idx // self.n]
        x2 = self.x2[idx % self.n]

        if self.op == 'add':
            y = (x1 + x2) % self.n
        elif self.op == 'mul':
            y = (x1 * x2) % self.n
        else:
            raise ValueError(f"Unknown operation: {self.op}")

        return torch.tensor([x1, x2]), y

def train(model, optimizer, dataloader, epochs):
    criterion = nn.CrossEntropyLoss()
    final_loss = None
    for epoch in range(epochs):
        for x, y in dataloader:
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            final_loss = loss.item()
    return final_loss

def evaluate(model, dataloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in dataloader:
            y_pred = model(x)
            _, predicted = torch.max(y_pred.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    return correct / total

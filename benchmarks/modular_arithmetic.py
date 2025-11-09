import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class ModularAdditionDataset(Dataset):
    def __init__(self, n, train=True, train_split=0.8):
        self.n = n
        self.train = train
        self.train_split = train_split

        # Create a list of all possible pairs (a, b)
        self.all_pairs = []
        for a in range(n):
            for b in range(n):
                self.all_pairs.append((a, b))

        # Shuffle the pairs and split into train and test sets
        torch.manual_seed(42)
        indices = torch.randperm(len(self.all_pairs))
        split_idx = int(len(self.all_pairs) * self.train_split)

        if self.train:
            self.indices = indices[:split_idx]
        else:
            self.indices = indices[split_idx:]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        a, b = self.all_pairs[self.indices[idx]]
        result = (a + b) % self.n

        # One-hot encode the inputs and output
        a_one_hot = torch.nn.functional.one_hot(torch.tensor(a), num_classes=self.n).float()
        b_one_hot = torch.nn.functional.one_hot(torch.tensor(b), num_classes=self.n).float()

        return torch.cat([a_one_hot, b_one_hot]), result

class SimpleMLP(nn.Module):
    def __init__(self, n, hidden_dim=128):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(2 * n, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

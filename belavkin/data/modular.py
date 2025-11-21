
import torch
from torch.utils.data import Dataset
import random

class ModularDataset(Dataset):
    def __init__(self, p=97, operation='add', train_fraction=0.5, seed=42):
        self.p = p
        self.operation = operation
        self.data = []

        random.seed(seed)

        pairs = [(i, j) for i in range(p) for j in range(p)]
        random.shuffle(pairs)

        for i, j in pairs:
            if operation == 'add':
                res = (i + j) % p
            elif operation == 'mult':
                res = (i * j) % p
            else:
                raise ValueError("Operation must be 'add' or 'mult'")
            self.data.append((i, j, res))

        split_idx = int(len(self.data) * train_fraction)
        self.train_data = self.data[:split_idx]
        self.test_data = self.data[split_idx:]

        self.is_train = True

    def train(self):
        self.is_train = True

    def eval(self):
        self.is_train = False

    def __len__(self):
        return len(self.train_data) if self.is_train else len(self.test_data)

    def __getitem__(self, idx):
        data_source = self.train_data if self.is_train else self.test_data
        i, j, res = data_source[idx]
        return torch.tensor([i, j]), torch.tensor(res)

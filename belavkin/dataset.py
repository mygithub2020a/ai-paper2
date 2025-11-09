import torch
from torch.utils.data import Dataset

class ModularArithmeticDataset(Dataset):
    def __init__(self, num_samples, max_val=100):
        self.num_samples = num_samples
        self.max_val = max_val
        self.data, self.labels = self._generate_data()

    def _generate_data(self):
        data = []
        labels = []
        for _ in range(self.num_samples):
            a = torch.randint(0, self.max_val, (1,)).item()
            b = torch.randint(0, self.max_val, (1,)).item()
            p = torch.randint(1, self.max_val, (1,)).item()
            data.append((a, b, p))
            labels.append((a + b) % p)
        return torch.tensor(data), torch.tensor(labels)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class ModularCompositionDataset(Dataset):
    def __init__(self, num_samples, max_val=100):
        self.num_samples = num_samples
        self.max_val = max_val
        self.data, self.labels = self._generate_data()

    def _generate_data(self):
        data = []
        labels = []
        for _ in range(self.num_samples):
            a = torch.randint(0, self.max_val, (1,)).item()
            b = torch.randint(0, self.max_val, (1,)).item()
            x = torch.randint(0, self.max_val, (1,)).item()
            p = torch.randint(1, self.max_val, (1,)).item()
            data.append((a, b, x, p))
            labels.append((a * b * x) % p)
        return torch.tensor(data), torch.tensor(labels)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

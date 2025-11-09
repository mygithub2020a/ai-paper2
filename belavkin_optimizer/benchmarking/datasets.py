import torch
from torch.utils.data import Dataset, DataLoader

class ModularArithmeticDataset(Dataset):
    def __init__(self, num_samples, modulus, num_features=2):
        self.num_samples = num_samples
        self.modulus = modulus
        self.num_features = num_features
        self.X = torch.randint(0, modulus, (num_samples, num_features))
        self.a = torch.randint(0, modulus, (num_features, 1))
        self.b = torch.randint(0, modulus, (1,))
        self.y = (torch.matmul(self.X, self.a) + self.b) % modulus

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class ModularCompositionDataset(Dataset):
    def __init__(self, num_samples, modulus1, modulus2):
        self.num_samples = num_samples
        self.modulus1 = modulus1
        self.modulus2 = modulus2
        self.X = torch.randint(0, modulus1, (num_samples, 1))
        self.a = torch.randint(0, modulus1, (1,))
        self.b = torch.randint(0, modulus1, (1,))
        self.d = torch.randint(0, modulus2, (1,))
        self.e = torch.randint(0, modulus2, (1,))

        f_x = (self.a * self.X + self.b) % self.modulus1
        self.y = (self.d * f_x + self.e) % self.modulus2

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

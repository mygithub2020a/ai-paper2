import torch
from torch.utils.data import Dataset, DataLoader

class ModularArithmeticDataset(Dataset):
    """
    Dataset for modular arithmetic tasks.

    Args:
        p (int): The modulus.
        operation (str): 'add', 'compose', or 'poly'.
        num_samples (int): The number of samples to generate.
    """
    def __init__(self, p, operation='add', num_samples=10000):
        self.p = p
        self.operation = operation
        self.num_samples = num_samples

        self.X = torch.randint(0, p, (num_samples, 2))
        if self.operation == 'add':
            self.y = (self.X[:, 0] + self.X[:, 1]) % p
        elif self.operation == 'compose':
            self.y = (self.X[:, 0] * self.X[:, 1] + self.X[:, 0] + self.X[:, 1]) % p
        elif self.operation == 'poly':
            # f(x) = (ax^2 + bx + c) mod p
            # For simplicity, we'll fix the coefficients
            a, b, c = 2, 3, 5
            self.X = torch.randint(0, p, (num_samples, 1))
            self.y = (a * self.X[:, 0]**2 + b * self.X[:, 0] + c) % p
        else:
            raise ValueError(f"Unknown operation: {self.operation}")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def get_modular_arithmetic_dataloader(p, operation, num_samples, batch_size):
    dataset = ModularArithmeticDataset(p, operation, num_samples)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

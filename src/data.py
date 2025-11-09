import torch
from torch.utils.data import Dataset, TensorDataset

class ModularArithmeticDataset(Dataset):
    """
    Dataset for the modular arithmetic task: f(x) = (a * x + b) mod p.
    """
    def __init__(self, num_samples, p, a, b):
        self.num_samples = num_samples
        self.p = p
        self.a = a
        self.b = b
        self.X = torch.randint(0, p, (num_samples, 1))
        self.Y = (self.a * self.X + self.b) % self.p

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

def create_modular_arithmetic_dataset(num_samples=10000, p=97, a=3, b=7):
    return ModularArithmeticDataset(num_samples, p, a, b)

class ModularCompositionDataset(Dataset):
    """
    Dataset for the modular composition task: f(g(x)).
    f(x) = (a1 * x + b1) mod p
    g(x) = (a2 * x + b2) mod p
    """
    def __init__(self, num_samples, p, a1, b1, a2, b2):
        self.num_samples = num_samples
        self.p = p
        self.a1, self.b1, self.a2, self.b2 = a1, b1, a2, b2
        self.X = torch.randint(0, p, (num_samples, 1))
        g_x = (self.a2 * self.X + self.b2) % self.p
        self.Y = (self.a1 * g_x + self.b1) % self.p

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

def create_modular_composition_dataset(num_samples=10000, p=97, a1=3, b1=7, a2=5, b2=11):
    return ModularCompositionDataset(num_samples, p, a1, b1, a2, b2)

class SparseParityDataset(Dataset):
    """
    Dataset for the k-sparse parity task.
    """
    def __init__(self, num_samples, n_bits, k_sparsity):
        self.num_samples = num_samples
        self.n_bits = n_bits
        self.k_sparsity = k_sparsity

        # Choose k indices to be the sparse ones
        self.sparse_indices = torch.randperm(n_bits)[:k_sparsity]

        self.X = torch.randint(0, 2, (num_samples, n_bits)).float()

        # Calculate parity only on the sparse indices
        sparse_bits = self.X[:, self.sparse_indices]
        self.Y = torch.sum(sparse_bits, dim=1) % 2

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx].long()

def create_sparse_parity_dataset(num_samples=10000, n_bits=16, k_sparsity=4):
    return SparseParityDataset(num_samples, n_bits, k_sparsity)

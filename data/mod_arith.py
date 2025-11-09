import torch
from torch.utils.data import Dataset, DataLoader

class ModularArithmeticDataset(Dataset):
    """
    A dataset for modular arithmetic tasks.
    """
    def __init__(self, p, op='add', size=10000):
        self.p = p
        self.op = op
        self.size = size

        if op in ['add', 'multiply', 'power']:
            self.x = torch.randint(0, p, (size, 2))
        elif op == 'invert':
            self.x = torch.randint(1, p, (size, 1)) # Inverse of 0 is undefined
        else:
            raise ValueError(f"Unknown operation: {op}")

        if op == 'add':
            self.y = (self.x[:, 0] + self.x[:, 1]) % p
        elif op == 'multiply':
            self.y = (self.x[:, 0] * self.x[:, 1]) % p
        elif op == 'invert':
            # Using Python's pow(base, exp, mod) for modular inverse
            self.y = torch.tensor([pow(val.item(), -1, p) for val in self.x])
        elif op == 'power':
            # Using Python's pow for modular exponentiation for safety with large exponents
            self.y = torch.tensor([pow(b.item(), e.item(), p) for b, e in self.x])


    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

def get_dataloader(p, op, batch_size, **kwargs):
    dataset = ModularArithmeticDataset(p, op, **kwargs)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

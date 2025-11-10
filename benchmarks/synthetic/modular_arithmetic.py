import torch

def create_modular_arithmetic_dataset(p, a, b, num_samples=1000):
    """
    Creates a dataset for the modular arithmetic task f(x) = (ax + b) mod p.
    """
    x = torch.randint(0, p, (num_samples, 1), dtype=torch.long)
    y = (a * x + b) % p
    return x, y

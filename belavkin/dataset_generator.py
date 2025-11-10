import torch
from torch.utils.data import TensorDataset

def modular_arithmetic_dataset(p, a, b, num_samples=1000):
    """
    Generates a dataset for the task f(x) = (ax + b) mod p.

    Args:
        p (int): The prime modulus.
        a (int): The multiplicative factor.
        b (int): The additive factor.
        num_samples (int): The number of samples to generate.

    Returns:
        TensorDataset: A PyTorch TensorDataset.
    """
    x = torch.randint(0, p, (num_samples, 1), dtype=torch.float32)
    y = (a * x.long() + b) % p
    return TensorDataset(x, y.squeeze(-1))

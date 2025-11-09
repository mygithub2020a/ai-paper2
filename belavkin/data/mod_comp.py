import torch

def generate_random_affine_map(p, dim):
    """Generates a random affine map (Ax + b) mod p."""
    A = torch.randint(0, p, (dim, dim), dtype=torch.float32)
    b = torch.randint(0, p, (dim,), dtype=torch.float32)

    def affine_map(x):
        return (torch.matmul(x, A.T) + b) % p
    return affine_map

def generate_modular_composition_data(p, dim, depth, num_samples):
    """
    Generates data for modular composition tasks.

    Args:
        p (int): The modulus.
        dim (int): The input dimension.
        depth (int): The depth of composition.
        num_samples (int): The number of samples to generate.

    Returns:
        (torch.Tensor, torch.Tensor): A tuple of (X, y) tensors.
    """
    maps = [generate_random_affine_map(p, dim) for _ in range(depth)]

    X = torch.randint(0, p, (num_samples, dim)).float()
    y = X.clone()

    with torch.no_grad():
        for f in maps:
            y = f(y)

    return X, y

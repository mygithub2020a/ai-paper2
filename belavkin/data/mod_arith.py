import torch

def generate_modular_arithmetic_data(task, p, dim, num_samples):
    """
    Generates data for modular arithmetic tasks.

    Args:
        task (str): One of 'add', 'mul', 'inv', 'pow'.
        p (int): The modulus.
        dim (int): The input dimension.
        num_samples (int): The number of samples to generate.

    Returns:
        (torch.Tensor, torch.Tensor): A tuple of (X, y) tensors.
    """
    if task in ['add', 'mul']:
        X1 = torch.randint(0, p, (num_samples, dim))
        X2 = torch.randint(0, p, (num_samples, dim))
        X = torch.cat([X1, X2], dim=1)
        if task == 'add':
            y = (X1 + X2) % p
        else:
            y = (X1 * X2) % p
    elif task == 'inv':
        X = torch.randint(1, p, (num_samples, dim))
        y = torch.tensor([pow(x.item(), -1, p) for x in X]).view(-1, dim)
    elif task == 'pow':
        X = torch.randint(0, p, (num_samples, dim))
        # Use small exponents for simplicity
        exponents = torch.randint(2, 10, (num_samples, dim))
        y = torch.tensor([pow(x.item(), e.item(), p) for x, e in zip(X, exponents)]).view(-1, dim)
        X = torch.cat([X, exponents], dim=1)
    else:
        raise ValueError(f"Unknown task: {task}")

    return X.float(), y.float()

import torch

def modular_addition_data(p, n_samples):
    """
    Generates data for the modular addition task: (a, b) -> (a + b) % p.
    """
    x = torch.randint(0, p, (n_samples, 2))
    y = (x[:, 0] + x[:, 1]) % p
    return x, y

if __name__ == '__main__':
    p = 97
    n_samples = 10
    x, y = modular_addition_data(p, n_samples)
    print("x:", x)
    print("y:", y)

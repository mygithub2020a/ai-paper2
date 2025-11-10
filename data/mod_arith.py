import torch

def modular_addition_data(p, n_samples):
    """
    Generates data for the modular addition task: (a, b) -> (a + b) % p.
    """
    x = torch.randint(0, p, (n_samples, 2))
    y = (x[:, 0] + x[:, 1]) % p
    return x, y

def modular_multiplication_data(p, n_samples):
    """
    Generates data for the modular multiplication task: (a, b) -> (a * b) % p.
    """
    x = torch.randint(0, p, (n_samples, 2))
    y = (x[:, 0] * x[:, 1]) % p
    return x, y

def modular_inversion_data(p, n_samples):
    """
    Generates data for the modular inversion task: a -> a^(-1) % p.
    Note: p must be a prime number.
    """
    # Inverse of 0 is undefined, so we sample from [1, p-1]
    x = torch.randint(1, p, (n_samples, 1))
    # Using Fermat's Little Theorem: a^(p-2) is the modular inverse of a mod p
    y = torch.tensor([pow(val.item(), p - 2, p) for val in x], dtype=torch.long)
    return x, y

def modular_power_data(p, n_samples):
    """
    Generates data for the modular power task: (a, b) -> a^b % p.
    """
    # Sample base 'a' from [0, p-1] and exponent 'b' from a smaller range for feasibility
    a = torch.randint(0, p, (n_samples, 1))
    b = torch.randint(0, p, (n_samples, 1)) # Exponent can also be large
    x = torch.cat((a, b), dim=1)
    y = torch.tensor([pow(base.item(), exp.item(), p) for base, exp in x], dtype=torch.long)
    return x, y

if __name__ == '__main__':
    p = 97
    n_samples = 5

    print("--- Modular Addition ---")
    x_add, y_add = modular_addition_data(p, n_samples)
    print("x:", x_add)
    print("y:", y_add)

    print("\n--- Modular Multiplication ---")
    x_mul, y_mul = modular_multiplication_data(p, n_samples)
    print("x:", x_mul)
    print("y:", y_mul)

    print("\n--- Modular Inversion ---")
    x_inv, y_inv = modular_inversion_data(p, n_samples)
    print("x:", x_inv)
    print("y:", y_inv)
    # Verification
    print("Verification (x * y) % p:", (x_inv * y_inv.unsqueeze(1)) % p)


    print("\n--- Modular Power ---")
    x_pow, y_pow = modular_power_data(p, n_samples)
    print("x (base, exp):", x_pow)
    print("y (result):", y_pow)

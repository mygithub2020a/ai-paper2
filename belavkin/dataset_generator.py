import torch

def generate_modular_arithmetic_dataset(num_samples, modulus):
    """
    Generates a dataset for modular arithmetic.
    Each sample consists of two random integers and their sum modulo the given modulus.
    """
    X = torch.randint(0, modulus, (num_samples, 2))
    y = torch.sum(X, dim=1) % modulus
    return X, y

def generate_modular_composition_dataset(num_samples, modulus):
    """
    Generates a dataset for modular composition.
    Each sample consists of three random integers and the composition (a * b + c) % modulus.
    """
    X = torch.randint(0, modulus, (num_samples, 3))
    y = (X[:, 0] * X[:, 1] + X[:, 2]) % modulus
    return X, y

if __name__ == '__main__':
    # Example usage
    num_samples = 1000
    modulus = 117
    X_arith, y_arith = generate_modular_arithmetic_dataset(num_samples, modulus)
    print("Generated modular arithmetic dataset:")
    print("X shape:", X_arith.shape)
    print("y shape:", y_arith.shape)
    print("Sample X:", X_arith[0])
    print("Sample y:", y_arith[0])

    print("-" * 20)

    X_comp, y_comp = generate_modular_composition_dataset(num_samples, modulus)
    print("\nGenerated modular composition dataset:")
    print("X shape:", X_comp.shape)
    print("y shape:", y_comp.shape)
    print("Sample X:", X_comp[0])
    print("Sample y:", y_comp[0])

import torch

def generate_mod_arith_dataset(num_samples, modulus, seed=42):
    """
    Generates a dataset for the modular arithmetic task (ax + by) mod m.

    Args:
        num_samples (int): The total number of samples to generate.
        modulus (int): The modulus 'm' for the arithmetic operations.
        seed (int, optional): A random seed for reproducibility. Defaults to 42.

    Returns:
        tuple: A tuple containing:
            - data (torch.Tensor): A tensor of shape (num_samples, 4) containing the inputs (a, x, b, y).
            - targets (torch.Tensor): A tensor of shape (num_samples,) containing the targets.
    """
    torch.manual_seed(seed)

    # Sample a, x, b, y from [0, modulus - 1]
    # The shape will be (num_samples, 4)
    data = torch.randint(0, modulus, (num_samples, 4), dtype=torch.long)

    a = data[:, 0]
    x = data[:, 1]
    b = data[:, 2]
    y = data[:, 3]

    # Calculate the targets: (a*x + b*y) % modulus
    targets = (a * x + b * y) % modulus

    return data, targets

if __name__ == '__main__':
    # Example usage:
    num_train_samples = 1000
    num_val_samples = 200
    num_test_samples = 200
    mod = 97

    # Generate splits with different seeds to ensure they don't overlap
    train_data, train_targets = generate_mod_arith_dataset(num_train_samples, mod, seed=42)
    val_data, val_targets = generate_mod_arith_dataset(num_val_samples, mod, seed=43)
    test_data, test_targets = generate_mod_arith_dataset(num_test_samples, mod, seed=44)

    print(f"Generated modular arithmetic dataset with modulus {mod}")
    print(f"Training set size: {train_data.shape[0]}")
    print(f"Validation set size: {val_data.shape[0]}")
    print(f"Test set size: {test_data.shape[0]}")
    print("\nExample data point:")
    print(f"  Inputs (a, x, b, y): {train_data[0]}")
    print(f"  Target ((a*x + b*y) % {mod}): {train_targets[0]}")

    # Verification check
    a, x, b, y = train_data[0]
    expected = (a * x + b * y) % mod
    print(f"  Verification: {expected == train_targets[0]}")

import torch
from torch.utils.data import Dataset
import numpy as np

class ModularArithmeticDataset(Dataset):
    """
    Generates a dataset for modular arithmetic tasks.

    Task: Predict (a*x + b*y) mod m
    - a, b, m are fixed for the dataset instance.
    - x, y are sampled uniformly from [0, m-1].
    """
    def __init__(self, num_samples, modulus, a=None, b=None, seed=42):
        """
        Args:
            num_samples (int): The total number of samples in the dataset.
            modulus (int): The modulus 'm' for the arithmetic operations.
            a (int, optional): The coefficient 'a'. If None, a random integer
                in [1, m-1] is chosen.
            b (int, optional): The coefficient 'b'. If None, a random integer
                in [1, m-1] is chosen.
            seed (int): Random seed for reproducibility.
        """
        self.num_samples = num_samples
        self.modulus = modulus

        rng = np.random.default_rng(seed)

        self.a = a if a is not None else rng.integers(1, self.modulus)
        self.b = b if b is not None else rng.integers(1, self.modulus)

        # Generate all data in advance
        self.X = torch.from_numpy(
            rng.integers(0, self.modulus, size=(num_samples, 2), dtype=np.int64)
        )

        # Calculate targets: (a*x + b*y) mod m
        term1 = self.a * self.X[:, 0]
        term2 = self.b * self.X[:, 1]
        self.y = (term1 + term2) % self.modulus

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def generate_datasets(modulus, num_train=100000, num_val=20000, num_test=20000, seed=42):
    """
    Generates the train, validation, and test splits for a given modulus.
    """
    # Fix a, b for all splits for a given run
    rng = np.random.default_rng(seed)
    a = rng.integers(1, modulus)
    b = rng.integers(1, modulus)

    train_dataset = ModularArithmeticDataset(num_train, modulus, a=a, b=b, seed=seed)
    val_dataset = ModularArithmeticDataset(num_val, modulus, a=a, b=b, seed=seed+1)
    test_dataset = ModularArithmeticDataset(num_test, modulus, a=a, b=b, seed=seed+2)

    return train_dataset, val_dataset, test_dataset

if __name__ == '__main__':
    # --- Example Usage ---
    MODULUS = 97

    train_data, val_data, test_data = generate_datasets(MODULUS)

    print(f"--- Modular Arithmetic Dataset (mod {MODULUS}) ---")
    print(f"Train samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    print(f"Test samples: {len(test_data)}")

    # Print one sample
    x_sample, y_sample = train_data[0]
    a, b = train_data.a, train_data.b
    print(f"\nSample 0:")
    print(f"  a = {a}, b = {b}")
    print(f"  x = {x_sample.tolist()}")
    print(f"  y_true = {y_sample.item()}")
    print(f"  Check: ({a}*{x_sample[0]} + {b}*{x_sample[1]}) % {MODULUS} = {(a*x_sample[0] + b*x_sample[1]) % MODULUS}")

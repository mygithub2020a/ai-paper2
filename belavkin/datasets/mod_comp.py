import torch
from torch.utils.data import Dataset
import numpy as np

class ModularCompositionDataset(Dataset):
    """
    Generates a dataset for the modular composition task.

    Task: Compute (f_k ∘ ... ∘ f_1)(x) mod m
    - Each f_i is an affine map: f_i(z) = (a_i * z + b_i) mod m
    - The sequence of functions (a_i, b_i) is fixed for the dataset.
    - x is sampled uniformly from [0, m-1].
    """
    def __init__(self, num_samples, modulus, num_functions, seed=42):
        """
        Args:
            num_samples (int): The total number of samples in the dataset.
            modulus (int): The modulus 'm' for the arithmetic operations.
            num_functions (int): The number of functions 'k' to compose.
            seed (int): Random seed for reproducibility.
        """
        self.num_samples = num_samples
        self.modulus = modulus
        self.num_functions = num_functions

        rng = np.random.default_rng(seed)

        # Generate the sequence of functions (a_i, b_i)
        self.functions = torch.from_numpy(
            rng.integers(1, self.modulus, size=(num_functions, 2), dtype=np.int64)
        )

        # Generate input data x
        self.X = torch.from_numpy(
            rng.integers(0, self.modulus, size=(num_samples, 1), dtype=np.int64)
        )

        # Calculate the targets by composing the functions
        self.y = self.X.clone().squeeze(1)
        for i in range(self.num_functions):
            a, b = self.functions[i]
            self.y = (a * self.y + b) % self.modulus

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # We return the initial value x and the full sequence of functions
        return self.X[idx], self.functions, self.y[idx]

def generate_composition_datasets(modulus, num_functions, num_train=100000, num_val=20000, num_test=20000, seed=42):
    """
    Generates the train, validation, and test splits for a given composition task.
    """
    train_dataset = ModularCompositionDataset(num_train, modulus, num_functions, seed=seed)
    val_dataset = ModularCompositionDataset(num_val, modulus, num_functions, seed=seed+1)
    test_dataset = ModularCompositionDataset(num_test, modulus, num_functions, seed=seed+2)

    return train_dataset, val_dataset, test_dataset

if __name__ == '__main__':
    # --- Example Usage ---
    MODULUS = 97
    NUM_FUNCTIONS = 4 # k=4

    train_data, _, _ = generate_composition_datasets(MODULUS, NUM_FUNCTIONS)

    print(f"--- Modular Composition Dataset (mod {MODULUS}, k={NUM_FUNCTIONS}) ---")
    print(f"Train samples: {len(train_data)}")

    # Print one sample
    x_sample, funcs_sample, y_sample = train_data[0]
    print(f"\nSample 0:")
    print(f"  x = {x_sample.item()}")
    print(f"  Functions (a, b):")
    for i in range(NUM_FUNCTIONS):
        print(f"    f_{i+1}: a={funcs_sample[i, 0].item()}, b={funcs_sample[i, 1].item()}")
    print(f"  y_true = {y_sample.item()}")

    # Manually check the composition
    y_check = x_sample.item()
    for i in range(NUM_FUNCTIONS):
        a, b = funcs_sample[i]
        y_check = (a * y_check + b) % MODULUS
    print(f"  Check: {y_check}")

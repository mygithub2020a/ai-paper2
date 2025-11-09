import torch

def generate_modular_arithmetic_dataset(p, n_samples):
    """
    Generates a dataset for modular arithmetic.
    The task is to predict (a + b) mod p.
    Inputs are one-hot encoded integers a and b.
    Output is the integer (a + b) mod p.
    """
    # Generate all possible pairs (a, b)
    a = torch.arange(p)
    b = torch.arange(p)
    aa, bb = torch.meshgrid(a, b, indexing='ij')
    # Flatten the pairs
    a_flat = aa.flatten()
    b_flat = bb.flatten()

    # Create one-hot encoded inputs
    a_one_hot = torch.nn.functional.one_hot(a_flat, num_classes=p).float()
    b_one_hot = torch.nn.functional.one_hot(b_flat, num_classes=p).float()
    X = torch.cat([a_one_hot, b_one_hot], dim=1)

    # Calculate the target
    y = (a_flat + b_flat) % p

    # Shuffle the dataset
    indices = torch.randperm(X.shape[0])
    X = X[indices]
    y = y[indices]

    # Take the requested number of samples
    if n_samples > X.shape[0]:
        n_samples = X.shape[0]

    return X[:n_samples], y[:n_samples]

def generate_modular_composition_dataset(p, n_functions, n_samples):
    """
    Generates a dataset for modular composition of affine functions.
    f_i(x) = (a_i * x + b_i) mod p
    The task is to predict f_j(f_i(x)) given indices i, j and input x.
    Inputs are one-hot encoded i, j, and x.
    Output is the integer f_j(f_i(x)).
    """
    # Generate random functions
    a = torch.randint(0, p, (n_functions,))
    b = torch.randint(0, p, (n_functions,))

    # Generate random data
    i = torch.randint(0, n_functions, (n_samples,))
    j = torch.randint(0, n_functions, (n_samples,))
    x = torch.randint(0, p, (n_samples,))

    # One-hot encode inputs
    i_one_hot = torch.nn.functional.one_hot(i, num_classes=n_functions).float()
    j_one_hot = torch.nn.functional.one_hot(j, num_classes=n_functions).float()
    x_one_hot = torch.nn.functional.one_hot(x, num_classes=p).float()
    X = torch.cat([i_one_hot, j_one_hot, x_one_hot], dim=1)

    # f_i(x) = (a[i] * x + b[i])
    # f_j(f_i(x)) = a[j] * (a[i] * x + b[i]) + b[j]
    #             = a[j] * a[i] * x + a[j] * b[i] + b[j]
    y = (a[j] * (a[i] * x + b[i]) + b[j]) % p

    return X, y

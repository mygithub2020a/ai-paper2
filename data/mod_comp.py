import torch
import numpy as np

def generate_polynomial(degree, p):
    """Generates a random polynomial of a given degree with coefficients in [0, p-1]."""
    return np.poly1d(np.random.randint(0, p, degree + 1))

def apply_poly(poly, x, p):
    """Applies a polynomial to x and takes the result modulo p."""
    res = np.polyval(poly, x)
    return int(res % p)

def modular_composition_data(p, n_samples, composition_depth=2, degree=2):
    """
    Generates data for the modular composition task: y = f_n(...f_1(x)) mod p.
    """
    # Generate a set of polynomials
    polynomials = [generate_polynomial(degree, p) for _ in range(composition_depth)]

    x = torch.randint(0, p, (n_samples, 1))
    y = torch.zeros(n_samples, dtype=torch.long)

    for i, val in enumerate(x):
        res = val.item()
        for poly in polynomials:
            res = apply_poly(poly, res, p)
        y[i] = res

    return x, y, polynomials

if __name__ == '__main__':
    p = 97
    n_samples = 5
    depth = 3
    degree = 2

    x, y, polys = modular_composition_data(p, n_samples, composition_depth=depth, degree=degree)

    print(f"--- Modular Composition (depth={depth}, degree={degree}) ---")
    print("Generated Polynomials:")
    for i, poly in enumerate(polys):
        print(f"f_{i+1}(x) = {poly}")

    print("\nx:", x.squeeze())
    print("y:", y)

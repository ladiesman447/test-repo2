import numpy as np
import matplotlib.pyplot as plt
from sympy.abc import x, y
from scipy.sparse.linalg import cg, spsolve
import torch
import torch.nn as nn
from scipy.sparse import identity, kron, diags
from pathlib import Path

def sample_field_params(n_modes=6, rng=None):
    """
    Sample the underlying continuous source field as a tuple of
    sinusoid parameters (kx, ky, amp, phase_x, phase_y).

    Separating "sample the field" from "evaluate the field on a grid"
    (see `evaluate_field` below) is what lets us evaluate the exact
    same continuous field at several different resolutions. If we only
    ever sampled f directly on a grid, there'd be no way to ask "what
    does this same field look like at 256x256" after only ever having
    seen it at 64x64.

    Args:
        n_modes: Number of sinusoidal modes to sample
        rng: Random number generator
    
    Returns:
        params: List of tuples (kx, ky, amp, phase_x, phase_y)
    """
    if rng is None:
        rng = np.random.default_rng()

    params = []
    for _ in range(n_modes): 
        k_x = rng.integers(1, 5)
        k_y = rng.integers(1, 5)
        amp = rng.uniform(-1, 1)
        phase_x = rng.uniform(0, 2 * np.pi)
        phase_y = rng.uniform(0, 2 * np.pi)
        params.append((k_x, k_y, amp, phase_x, phase_y))
    return params

def evaluate_field(params, n):
    """
    Evaluate a field sampled by `sample_field_params` on an n x n grid.
    
    Args:
        params: List of tuples (k_x, k_y, amp, phase_x, phase_y)
        n: Grid size (n x n)
    
    Returns:
        f: Numpy array of shape (n, n) of the params evaluated as a sum of sines

        Each nth mode f is of the following form: 
            f(x, y)_nth = amp * sin(k_x * pi * x + phase_x) * sin(k_y * pi * y + phase_y)
        
        Where:
            amp: real number in [-1, 1]
            k_x, k_y: integer in [1, 4]
            phase_x, phase_y: real number in [0, 2*pi]
    """
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)
    X, Y = np.meshgrid(x, y, indexing="ij")

    f = np.zeros((n, n))
    for k_x, k_y, amp, phase_x, phase_y in params:
        f += amp * np.sin(k_x * np.pi * X + phase_x) * np.sin(k_y * np.pi * Y + phase_y)
    return f

def random_source_field(n, n_modes=6, rng=None):
    """
    Sample a random source term f(x, y) directly on an n x n grid.
    Equivalent to `evaluate_field(sample_field_params(n_modes, rng), n)`;
    kept as a convenience wrapper since Phases 1-4 only ever need f at a
    single, fixed resolution.
    """
    params = sample_field_params(n_modes=n_modes, rng=rng)
    return evaluate_field(params, n)

def discrete_laplacian(n):
    """
    Build the discrete Laplacian matrix for a 2D grid using the finite 
    difference analog of the poisson equation. Output shape is a square
    (n-2) x (n-2) matrix where each row represents the linear equation 
    of the interior point (i,j):

        -u_{i+1,j} - u_{i-1,j} - u_{i,j+1} - u_{i,j-1} + 4*u_{i,j} = h^2 * f_{i,j}
    
    This is done so so that the matrix is posivite-definite and we can 
    use the conjugate gradient method to solve the system.

    Example of the stencil when n = 4:
     ___  ___  ___
    |   ||   ||   |
    |___||___||___|
    |   ||   ||   |
    |___||___||___|
    |   ||   ||   |
    |___||___||___|

    n = 3:
     ______  ______
    |      ||      |
    |      ||      |
    |______||______|
    |      ||      |
    |      ||      |
    |______||______|

    n = 2:
     _____________
    |             |
    |             |
    |             |
    |             |
    |             |
    |_____________|
    

    Args:
        n: Grid size (n x n) where n > 2
    
    Returns:
        A: CSR Laplacian matrix of shape (n - 2) x (n - 2)
    """

    if (n <= 2):
        raise Exception("Matrix not needed. No interior points.")
    
    n_int = n - 2

    e = np.ones(n_int)
    T = diags([e[:-1], -2 * e, e[:-1]], offsets=[-1, 0, 1])
    I = identity(n_int)

    A = kron(I, T) + kron(T, I)
    return -A.tocsr()

def solve_poisson(f, n):
    """
    Solve the 2D Poisson equation on an n x n grid with the following
    Dirichlet boundary conditions:

        u = 0 on the boundary

    Args:
        n: Grid size (n x n)
        f: Source term of shape (n x n)

    Returns:
        u: Solution of shape (n x n)
    """
    h = 1 / (n - 1)

    A = discrete_laplacian(n)
    f_int = f[1:-1, 1:-1].flatten() * h**2
    u_int = spsolve(A, f_int) 

    u = np.zeros((n, n))
    u[1:-1, 1:-1] = u_int.reshape(n-2, n-2)
    return u

def generate_dataset(n_samples, grid_size=64, n_modes=6, seed=0):
    rng = np.random.default_rng(seed)
    F = np.zeros((n_samples, grid_size, grid_size), dtype=np.float32)
    U = np.zeros((n_samples, grid_size, grid_size), dtype=np.float32)

    for i in range(n_samples):
        f = random_source_field(grid_size, n_modes=n_modes, rng=rng)
        u = solve_poisson(f, grid_size)
        F[i] = f
        U[i] = u
        if (i + 1) % 200 == 0:
            print(f"  generated {i + 1}/{n_samples}")

    return F, U

def main():
    n = 5
    A = discrete_laplacian(n).toarray()
    f = random_source_field(n)
    F = f[1:-1, 1:-1].flatten()
    print(
        f"Matrix A:\n{A}",
        f"\nf: \n{f}",
        f"\nf[1:-1, 1:-1]: \n{f[1:-1, 1:-1]}",
        f"\nF (flattened): \n{F}"
        #f"\nSolution u: \n{solve_poisson(f, n)}"
        )

if __name__ == "__main__":
    OUT_DIR = Path(__file__).resolve().parent.parent.parent / "data"
    grid_size = 64

    print("Generating training set...")
    F_train, U_train = generate_dataset(2000, grid_size=grid_size, n_modes=6, seed=0)

    print("Generating validation set...")
    F_val, U_val = generate_dataset(200, grid_size=grid_size, n_modes=6, seed=1)

    print("Generating in-distribution test set...")
    F_test, U_test = generate_dataset(200, grid_size=grid_size, n_modes=6, seed=2)

    print("Generating out-of-distribution ('hard') test set (sharper features)...")
    F_hard, U_hard = generate_dataset(200, grid_size=grid_size, n_modes=14, seed=3)

    np.savez(OUT_DIR / "train.npz", f=F_train, u=U_train)
    np.savez(OUT_DIR / "val.npz", f=F_val, u=U_val)
    np.savez(OUT_DIR / "test.npz", f=F_test, u=U_test)
    np.savez(OUT_DIR / "test_hard.npz", f=F_hard, u=U_hard)

    print("Done. Saved train.npz, val.npz, test.npz, test_hard.npz")

    



    

    
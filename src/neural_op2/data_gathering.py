import numpy as np
import matplotlib.pyplot as plt
from sympy.abc import x, y
from scipy.sparse.linalg import cg, spsolve
import torch
import torch.nn as nn
from scipy.sparse import identity, kron, diags

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

def solve_poisson(n, f):
    A = discrete_laplacian(n)
    f_int = f[1:-1, 1:-1].flatten()
    u_int = spsolve(A, f_int) # spsolve uses LU decomposition
    u = np.zeros((n, n))
    u[1:-1, 1:-1] = u_int.reshape(n-2, n-2)
    return u

if __name__ == "__main__":
    n = 4
    A = discrete_laplacian(n).toarray()
    f = random_source_field(n)
    print(
        f"Matrix A:\n{A}",
        f"\nf: \n{f}",
        f"\nf[1:-1, 1:-1]: \n{f[1:-1, 1:-1]}"
        )
    



    

    
import numpy as np
import matplotlib.pyplot as plt
from sympy.abc import x, y
from scipy.sparse.linalg import cg
import torch
import torch.nn as nn

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

def build_laplacian_matrix(n):
    """
    Build the discrete Laplacian matrix for a 2D grid using the finite 
    difference analog of the poisson equation.
    
    Args:
        n: Grid size (n x n) where n > 2
    
    Returns:
        L: Discrete Laplacian matrix of shape (n^2, n^2)
    """
    n_int = (n-2)**2
    m_int = int(np.sqrt(n_int))
    L = np.zeros((n_int, n_int))

    # Diagonals of L
    # Main diagonal
    L[np.arange(n_int), np.arange(n_int)] = -4
    # Off-diagonals
    L[np.arange(n_int-1), np.arange(1, n_int)] = 1
    L[np.arange(1, n_int), np.arange(n_int-1)] = 1

    

    return L

if __name__ == "__main__":
    L = build_laplacian_matrix(5)
    print(L)



    

    
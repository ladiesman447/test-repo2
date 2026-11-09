import numpy as np
import torch
import matplotlib.pyplot as plt
import time

from neuralop.models import FNO
from src.neural_op2.data_gathering import solve_poisson
from src.neural_op2.pinn import train_PINN, evaluate_pinn_on_grid
from utils_.utils import relative_l2_error

DEVICE = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

def benchmark_classical_solver(F_test, U_test, grid_size):
    delta_classical = []
    times = []
    
    for i in range(len(F_test)):
        f = F_test[i]
        u = U_test[i]

        start_time = time.time()
        u_pred = solve_poisson(f, grid_size)
        end_time = time.time() - start_time

        times.append(end_time)

        delta_classical.append(relative_l2_error(u_pred, u))

    return delta_classical, times

def benchmark_PINN(F_test, U_test, grid_size):
    delta_pinn = []
    times = []
    
    pass

def benchmark_HardConstrainedPINN():
    pass

def benchmark_FNO():
    pass
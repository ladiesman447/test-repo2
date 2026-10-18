import numpy as np
import torch
import torch.nn as nn
from torch.autograd import grad
import time

DEVICE = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

class PINN(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x, y):
        input = torch.stack([x, y], dim=1)
        return self.net(input)

def PDE_loss(model, x, y, f):
    u = model(x, y)
    u_x = grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    u_y = grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_yy = grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
    return torch.mean((u_xx + u_yy - f) ** 2)

def interpolate_f_grid(f_grid):
    # TODO: Implement interpolation logic
    pass

def train_PINN(f_grid, n_iters=2000, n_collocation=1000, n_boundary=200, lr=1e-3, device="cpu") -> PINN:
    # Example training loop (not fully implemented)
    f_grid = f_grid.to(device)
    model = PINN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    t0 = time.time()
    for epoch in range(n_iters):
        optimizer.zero_grad()

        # loss inside boundary
        y_pred = model(x_train, y_train)
        data_loss = criterion(y_pred, y_train)
        pde_loss_val = PDE_loss(model, x_train, y_train, y_train)
        loss = data_loss + pde_loss_val
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

    training_time = time.time() - t0
    return model, training_time

if __name__ == "__main__":
    print(f"Using {DEVICE} device")
    # TESTING
    test_data = np.load("data/train.npz")
    F_test = test_data["f"]
    f_grid = torch.from_numpy(F_test[0]).float()

    model, training_time = train_PINN(f_grid=f_grid, n_iters=15, device=DEVICE)
    
    x = torch.rand(5, 1)  # 5 samples, 1D input
    y = torch.rand(5, 1)  # 5 samples, 1D output
    output = torch.cat([x, y], axis=1) # don't whether to use torch.cat or torch.stack
    print(output.shape)  # Should be (5, 2)
    print(f"Training time: {training_time:.2f} seconds")
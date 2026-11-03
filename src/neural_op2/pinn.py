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
    
    def forward(self, xy):
        return self.net(xy)

class HardConstrainedPINN(nn.Module):
    def __init__(self, hidden_dim=64, constant=0.0):
        super().__init__()
        self.constant = constant
        self.net = PINN(hidden_dim)
    
    def forward(self, xy):
        # Hard constraints: u(0, y) = 0, u(1, y) = 0, u(x, 0) = 0, u(x, 1) = 0
        x, y = xy[:, 0], xy[:, 1]
        d = x * (1 - x) * y * (1 - y)
        u = self.net(xy)
        c = self.constant
        return c + d * u 

def PDE_loss(model, xy, f):
    u = model(xy)
    u_x = grad(u, xy[:, 0], grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = grad(u_x, xy[:, 0], grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    u_y = grad(u, xy[:, 1], grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_yy = grad(u_y, xy[:, 1], grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
    return torch.mean((u_xx + u_yy - f) ** 2)

def interpolate_f_grid(f_grid, xy):
    H, W = f_grid.shape
    grid = xy.clone() * 2 - 1        # normalize to [-1, 1]
    grid = grid[:, [1, 0]]           # grid_sample wants (col_norm, row_norm) = (y_norm, x_norm)
    grid = grid.view(1, -1, 1, 2)

    f_input = f_grid.view(1, 1, H, W)
    sampled = F.grid_sample(f_input, grid, align_corners=True, mode="bilinear")
    return sampled.view(-1)

def train_PINN(f_grid, n_iters=2000, n_collocation=1000, n_boundary=200, lr=1e-3, device="cpu"):
    f_grid = f_grid.to(device)
    model = PINN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    t0 = time.time()
    for epoch in range(n_iters):
        model.train()
        optimizer.zero_grad()
        
        x = torch.rand(n_collocation, 1, device=device)
        y = torch.rand(n_collocation, 1, device=device)
        xy = torch.cat([x, y], dim=1)
        f_interpolated = interpolate_f_grid(f_grid, xy)

        u_pred = model(xy)

        pde_loss_val = PDE_loss(u_pred, xy, f_interpolated)

        # Boundary 1: y = 0, x in [0, 1]
        # Boundary 2: x = 1, y in [0, 1]
        # Boundary 3: y = 1, x in [0, 1]
        # Boundary 4: x = 0, y in [0, 1]

        # boundary variable is written as such for each boundary [x,y]
        edge = torch.rand(n_boundary // 4, 1, device=device)
        boundary = [
            torch.cat([edge, torch.zeros_like(edge)], dim=1),
            torch.cat([torch.ones_like(edge), edge], dim=1),
            torch.cat([edge, torch.ones_like(edge)], dim=1),
            torch.cat([torch.zeros_like(edge), edge], dim=1),
        ]
        xy_boundary = torch.cat(boundary, dim=0)
        boundary_pred = model(xy_boundary)
        boundary_loss = criterion(boundary_pred)

        loss = pde_loss_val + boundary_loss
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

    training_time = time.time() - t0
    return model, training_time

def train_hard_constrained_PINN(f_grid, n_iters=2000, n_collocation=1000, lr=1e-3, device="cpu"):
    f_grid = f_grid.to(device)
    model = HardConstrainedPINN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    t0 = time.time()
    for epoch in range(n_iters):
        model.train()
        optimizer.zero_grad()
        
        x = torch.rand(n_collocation, 1, device=device)
        y = torch.rand(n_collocation, 1, device=device)
        xy = torch.cat([x, y], dim=1)
        f_interpolated = interpolate_f_grid(f_grid, xy)

        u_pred = model(xy)

        loss = PDE_loss(u_pred, xy, f_interpolated)

        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

    training_time = time.time() - t0
    return model, training_time

@torch.no_grad()
def evaluate_pinn_on_grid(model, grid_size, device="cpu"):
    """Evaluate a trained PINN on a uniform grid_size x grid_size grid."""
    x = torch.linspace(0, 1, grid_size, device=device)
    y = torch.linspace(0, 1, grid_size, device=device)
    X, Y = torch.meshgrid(x, y, indexing="ij")
    xy = torch.stack([X.flatten(), Y.flatten()], dim=1)

    u = model(xy).view(grid_size, grid_size)
    return u.cpu()

if __name__ == "__main__":
    print(f"Using {DEVICE} device")
    # TESTING
    #print(torch.__version__)
    test_data = np.load("data/train_data.npz")
    F_test = test_data["f"]
    f_grid = torch.from_numpy(F_test[0]).float()
    #print(F_test)
    n_collocation = 4
    x = torch.rand(n_collocation, 1)
    y = torch.rand(n_collocation, 1)
    xy = torch.cat([x, y], dim=1)
    print(f'xy torch.cat([x, y], dim=1): {xy}')

    #model, training_time = train_PINN(f_grid=f_grid, n_iters=15, device=DEVICE)
    
    # x = torch.rand(5, 1)  # 5 samples, 1D input
    # y = torch.rand(5, 1)  # 5 samples, 1D output
    # output = torch.cat([x, y], axis=1) # don't whether to use torch.cat or torch.stack
    # print(output.shape)  # Should be (5, 2)
    # print(f"Training time: {training_time:.2f} seconds")
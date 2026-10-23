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

def PDE_loss(model, xy, f):
    u = model(xy)
    u_x = grad(u, xy[:, 0], grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = grad(u_x, xy[:, 0], grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    u_y = grad(u, xy[:, 1], grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_yy = grad(u_y, xy[:, 1], grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
    return torch.mean((u_xx + u_yy - f) ** 2)

def interpolate_f_grid(f_grid, xy):
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
        
        x = torch.rand(n_collocation, 1, device=device)
        y = torch.rand(n_collocation, 1, device=device)
        xy = torch.cat([x, y], dim=1)
        f_interpolated = interpolate_f_grid(f_grid, xy)

        u_pred = model(xy)

        pde_loss_val = PDE_loss(model, xy, f_interpolated)

        edge = torch.rand(n_boundary // 4, 1, device=device)

        # Boundary 1: y = 0, x in [0, 1]
        # Boundary 2: y = 1, x in [0, 1]
        # Boundary 3: x = 0, y in [0, 1]
        # Boundary 4: x = 1, y in [0, 1]
        # boundary variable is written as such for each boundary [x,y]
        boundary = [
            torch.cat([edge, torch.zeros_like(edge)], dim=1),
            torch.cat([edge, torch.ones_like(edge)], dim=1),
            torch.cat([torch.zeros_like(edge), edge], dim=1),
            torch.cat([torch.ones_like(edge), edge], dim=1),
        ]

        boundary_pred = model(torch.cat(boundary, dim=0))

        
        loss = pde_loss_val
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

    training_time = time.time() - t0
    return model, training_time

def train_hard_constrained_PINN():
    # TODO: Implement hard constrained PINN training i.e. forcing boundary to be zero
    pass

if __name__ == "__main__":
    print(f"Using {DEVICE} device")
    # TESTING
    #print(torch.__version__)
    test_data = np.load("data/train_data.npz")
    F_test = test_data["f"]
    f_grid = torch.from_numpy(F_test[0]).float()
    #print(F_test)
    x = torch.rand(3, 1)
    y = torch.rand(3, 1)
    x1 = torch.zeros_like(x)
    y1 = torch.rand_like(y)
    dim = 1
    xy = torch.cat([x, y], dim=dim)
    print(f'x: {x}')
    print(f'y: {y}')
    print(f'xy with dim={dim}: {xy}')
    print(f'xy shape: {xy.shape}')
    xy1 = torch.cat([x1, y1], dim=dim)
    XY = torch.cat([xy, xy1], dim=0)
    print(f'XY: {XY}')

    #model, training_time = train_PINN(f_grid=f_grid, n_iters=15, device=DEVICE)
    
    # x = torch.rand(5, 1)  # 5 samples, 1D input
    # y = torch.rand(5, 1)  # 5 samples, 1D output
    # output = torch.cat([x, y], axis=1) # don't whether to use torch.cat or torch.stack
    # print(output.shape)  # Should be (5, 2)
    # print(f"Training time: {training_time:.2f} seconds")
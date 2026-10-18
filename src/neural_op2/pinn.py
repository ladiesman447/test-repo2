import torch
import torch.nn as nn
from torch.autograd import grad

class PINN(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, output_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x, y):
        input = torch.stack([x, y], dim=1)
        return self.net(input)

def PDE_loss(model, x, y, f):
    u = model(x, y)
    u_x = grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    return torch.mean((u_xx - f) ** 2)

def train_PINN():
    # Example training loop (not fully implemented)
    model = PINN()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    # Dummy data for demonstration
    x_train = torch.rand(100, 2)  # 100 samples, 2D input
    y_train = torch.rand(100, 1)  # 100 samples, 1D output

    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        y_pred = model(x_train, y_train)
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

if __name__ == "__main__":
    # Example usage
    model = PINN()
    x = torch.rand(5, 1)  # 5 samples, 1D input
    y = torch.rand(5, 1)  # 5 samples, 1D output
    output = torch.cat([x, y], axis=1) # don't whether to use torch.cat or torch.stack
    print(output.shape)  # Should be (5, 2)
from neuralop.models import FNO
from utils_.utils import PoissonDataset, relative_l2_error
from torch.utils.data import DataLoader
import torch
import time
import json
from pathlib import Path
import argparse

def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    data_dir = Path(args.data_dir)
    train_ds = PoissonDataset(data_dir / "train.npz")
    val_ds = PoissonDataset(data_dir / "val.npz")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = FNO(
        in_channels=3, out_channels=1,
        hidden_channels=args.hidden, n_modes=(args.modes, args.modes),
        n_layers=args.n_layers,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=max(1, args.epochs // 4), gamma=0.5
    )

    history = {"train_loss": [], "val_loss": [], "epoch_time": []}
    best_val_loss = float("inf")

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        model.train()
        train_loss = 0.0
        for f, u in train_loader:
            f, u = f.to(device), u.to(device)
            optimizer.zero_grad()
            u_pred = model(f)
            loss = relative_l2_error(u_pred, u)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * f.shape[0]
        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for f, u in val_loader:
                f, u = f.to(device), u.to(device)
                u_pred = model(f)
                val_loss += relative_l2_error(u_pred, u).item() * f.shape[0]
        val_loss /= len(val_ds)

        scheduler.step()
        epoch_time = time.time() - t0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["epoch_time"].append(epoch_time)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {"model_state": model.state_dict(), "config": vars(args), "val_loss": val_loss},
                ckpt_dir / "fno_best.pt",
            )

        if epoch % args.log_every == 0 or epoch == 1:
            print(f"epoch {epoch:4d} | train {train_loss:.4e} | val {val_loss:.4e} | {epoch_time:.2f}s")

    with open(ckpt_dir / "history.json", "w") as fh:
        json.dump(history, fh, indent=2)

    print(f"\nBest val relative-L2: {best_val_loss:.4e}")
    print(f"Checkpoint saved to {ckpt_dir / 'fno_best.pt'}")
    print(f"Total training time: {sum(history['epoch_time']):.1f}s")

    return history

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="data")
    p.add_argument("--ckpt_dir", type=str, default="checkpoints")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--modes", type=int, default=12)
    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument("--log_every", type=int, default=5)
    return p.parse_args()


if __name__ == "__main__":
    model = FNO(n_modes=(12, 12), in_channels=3, out_channels=1, hidden_channels=32, n_layers=4)
    print(model)

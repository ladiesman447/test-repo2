import numpy as np
import torch
from torch.utils.data import Dataset


class PoissonDataset(Dataset):
    """Wraps an (f, u) .npz file as a PyTorch Dataset."""

    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.f = torch.from_numpy(data["f"]).float()
        self.u = torch.from_numpy(data["u"]).float()

    def __len__(self):
        return self.f.shape[0]

    def __getitem__(self, idx):
        return self.f[idx], self.u[idx]

def relative_l2_error(pred, true, eps=1e-8):
    """
    Per-sample relative L2 error, averaged over the batch:
        ||pred - true||_2 / ||true||_2

    This is the standard metric in the operator-learning literature
    (scale-invariant, comparable across resolutions). Used both as the
    training loss and as the Phase 4 accuracy metric.
    """
    pred_flat = pred.reshape(pred.shape[0], -1)
    true_flat = true.reshape(true.shape[0], -1)

    num = torch.norm(pred_flat - true_flat, dim=1)
    den = torch.norm(true_flat, dim=1) + eps

    return (num / den).mean()
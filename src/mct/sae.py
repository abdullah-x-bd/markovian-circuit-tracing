from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


@dataclass
class SAEResult:
    reconstruction_mse: float
    mean_l1: float
    active_fraction: float


class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, top_k: int | None = None):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)
        self.top_k = top_k

    def _apply_top_k(self, z: torch.Tensor) -> torch.Tensor:
        if self.top_k is None or self.top_k <= 0 or self.top_k >= z.shape[-1]:
            return z
        values, indexes = torch.topk(z, k=self.top_k, dim=-1)
        sparse = torch.zeros_like(z)
        sparse.scatter_(dim=-1, index=indexes, src=values)
        return sparse

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z = torch.relu(self.encoder(x))
        return self._apply_top_k(z)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        recon = self.decode(z)
        return recon, z


def flatten_activations(activations: np.ndarray, max_samples: int | None = None, seed: int = 0) -> np.ndarray:
    flat = activations.reshape(-1, activations.shape[-1]).astype(np.float32)
    if max_samples is not None and flat.shape[0] > max_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(flat.shape[0], size=max_samples, replace=False)
        flat = flat[idx]
    return flat


def train_sae(
    activations: np.ndarray,
    hidden_dim: int = 256,
    l1_coef: float = 1e-3,
    epochs: int = 5,
    batch_size: int = 512,
    lr: float = 1e-3,
    max_samples: int | None = 50000,
    seed: int = 0,
    top_k: int | None = None,
) -> tuple[SparseAutoencoder, SAEResult]:
    torch.manual_seed(seed)
    flat = flatten_activations(activations, max_samples=max_samples, seed=seed)
    mean = flat.mean(axis=0, keepdims=True)
    std = flat.std(axis=0, keepdims=True) + 1e-6
    flat = (flat - mean) / std

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SparseAutoencoder(input_dim=flat.shape[-1], hidden_dim=hidden_dim, top_k=top_k).to(device)
    model.register_buffer("activation_mean", torch.tensor(mean, dtype=torch.float32, device=device))
    model.register_buffer("activation_std", torch.tensor(std, dtype=torch.float32, device=device))

    data = torch.from_numpy(flat)
    loader = DataLoader(TensorDataset(data), batch_size=batch_size, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    for _ in range(epochs):
        model.train()
        for (xb,) in tqdm(loader, leave=False):
            xb = xb.to(device)
            recon, z = model(xb)
            recon_loss = torch.mean((recon - xb) ** 2)
            l1_loss = z.abs().mean()
            loss = recon_loss + l1_coef * l1_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        all_z = []
        all_recon = []
        for (xb,) in DataLoader(TensorDataset(data), batch_size=batch_size):
            xb = xb.to(device)
            recon, z = model(xb)
            all_z.append(z.cpu())
            all_recon.append(recon.cpu())
        zcat = torch.cat(all_z, dim=0)
        rcat = torch.cat(all_recon, dim=0)
        recon_mse = torch.mean((rcat - data) ** 2).item()
        mean_l1 = zcat.abs().mean().item()
        active_fraction = (zcat > 1e-6).float().mean().item()

    return model, SAEResult(
        reconstruction_mse=float(recon_mse),
        mean_l1=float(mean_l1),
        active_fraction=float(active_fraction),
    )


def encode_activations(
    sae: SparseAutoencoder,
    activations: np.ndarray,
    batch_size: int = 1024,
) -> np.ndarray:
    device = next(sae.parameters()).device
    flat = activations.reshape(-1, activations.shape[-1]).astype(np.float32)
    encoded = []
    sae.eval()
    with torch.no_grad():
        for start in range(0, flat.shape[0], batch_size):
            xb = torch.from_numpy(flat[start : start + batch_size]).to(device)
            xb = (xb - sae.activation_mean) / sae.activation_std
            z = sae.encode(xb)
            encoded.append(z.cpu().numpy())
    zflat = np.concatenate(encoded, axis=0)
    return zflat.reshape(*activations.shape[:-1], zflat.shape[-1])

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


@dataclass
class TrainResult:
    train_loss: list[float]
    val_loss: list[float]


def evaluate_loss(model: nn.Module, x: torch.Tensor, y: torch.Tensor, batch_size: int = 256) -> float:
    model.eval()
    losses = []
    loader = DataLoader(TensorDataset(x, y), batch_size=batch_size)
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(next(model.parameters()).device)
            yb = yb.to(next(model.parameters()).device)
            logits = model(xb)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                yb.reshape(-1),
            )
            losses.append(float(loss.detach().cpu()))
    return sum(losses) / len(losses)


def train_model(
    model: nn.Module,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    epochs: int = 5,
    batch_size: int = 256,
    lr: float = 3e-4,
) -> TrainResult:
    device = next(model.parameters()).device
    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    train_losses = []
    val_losses = []

    for _ in range(epochs):
        model.train()
        running = []
        for xb, yb in tqdm(loader, leave=False):
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                yb.reshape(-1),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running.append(float(loss.detach().cpu()))

        train_losses.append(sum(running) / len(running))
        val_losses.append(evaluate_loss(model, val_x, val_y, batch_size=batch_size))

    return TrainResult(train_loss=train_losses, val_loss=val_losses)


def collect_activations(model: nn.Module, x: torch.Tensor, activation_name: str, batch_size: int = 256) -> torch.Tensor:
    model.eval()
    outs = []
    loader = DataLoader(TensorDataset(x), batch_size=batch_size)
    device = next(model.parameters()).device
    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device)
            _, acts = model(xb, return_activations=True)
            outs.append(acts[activation_name].detach().cpu())
    return torch.cat(outs, dim=0)

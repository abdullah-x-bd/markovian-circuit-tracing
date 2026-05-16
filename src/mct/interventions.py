from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


def kl_divergence_torch(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    return torch.sum(p * (torch.log(p) - torch.log(q)), dim=-1)


def state_forcing_kl(
    model: torch.nn.Module,
    x: torch.Tensor,
    centroids: np.ndarray,
    ideal_next_token_distributions: np.ndarray,
    activation_name: str,
    position: int,
    batch_size: int = 256,
) -> dict[str, float]:
    device = next(model.parameters()).device
    model.eval()
    results = {}
    loader = DataLoader(TensorDataset(x), batch_size=batch_size)

    for state_id in range(centroids.shape[0]):
        forced = torch.tensor(centroids[state_id], dtype=torch.float32, device=device)
        ideal = torch.tensor(
            ideal_next_token_distributions[state_id],
            dtype=torch.float32,
            device=device,
        )
        losses = []
        with torch.no_grad():
            for (xb,) in loader:
                xb = xb.to(device)
                logits = model(
                    xb,
                    intervention={
                        "name": activation_name,
                        "position": position,
                        "value": forced,
                    },
                )
                pred = F.softmax(logits[:, position, :], dim=-1)
                target = ideal.unsqueeze(0).expand_as(pred)
                losses.append(kl_divergence_torch(pred, target).mean().item())
        results[f"forced_state_{state_id}_kl"] = float(np.mean(losses))

    return results


def causal_faithfulness_score(original_kl: float, patched_kl: float, eps: float = 1e-9) -> float:
    return float(1.0 - patched_kl / max(original_kl, eps))

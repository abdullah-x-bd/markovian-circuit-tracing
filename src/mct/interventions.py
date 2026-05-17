from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


def kl_divergence_torch(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    return torch.sum(p * (torch.log(p) - torch.log(q)), dim=-1)


def _distribution_kl_for_logits(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = F.softmax(logits, dim=-1)
    expanded = target.unsqueeze(0).expand_as(pred)
    return kl_divergence_torch(pred, expanded)


def _run_patch_kl(
    model: torch.nn.Module,
    x: torch.Tensor,
    target_distribution: np.ndarray,
    activation_name: str,
    position: int,
    patch_value: np.ndarray | None,
    batch_size: int,
) -> float:
    device = next(model.parameters()).device
    target = torch.tensor(target_distribution, dtype=torch.float32, device=device)
    loader = DataLoader(TensorDataset(x), batch_size=batch_size)
    losses = []
    model.eval()
    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device)
            if patch_value is None:
                logits = model(xb)
            else:
                forced = torch.tensor(patch_value, dtype=torch.float32, device=device)
                logits = model(
                    xb,
                    intervention={
                        "name": activation_name,
                        "position": position,
                        "value": forced,
                    },
                )
            losses.append(_distribution_kl_for_logits(logits[:, position, :], target).mean().item())
    return float(np.mean(losses))


def state_forcing_kl(
    model: torch.nn.Module,
    x: torch.Tensor,
    centroids: np.ndarray,
    ideal_next_token_distributions: np.ndarray,
    activation_name: str,
    position: int,
    batch_size: int = 256,
) -> dict[str, float]:
    results = {}
    for state_id in range(centroids.shape[0]):
        results[f"forced_state_{state_id}_kl"] = _run_patch_kl(
            model,
            x,
            ideal_next_token_distributions[state_id],
            activation_name,
            position,
            patch_value=centroids[state_id],
            batch_size=batch_size,
        )
    return results


def state_forcing_control_report(
    model: torch.nn.Module,
    x: torch.Tensor,
    recovered_centroids: np.ndarray,
    true_centroids: np.ndarray,
    ideal_next_token_distributions: np.ndarray,
    activation_name: str,
    position: int,
    seed: int = 0,
    batch_size: int = 256,
) -> list[dict[str, float | int | str]]:
    """Evaluate state-forcing with reviewer-facing controls.

    For each target state k, this compares the unpatched model, the recovered
    centroid for k, a wrong recovered centroid, a mean activation patch, a
    random Gaussian patch, a shuffled-label centroid, and a true-state centroid
    oracle. Lower KL means closer to the exact forced-state HMM target.
    """
    rng = np.random.default_rng(seed)
    n_states = recovered_centroids.shape[0]
    mean_activation = recovered_centroids.mean(axis=0)
    std_activation = recovered_centroids.std(axis=0) + 1e-6
    shuffled_indexes = rng.permutation(n_states)
    rows: list[dict[str, float | int | str]] = []

    for target_state in range(n_states):
        target_distribution = ideal_next_token_distributions[target_state]
        wrong_state = (target_state + 1) % n_states
        random_patch = rng.normal(mean_activation, std_activation).astype(np.float32)

        patches: dict[str, np.ndarray | None] = {
            "unpatched": None,
            "recovered_centroid": recovered_centroids[target_state],
            "wrong_recovered_centroid": recovered_centroids[wrong_state],
            "mean_activation": mean_activation,
            "random_activation": random_patch,
            "shuffled_label_centroid": recovered_centroids[shuffled_indexes[target_state]],
            "true_state_centroid_oracle": true_centroids[target_state],
        }

        unpatched_kl = None
        for patch_name, patch_value in patches.items():
            kl_value = _run_patch_kl(
                model,
                x,
                target_distribution,
                activation_name,
                position,
                patch_value=patch_value,
                batch_size=batch_size,
            )
            if patch_name == "unpatched":
                unpatched_kl = kl_value
            improvement = 0.0 if unpatched_kl is None else unpatched_kl - kl_value
            rows.append(
                {
                    "target_state": target_state,
                    "patch_type": patch_name,
                    "kl_to_target": kl_value,
                    "improvement_over_unpatched": improvement,
                }
            )
    return rows


def causal_faithfulness_score(original_kl: float, patched_kl: float, eps: float = 1e-9) -> float:
    return float(1.0 - patched_kl / max(original_kl, eps))

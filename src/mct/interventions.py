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
    """Return KL(target || model) for the categorical output distribution."""
    pred = F.softmax(logits, dim=-1)
    expanded = target.unsqueeze(0).expand_as(pred)
    return kl_divergence_torch(expanded, pred)


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
    losses = []
    model.eval()
    with torch.no_grad():
        for (xb,) in DataLoader(TensorDataset(x), batch_size=batch_size):
            xb = xb.to(device)
            if patch_value is None:
                logits = model(xb)
            else:
                forced = torch.tensor(patch_value, dtype=torch.float32, device=device)
                logits = model(
                    xb,
                    intervention={"name": activation_name, "position": position, "value": forced},
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
    """Backward-compatible minimal forcing report."""
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
    target_distributions: np.ndarray,
    activation_name: str,
    position: int,
    seed: int = 0,
    batch_size: int = 256,
) -> list[dict[str, float | int | str]]:
    """Evaluate current-state forcing with reviewer-facing controls.

    target_distributions must represent P(x_t | s_t=k) for a same-position
    intervention. The benchmark passes E[k], not T[k] @ E.
    """
    rng = np.random.default_rng(seed)
    n_states = recovered_centroids.shape[0]
    mean_activation = recovered_centroids.mean(axis=0)
    std_activation = recovered_centroids.std(axis=0) + 1e-6
    shuffled_indexes = rng.permutation(n_states)
    rows: list[dict[str, float | int | str]] = []
    for target_state in range(n_states):
        target = target_distributions[target_state]
        wrong_state = (target_state + 1) % n_states
        patches: dict[str, np.ndarray | None] = {
            "unpatched": None,
            "recovered_centroid": recovered_centroids[target_state],
            "wrong_recovered_centroid": recovered_centroids[wrong_state],
            "mean_activation": mean_activation,
            "random_activation": rng.normal(mean_activation, std_activation).astype(np.float32),
            "shuffled_label_centroid": recovered_centroids[shuffled_indexes[target_state]],
            "true_state_centroid_oracle": true_centroids[target_state],
        }
        baseline = None
        for patch_name, patch_value in patches.items():
            kl_value = _run_patch_kl(
                model,
                x,
                target,
                activation_name,
                position,
                patch_value,
                batch_size,
            )
            if patch_name == "unpatched":
                baseline = kl_value
            rows.append(
                {
                    "target_state": target_state,
                    "patch_type": patch_name,
                    "kl_to_target": kl_value,
                    "improvement_over_unpatched": 0.0 if baseline is None else baseline - kl_value,
                }
            )
    return rows


def _pairwise_patch_kl(
    model: torch.nn.Module,
    x: torch.Tensor,
    donor_values: np.ndarray,
    activation_name: str,
    position: int,
) -> np.ndarray:
    device = next(model.parameters()).device
    xb = x.to(device)
    donors = torch.tensor(donor_values, dtype=torch.float32, device=device)
    model.eval()
    with torch.no_grad():
        base_logits = model(xb)[:, position, :]
        patched_logits = model(
            xb,
            intervention={"name": activation_name, "position": position, "value": donors},
        )[:, position, :]
        p = F.softmax(base_logits, dim=-1)
        q = F.softmax(patched_logits, dim=-1)
        return kl_divergence_torch(p, q).cpu().numpy()


def causal_scrubbing_report(
    model: torch.nn.Module,
    x: torch.Tensor,
    activations: np.ndarray,
    recovered_states: np.ndarray,
    activation_name: str,
    position: int,
    seed: int = 0,
    max_pairs: int = 256,
) -> dict[str, float]:
    """Compare same-state and different-state activation swaps on held-out sequences."""
    rng = np.random.default_rng(seed)
    states = recovered_states[:, position]
    acts = activations[:, position, :]
    n = min(len(states), max_pairs)
    recipients = rng.choice(len(states), size=n, replace=False)
    same_donors: list[int] = []
    diff_donors: list[int] = []
    valid_recipients: list[int] = []
    for i in recipients:
        same = np.where((states == states[i]) & (np.arange(len(states)) != i))[0]
        diff = np.where(states != states[i])[0]
        if len(same) == 0 or len(diff) == 0:
            continue
        valid_recipients.append(int(i))
        same_donors.append(int(rng.choice(same)))
        diff_donors.append(int(rng.choice(diff)))
    if not valid_recipients:
        raise ValueError("No valid donor pairs for causal scrubbing")
    rec = np.array(valid_recipients, dtype=np.int64)
    same_kl = _pairwise_patch_kl(
        model, x[rec], acts[np.array(same_donors)], activation_name, position
    )
    diff_kl = _pairwise_patch_kl(
        model, x[rec], acts[np.array(diff_donors)], activation_name, position
    )
    return {
        "scrubbing_pairs": int(len(rec)),
        "same_state_swap_kl_mean": float(same_kl.mean()),
        "different_state_swap_kl_mean": float(diff_kl.mean()),
        "different_minus_same_kl": float(diff_kl.mean() - same_kl.mean()),
        "same_state_swap_kl_median": float(np.median(same_kl)),
        "different_state_swap_kl_median": float(np.median(diff_kl)),
    }


def causal_faithfulness_score(original_kl: float, patched_kl: float, eps: float = 1e-9) -> float:
    return float(1.0 - patched_kl / max(original_kl, eps))

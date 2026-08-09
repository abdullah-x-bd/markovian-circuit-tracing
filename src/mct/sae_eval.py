from __future__ import annotations

import numpy as np

from mct.probes import belief_probe_metrics
from mct.sae import encode_activations, train_sae
from mct.states import fit_state_abstraction
from mct.transition import estimate_transition_matrix, transition_report


def run_sae_evaluation(
    *,
    calibration_activations: np.ndarray,
    selection_activations: np.ndarray,
    evaluation_activations: np.ndarray,
    calibration_states: np.ndarray,
    selection_states: np.ndarray,
    evaluation_states: np.ndarray,
    calibration_beliefs: np.ndarray,
    evaluation_beliefs: np.ndarray,
    true_transition: np.ndarray,
    seed: int,
    epochs: int,
    max_samples: int,
    batch_size: int,
    run_sweep: bool,
    hidden_dim: int,
    l1_coef: float,
    top_k: int,
    sweep_hidden_dims: str,
    sweep_l1_coefs: str,
    sweep_top_ks: str,
) -> tuple[dict[str, object], np.ndarray]:
    n_states = true_transition.shape[0]

    def candidate(h: int, l1: float, k: int, target: str, candidate_seed: int):
        topk = None if k <= 0 else k
        sae, result = train_sae(
            calibration_activations,
            hidden_dim=h,
            l1_coef=l1,
            epochs=epochs,
            batch_size=max(batch_size, 256),
            max_samples=max_samples,
            seed=candidate_seed,
            top_k=topk,
        )
        cal_z = encode_activations(sae, calibration_activations)
        target_acts = selection_activations if target == "selection" else evaluation_activations
        target_states = selection_states if target == "selection" else evaluation_states
        target_z = encode_activations(sae, target_acts)
        abstraction = fit_state_abstraction(cal_z, calibration_states, n_states=n_states, seed=candidate_seed)
        recovered = abstraction.predict(target_z)
        estimated_t = estimate_transition_matrix(recovered, n_states=n_states)
        return result, cal_z, target_z, recovered, estimated_t, transition_report(true_transition, estimated_t)

    if run_sweep:
        hs = [int(x) for x in sweep_hidden_dims.split(",") if x.strip()]
        l1s = [float(x) for x in sweep_l1_coefs.split(",") if x.strip()]
        ks = [int(x) for x in sweep_top_ks.split(",") if x.strip()]
        rows = []
        best = None
        for h in hs:
            for l1 in l1s:
                for k in ks:
                    candidate_seed = seed + h + int(l1 * 1_000_000) + k
                    result, _, _, _, _, report = candidate(h, l1, k, "selection", candidate_seed)
                    row = {
                        "hidden_dim": h,
                        "l1_coef": l1,
                        "top_k": k,
                        "reconstruction_mse": result.reconstruction_mse,
                        "active_fraction": result.active_fraction,
                        **report,
                    }
                    rows.append(row)
                    if best is None or row["rowwise_kl"] < best["rowwise_kl"]:
                        best = row
        assert best is not None
        hidden_dim = int(best["hidden_dim"])
        l1_coef = float(best["l1_coef"])
        top_k = int(best["top_k"])
        candidate_seed = seed + hidden_dim + int(l1_coef * 1_000_000) + top_k
    else:
        rows = []
        best = {"hidden_dim": hidden_dim, "l1_coef": l1_coef, "top_k": top_k}
        candidate_seed = seed + 500

    result, cal_z, eval_z, recovered, estimated_t, report = candidate(
        hidden_dim, l1_coef, top_k, "evaluation", candidate_seed
    )
    heldout = {
        "hidden_dim": hidden_dim,
        "l1_coef": l1_coef,
        "top_k": top_k,
        "reconstruction_mse": result.reconstruction_mse,
        "active_fraction": result.active_fraction,
        **report,
        **belief_probe_metrics(cal_z, calibration_beliefs, eval_z, evaluation_beliefs, evaluation_states),
        "cluster_state_recovery_accuracy": float((recovered == evaluation_states).mean()),
    }
    metrics: dict[str, object] = {"heldout_evaluation": heldout}
    if run_sweep:
        metrics.update(
            {
                "selection_rule": "minimum selection-set transition rowwise KL",
                "sweep": rows,
                "best_hyperparameters": best,
            }
        )
    return metrics, estimated_t

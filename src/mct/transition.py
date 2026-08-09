from __future__ import annotations

import numpy as np
from scipy.linalg import eigvals


def estimate_transition_matrix(states: np.ndarray, n_states: int, smoothing: float = 1e-6) -> np.ndarray:
    counts = np.full((n_states, n_states), smoothing, dtype=np.float64)
    for seq in states:
        for a, b in zip(seq[:-1], seq[1:], strict=False):
            counts[int(a), int(b)] += 1.0
    return counts / counts.sum(axis=1, keepdims=True)


def rowwise_kl(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return float(np.mean(np.sum(p * (np.log(p) - np.log(q)), axis=1)))


def frobenius_error(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.linalg.norm(p - q, ord="fro"))


def stationary_distribution(transition: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eig(transition.T)
    idx = int(np.argmin(np.abs(values - 1.0)))
    vec = np.abs(np.real(vectors[:, idx]))
    return vec / vec.sum()


def stationary_l1(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.abs(stationary_distribution(p) - stationary_distribution(q)).sum())


def spectral_error(p: np.ndarray, q: np.ndarray) -> float:
    ep = np.sort_complex(eigvals(p))
    eq = np.sort_complex(eigvals(q))
    return float(np.abs(ep - eq).mean())


def transition_report(true_t: np.ndarray, estimated_t: np.ndarray) -> dict[str, float]:
    return {
        "rowwise_kl": rowwise_kl(true_t, estimated_t),
        "frobenius_error": frobenius_error(true_t, estimated_t),
        "stationary_l1": stationary_l1(true_t, estimated_t),
        "spectral_error": spectral_error(true_t, estimated_t),
    }


def _fit_order2(states: np.ndarray, n_states: int, smoothing: float = 1e-6) -> np.ndarray:
    counts = np.full((n_states, n_states, n_states), smoothing, dtype=np.float64)
    for seq in states:
        for a, b, c in zip(seq[:-2], seq[1:-1], seq[2:], strict=False):
            counts[int(a), int(b), int(c)] += 1.0
    return counts / counts.sum(axis=-1, keepdims=True)


def markov_order_report(
    calibration_states: np.ndarray,
    evaluation_states: np.ndarray,
    n_states: int,
) -> dict[str, float]:
    """Fit order-0/1/2 predictors on calibration trajectories and score held-out trajectories."""
    majority = int(np.bincount(calibration_states.reshape(-1), minlength=n_states).argmax())
    order0 = float((evaluation_states[:, 1:] == majority).mean())
    t1 = estimate_transition_matrix(calibration_states, n_states)
    pred1 = t1[evaluation_states[:, :-1]].argmax(axis=-1)
    order1 = float((pred1 == evaluation_states[:, 1:]).mean())
    t2 = _fit_order2(calibration_states, n_states)
    pred2 = t2[evaluation_states[:, :-2], evaluation_states[:, 1:-1]].argmax(axis=-1)
    order2 = float((pred2 == evaluation_states[:, 2:]).mean())
    return {
        "order0_accuracy": order0,
        "order1_accuracy": order1,
        "order2_accuracy": order2,
        "order1_gain_over_order0": order1 - order0,
        "order2_gain_over_order1": order2 - order1,
    }


def markov_order_accuracy(states: np.ndarray, n_states: int) -> dict[str, float]:
    """Backward-compatible in-sample helper. New benchmark code uses markov_order_report."""
    return markov_order_report(states, states, n_states)

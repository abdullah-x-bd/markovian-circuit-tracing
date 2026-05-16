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
    vec = np.real(vectors[:, idx])
    vec = np.abs(vec)
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


def markov_order_accuracy(states: np.ndarray, n_states: int) -> dict[str, float]:
    flat = states.reshape(-1)
    majority = np.bincount(flat, minlength=n_states).argmax()
    order0_correct = (states[:, 1:] == majority).mean()

    t1 = estimate_transition_matrix(states, n_states)
    pred1 = t1[states[:, :-1]].argmax(axis=-1)
    order1_correct = (pred1 == states[:, 1:]).mean()

    counts2 = np.ones((n_states, n_states, n_states), dtype=np.float64) * 1e-6
    for seq in states:
        for a, b, c in zip(seq[:-2], seq[1:-1], seq[2:], strict=False):
            counts2[int(a), int(b), int(c)] += 1
    probs2 = counts2 / counts2.sum(axis=-1, keepdims=True)
    pred2 = probs2[states[:, :-2], states[:, 1:-1]].argmax(axis=-1)
    order2_correct = (pred2 == states[:, 2:]).mean()

    return {
        "order0_accuracy": float(order0_correct),
        "order1_accuracy": float(order1_correct),
        "order2_accuracy": float(order2_correct),
        "order1_gain_over_order0": float(order1_correct - order0_correct),
        "order2_gain_over_order1": float(order2_correct - order1_correct),
    }

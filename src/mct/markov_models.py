from __future__ import annotations

import numpy as np


def split_sequences(states: np.ndarray, frac: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    n = states.shape[0]
    cut = max(1, int(n * frac))
    return states[:cut], states[cut:]


def fit_order0(states: np.ndarray, n_states: int, smoothing: float = 1e-6) -> np.ndarray:
    counts = np.full(n_states, smoothing, dtype=np.float64)
    counts += np.bincount(states[:, 1:].reshape(-1), minlength=n_states)
    return counts / counts.sum()


def fit_order1(states: np.ndarray, n_states: int, smoothing: float = 1e-6) -> np.ndarray:
    counts = np.full((n_states, n_states), smoothing, dtype=np.float64)
    for seq in states:
        for prev_state, next_state in zip(seq[:-1], seq[1:], strict=False):
            counts[int(prev_state), int(next_state)] += 1.0
    return counts / counts.sum(axis=1, keepdims=True)


def fit_order2(states: np.ndarray, n_states: int, smoothing: float = 1e-6) -> np.ndarray:
    counts = np.full((n_states, n_states, n_states), smoothing, dtype=np.float64)
    for seq in states:
        for a, b, c in zip(seq[:-2], seq[1:-1], seq[2:], strict=False):
            counts[int(a), int(b), int(c)] += 1.0
    return counts / counts.sum(axis=2, keepdims=True)


def nll_order0(model: np.ndarray, states: np.ndarray, eps: float = 1e-12) -> float:
    targets = states[:, 1:].reshape(-1)
    probs = np.clip(model[targets], eps, 1.0)
    return float(-np.log(probs).mean())


def nll_order1(model: np.ndarray, states: np.ndarray, eps: float = 1e-12) -> float:
    prev_states = states[:, :-1].reshape(-1)
    targets = states[:, 1:].reshape(-1)
    probs = np.clip(model[prev_states, targets], eps, 1.0)
    return float(-np.log(probs).mean())


def nll_order2(model: np.ndarray, states: np.ndarray, eps: float = 1e-12) -> float:
    a = states[:, :-2].reshape(-1)
    b = states[:, 1:-1].reshape(-1)
    c = states[:, 2:].reshape(-1)
    probs = np.clip(model[a, b, c], eps, 1.0)
    return float(-np.log(probs).mean())


def markov_nll_report(states: np.ndarray, n_states: int) -> dict[str, float]:
    train, test = split_sequences(states)
    if test.shape[0] == 0:
        test = train

    order0 = fit_order0(train, n_states)
    order1 = fit_order1(train, n_states)
    order2 = fit_order2(train, n_states)

    nll0 = nll_order0(order0, test)
    nll1 = nll_order1(order1, test)
    nll2 = nll_order2(order2, test)

    return {
        "markov_nll_order0": nll0,
        "markov_nll_order1": nll1,
        "markov_nll_order2": nll2,
        "markov_nll_gain_0_to_1": nll0 - nll1,
        "markov_nll_gain_1_to_2": nll1 - nll2,
    }


def bootstrap_markov_nll(
    states: np.ndarray,
    n_states: int,
    n_bootstrap: int = 200,
    seed: int = 0,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    gains_01 = []
    gains_12 = []
    n_seq = states.shape[0]
    for _ in range(n_bootstrap):
        idx = rng.choice(n_seq, size=n_seq, replace=True)
        report = markov_nll_report(states[idx], n_states)
        gains_01.append(report["markov_nll_gain_0_to_1"])
        gains_12.append(report["markov_nll_gain_1_to_2"])

    gains_01 = np.asarray(gains_01)
    gains_12 = np.asarray(gains_12)
    return {
        "markov_nll_gain_0_to_1_ci_low": float(np.quantile(gains_01, 0.025)),
        "markov_nll_gain_0_to_1_ci_high": float(np.quantile(gains_01, 0.975)),
        "markov_nll_gain_1_to_2_ci_low": float(np.quantile(gains_12, 0.025)),
        "markov_nll_gain_1_to_2_ci_high": float(np.quantile(gains_12, 0.975)),
    }

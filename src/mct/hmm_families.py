from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mct.data import HMM


@dataclass(frozen=True)
class HMMFamilyConfig:
    name: str
    n_states: int
    vocab_size: int
    description: str


FAMILIES: dict[str, HMMFamilyConfig] = {
    "easy_separable": HMMFamilyConfig(
        name="easy_separable",
        n_states=4,
        vocab_size=6,
        description="Low emission overlap, moderate transition structure.",
    ),
    "ambiguous_emissions": HMMFamilyConfig(
        name="ambiguous_emissions",
        n_states=4,
        vocab_size=6,
        description="High emission overlap, belief inference is needed.",
    ),
    "persistent": HMMFamilyConfig(
        name="persistent",
        n_states=4,
        vocab_size=6,
        description="Diagonal-heavy transition matrix with strong state persistence.",
    ),
    "high_entropy": HMMFamilyConfig(
        name="high_entropy",
        n_states=4,
        vocab_size=6,
        description="Near-uniform transition matrix, weak transition signal.",
    ),
    "three_state": HMMFamilyConfig(
        name="three_state",
        n_states=3,
        vocab_size=5,
        description="Smaller latent system.",
    ),
    "six_state": HMMFamilyConfig(
        name="six_state",
        n_states=6,
        vocab_size=8,
        description="Larger latent system.",
    ),
}


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = np.clip(matrix, 1e-12, None)
    return matrix / matrix.sum(axis=1, keepdims=True)


def _sticky_transition(n_states: int, stickiness: float, noise: float = 0.0) -> np.ndarray:
    off_diag = (1.0 - stickiness) / (n_states - 1)
    transition = np.full((n_states, n_states), off_diag, dtype=np.float64)
    np.fill_diagonal(transition, stickiness)
    if noise > 0:
        rng = np.random.default_rng(12345 + n_states)
        transition += rng.uniform(0.0, noise, size=transition.shape)
    return _normalize_rows(transition)


def _banded_transition(n_states: int, self_prob: float = 0.55) -> np.ndarray:
    transition = np.zeros((n_states, n_states), dtype=np.float64)
    for i in range(n_states):
        transition[i, i] = self_prob
        transition[i, (i - 1) % n_states] += (1 - self_prob) / 2
        transition[i, (i + 1) % n_states] += (1 - self_prob) / 2
    return _normalize_rows(transition)


def _emissions(n_states: int, vocab_size: int, concentration: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = np.full((n_states, vocab_size), concentration, dtype=np.float64)
    for state in range(n_states):
        base[state, state % vocab_size] += 4.0
        base[state, (state + n_states) % vocab_size] += 2.0 if vocab_size > n_states else 0.0
    emission = np.vstack([rng.dirichlet(row) for row in base])
    return _normalize_rows(emission)


def make_hmm_family(name: str, seed: int = 0) -> HMM:
    if name not in FAMILIES:
        raise KeyError(f"Unknown HMM family: {name}")

    cfg = FAMILIES[name]
    rng = np.random.default_rng(seed)
    initial = rng.dirichlet(np.ones(cfg.n_states))

    if name == "easy_separable":
        transition = _banded_transition(cfg.n_states, self_prob=0.60)
        emission = _emissions(cfg.n_states, cfg.vocab_size, concentration=0.20, seed=seed + 11)
    elif name == "ambiguous_emissions":
        transition = _banded_transition(cfg.n_states, self_prob=0.60)
        emission = _emissions(cfg.n_states, cfg.vocab_size, concentration=2.50, seed=seed + 13)
    elif name == "persistent":
        transition = _sticky_transition(cfg.n_states, stickiness=0.82, noise=0.02)
        emission = _emissions(cfg.n_states, cfg.vocab_size, concentration=0.80, seed=seed + 17)
    elif name == "high_entropy":
        transition = _normalize_rows(np.ones((cfg.n_states, cfg.n_states)) + rng.uniform(0.0, 0.08, size=(cfg.n_states, cfg.n_states)))
        emission = _emissions(cfg.n_states, cfg.vocab_size, concentration=1.20, seed=seed + 19)
    elif name == "three_state":
        transition = _banded_transition(cfg.n_states, self_prob=0.64)
        emission = _emissions(cfg.n_states, cfg.vocab_size, concentration=0.80, seed=seed + 23)
    elif name == "six_state":
        transition = _banded_transition(cfg.n_states, self_prob=0.56)
        emission = _emissions(cfg.n_states, cfg.vocab_size, concentration=1.00, seed=seed + 29)
    else:
        raise AssertionError("unreachable")

    return HMM(transition=transition, emission=emission, initial=initial)


def transition_entropy(hmm: HMM) -> float:
    t = np.clip(hmm.transition, 1e-12, 1.0)
    return float(np.mean(-np.sum(t * np.log(t), axis=1)))


def emission_entropy(hmm: HMM) -> float:
    e = np.clip(hmm.emission, 1e-12, 1.0)
    return float(np.mean(-np.sum(e * np.log(e), axis=1)))

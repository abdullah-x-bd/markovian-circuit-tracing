from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class HMM:
    transition: np.ndarray
    emission: np.ndarray
    initial: np.ndarray
    name: str = "custom"

    @property
    def n_states(self) -> int:
        return int(self.transition.shape[0])

    @property
    def vocab_size(self) -> int:
        return int(self.emission.shape[1])


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return values / values.sum(axis=1, keepdims=True)


def make_hmm(observability: str = "medium") -> HMM:
    """Return a four-state HMM with fixed dynamics and controlled emission overlap.

    easy, medium, and hard share the same latent transition law. Only the
    emission matrix changes, so observability can be studied without changing
    the hidden dynamics.
    """
    transition = np.array(
        [
            [0.70, 0.20, 0.10, 0.00],
            [0.10, 0.60, 0.20, 0.10],
            [0.00, 0.20, 0.70, 0.10],
            [0.25, 0.25, 0.25, 0.25],
        ],
        dtype=np.float64,
    )
    emissions = {
        "easy": np.array(
            [
                [0.78, 0.10, 0.03, 0.03, 0.03, 0.03],
                [0.03, 0.03, 0.78, 0.06, 0.05, 0.05],
                [0.03, 0.03, 0.03, 0.72, 0.10, 0.09],
                [0.10, 0.08, 0.08, 0.08, 0.34, 0.32],
            ],
            dtype=np.float64,
        ),
        "medium": np.array(
            [
                [0.45, 0.35, 0.05, 0.05, 0.05, 0.05],
                [0.05, 0.05, 0.65, 0.10, 0.10, 0.05],
                [0.05, 0.05, 0.05, 0.35, 0.25, 0.25],
                [0.20, 0.15, 0.15, 0.15, 0.20, 0.15],
            ],
            dtype=np.float64,
        ),
        "hard": np.array(
            [
                [0.24, 0.20, 0.15, 0.14, 0.14, 0.13],
                [0.14, 0.14, 0.25, 0.18, 0.16, 0.13],
                [0.13, 0.14, 0.14, 0.22, 0.19, 0.18],
                [0.18, 0.16, 0.16, 0.16, 0.18, 0.16],
            ],
            dtype=np.float64,
        ),
    }
    if observability not in emissions:
        raise ValueError(f"Unknown observability={observability!r}; choose easy, medium, or hard")
    emission = _normalize_rows(emissions[observability])
    initial = np.full(4, 0.25, dtype=np.float64)
    return HMM(transition=transition, emission=emission, initial=initial, name=observability)


def default_hmm() -> HMM:
    return make_hmm("medium")


def _sample_categorical(rng: np.random.Generator, probs: np.ndarray) -> int:
    return int(rng.choice(len(probs), p=probs))


def sample_hmm_sequences(
    hmm: HMM,
    n_sequences: int,
    seq_len: int,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample x_t ~ E[s_t], then s_{t+1} ~ T[s_t]."""
    rng = np.random.default_rng(seed)
    states = np.zeros((n_sequences, seq_len), dtype=np.int64)
    tokens = np.zeros((n_sequences, seq_len), dtype=np.int64)
    for n in range(n_sequences):
        state = _sample_categorical(rng, hmm.initial)
        for t in range(seq_len):
            states[n, t] = state
            tokens[n, t] = _sample_categorical(rng, hmm.emission[state])
            state = _sample_categorical(rng, hmm.transition[state])
    return tokens, states


def bayes_predictive_state_beliefs(hmm: HMM, tokens: np.ndarray) -> np.ndarray:
    """Return b^-_t = P(s_t | x_<t) at every position.

    These priors are temporally aligned with a causal LM activation at position t
    when the model input is x_<t and the target is x_t.
    """
    batch, seq_len = tokens.shape
    priors = np.zeros((batch, seq_len, hmm.n_states), dtype=np.float64)
    for n in range(batch):
        prior = hmm.initial.copy()
        for t in range(seq_len):
            priors[n, t] = prior
            likelihood = hmm.emission[:, tokens[n, t]]
            posterior = prior * likelihood
            posterior = posterior / posterior.sum()
            prior = posterior @ hmm.transition
    return priors


def bayes_filter(hmm: HMM, tokens: np.ndarray) -> np.ndarray:
    """Return b^+_t = P(s_t | x_<=t) after consuming x_t."""
    priors = bayes_predictive_state_beliefs(hmm, tokens)
    posteriors = np.zeros_like(priors)
    for n in range(tokens.shape[0]):
        for t in range(tokens.shape[1]):
            posterior = priors[n, t] * hmm.emission[:, tokens[n, t]]
            posteriors[n, t] = posterior / posterior.sum()
    return posteriors


def bayes_predictive_distribution(hmm: HMM, tokens: np.ndarray) -> np.ndarray:
    """Return P(x_t | x_<t), aligned with the causal LM target at position t."""
    return bayes_predictive_state_beliefs(hmm, tokens) @ hmm.emission


def bayes_next_token_distribution(hmm: HMM, posterior_beliefs: np.ndarray) -> np.ndarray:
    """Return P(x_{t+1}|x_<=t) from posterior beliefs over s_t."""
    return (posterior_beliefs @ hmm.transition) @ hmm.emission


def current_state_emission_distribution(hmm: HMM, state_id: int) -> np.ndarray:
    """Exact P(x_t | s_t=state_id)."""
    return hmm.emission[state_id].copy()


def next_state_predictive_distribution(hmm: HMM, state_id: int) -> np.ndarray:
    """Exact P(x_{t+1} | s_t=state_id), after one latent transition."""
    return hmm.transition[state_id] @ hmm.emission


def sequence_cross_entropy(tokens: np.ndarray, probs: np.ndarray, eps: float = 1e-12) -> float:
    chosen = np.take_along_axis(probs, tokens[..., None], axis=-1).squeeze(-1)
    return float(-np.log(np.clip(chosen, eps, 1.0)).mean())


def unigram_distribution(tokens: np.ndarray, vocab_size: int, smoothing: float = 1e-6) -> np.ndarray:
    counts = np.bincount(tokens.reshape(-1), minlength=vocab_size).astype(np.float64)
    counts += smoothing
    return counts / counts.sum()


def forced_state_next_token_distribution(hmm: HMM, state_id: int) -> np.ndarray:
    """Backward-compatible alias for the one-transition-ahead distribution.

    New same-position interventions must use current_state_emission_distribution.
    """
    return next_state_predictive_distribution(hmm, state_id)


def make_lm_tensors(tokens: np.ndarray, bos_token: int) -> tuple[torch.Tensor, torch.Tensor]:
    """At position t, input contains x_<t and target is x_t."""
    batch, seq_len = tokens.shape
    x = np.full((batch, seq_len), bos_token, dtype=np.int64)
    x[:, 1:] = tokens[:, :-1]
    y = tokens.copy()
    return torch.from_numpy(x), torch.from_numpy(y)

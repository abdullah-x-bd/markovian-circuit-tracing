from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class HMM:
    transition: np.ndarray
    emission: np.ndarray
    initial: np.ndarray

    @property
    def n_states(self) -> int:
        return int(self.transition.shape[0])

    @property
    def vocab_size(self) -> int:
        return int(self.emission.shape[1])


def default_hmm() -> HMM:
    transition = np.array(
        [
            [0.70, 0.20, 0.10, 0.00],
            [0.10, 0.60, 0.20, 0.10],
            [0.00, 0.20, 0.70, 0.10],
            [0.25, 0.25, 0.25, 0.25],
        ],
        dtype=np.float64,
    )
    emission = np.array(
        [
            [0.45, 0.35, 0.05, 0.05, 0.05, 0.05],
            [0.05, 0.05, 0.65, 0.10, 0.10, 0.05],
            [0.05, 0.05, 0.05, 0.35, 0.25, 0.25],
            [0.20, 0.15, 0.15, 0.15, 0.20, 0.15],
        ],
        dtype=np.float64,
    )
    initial = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float64)
    return HMM(transition=transition, emission=emission, initial=initial)


def _sample_categorical(rng: np.random.Generator, probs: np.ndarray) -> int:
    return int(rng.choice(len(probs), p=probs))


def sample_hmm_sequences(
    hmm: HMM,
    n_sequences: int,
    seq_len: int,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
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


def bayes_filter(hmm: HMM, tokens: np.ndarray) -> np.ndarray:
    batch, seq_len = tokens.shape
    beliefs = np.zeros((batch, seq_len, hmm.n_states), dtype=np.float64)

    for n in range(batch):
        belief = hmm.initial.copy()
        for t in range(seq_len):
            likelihood = hmm.emission[:, tokens[n, t]]
            posterior = belief * likelihood
            posterior = posterior / posterior.sum()
            beliefs[n, t] = posterior
            belief = posterior @ hmm.transition

    return beliefs


def bayes_next_token_distribution(hmm: HMM, beliefs: np.ndarray) -> np.ndarray:
    next_state_prior = beliefs @ hmm.transition
    return next_state_prior @ hmm.emission


def forced_state_next_token_distribution(hmm: HMM, state_id: int) -> np.ndarray:
    return hmm.transition[state_id] @ hmm.emission


def make_lm_tensors(tokens: np.ndarray, bos_token: int) -> tuple[torch.Tensor, torch.Tensor]:
    batch, seq_len = tokens.shape
    x = np.full((batch, seq_len), bos_token, dtype=np.int64)
    x[:, 1:] = tokens[:, :-1]
    y = tokens.copy()
    return torch.from_numpy(x), torch.from_numpy(y)

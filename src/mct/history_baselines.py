from __future__ import annotations

import numpy as np

from mct.probes import belief_probe_metrics, state_probe_metrics


def token_history_features(
    tokens: np.ndarray,
    vocab_size: int,
    history: int,
    bos_token: int | None = None,
) -> np.ndarray:
    """One-hot features for observations available before x_t only."""
    if history < 1:
        raise ValueError("history must be >= 1")
    if bos_token is None:
        bos_token = vocab_size
    alphabet = vocab_size + 1
    batch, seq_len = tokens.shape
    features = np.zeros((batch, seq_len, history * alphabet), dtype=np.float32)
    for lag in range(1, history + 1):
        block = (lag - 1) * alphabet
        for t in range(seq_len):
            source_t = t - lag
            ids = tokens[:, source_t] if source_t >= 0 else np.full(batch, bos_token, dtype=np.int64)
            features[np.arange(batch), t, block + ids] = 1.0
    return features


def history_baseline_report(
    calibration_tokens: np.ndarray,
    evaluation_tokens: np.ndarray,
    calibration_beliefs: np.ndarray,
    evaluation_beliefs: np.ndarray,
    calibration_states: np.ndarray,
    evaluation_states: np.ndarray,
    vocab_size: int,
    histories: tuple[int, ...] = (1, 2, 4),
) -> dict[str, float]:
    """Score causal token-history features with the same held-out probe protocol."""
    report: dict[str, float] = {}
    for history in histories:
        cal = token_history_features(calibration_tokens, vocab_size=vocab_size, history=history)
        ev = token_history_features(evaluation_tokens, vocab_size=vocab_size, history=history)
        belief = belief_probe_metrics(cal, calibration_beliefs, ev, evaluation_beliefs, evaluation_states)
        state = state_probe_metrics(cal, calibration_states, ev, evaluation_states)
        report.update({f"history{history}_{k}": v for k, v in {**belief, **state}.items()})
    return report

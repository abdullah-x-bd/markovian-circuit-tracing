from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SequenceSplit:
    calibration: np.ndarray
    selection: np.ndarray
    evaluation: np.ndarray


def split_sequence_indices(
    n_sequences: int,
    seed: int = 0,
    fractions: tuple[float, float, float] = (0.60, 0.20, 0.20),
) -> SequenceSplit:
    """Split by whole sequence to prevent token-level train/test leakage."""
    if n_sequences < 5:
        raise ValueError("Need at least five sequences for calibration/selection/evaluation")
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError("fractions must sum to 1")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_sequences)
    n_cal = max(1, int(np.floor(fractions[0] * n_sequences)))
    n_sel = max(1, int(np.floor(fractions[1] * n_sequences)))
    n_eval = n_sequences - n_cal - n_sel
    if n_eval < 1:
        raise ValueError("Split leaves no evaluation sequences")
    return SequenceSplit(
        calibration=np.sort(perm[:n_cal]),
        selection=np.sort(perm[n_cal:n_cal + n_sel]),
        evaluation=np.sort(perm[n_cal + n_sel:]),
    )

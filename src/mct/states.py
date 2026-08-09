from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans


@dataclass
class StateAbstraction:
    kmeans: KMeans
    mapping: dict[int, int]
    n_states: int

    def predict(self, activations: np.ndarray) -> np.ndarray:
        flat = activations.reshape(-1, activations.shape[-1])
        raw = self.kmeans.predict(flat)
        mapped = np.array([self.mapping[int(z)] for z in raw], dtype=np.int64)
        return mapped.reshape(activations.shape[:-1])


def _confusion_mapping(pred: np.ndarray, true: np.ndarray, n_states: int) -> dict[int, int]:
    confusion = np.zeros((n_states, n_states), dtype=np.int64)
    for p, t in zip(pred.reshape(-1), true.reshape(-1), strict=False):
        confusion[int(p), int(t)] += 1
    row_ind, col_ind = linear_sum_assignment(confusion.max() - confusion)
    return {int(r): int(c) for r, c in zip(row_ind, col_ind, strict=True)}


def fit_state_abstraction(
    calibration_activations: np.ndarray,
    calibration_true_states: np.ndarray,
    n_states: int,
    seed: int = 0,
) -> StateAbstraction:
    """Fit KMeans and cluster-to-state alignment on calibration sequences only."""
    flat = calibration_activations.reshape(-1, calibration_activations.shape[-1])
    km = KMeans(n_clusters=n_states, random_state=seed, n_init=10)
    raw = km.fit_predict(flat).reshape(calibration_true_states.shape)
    mapping = _confusion_mapping(raw, calibration_true_states, n_states)
    return StateAbstraction(kmeans=km, mapping=mapping, n_states=n_states)


def state_recovery_accuracy(
    abstraction: StateAbstraction,
    activations: np.ndarray,
    true_states: np.ndarray,
) -> float:
    pred = abstraction.predict(activations)
    return float((pred == true_states).mean())


def state_centroids(activations: np.ndarray, states: np.ndarray, n_states: int) -> np.ndarray:
    flat_x = activations.reshape(-1, activations.shape[-1])
    flat_s = states.reshape(-1)
    centroids = np.zeros((n_states, flat_x.shape[-1]), dtype=np.float32)
    for state_id in range(n_states):
        mask = flat_s == state_id
        if not mask.any():
            raise ValueError(f"No samples for state {state_id}")
        centroids[state_id] = flat_x[mask].mean(axis=0)
    return centroids


# Backward-compatible helpers retained for notebooks that use the v0.1 API.
def cluster_internal_states(activations: np.ndarray, n_states: int, seed: int = 0) -> np.ndarray:
    flat = activations.reshape(-1, activations.shape[-1])
    km = KMeans(n_clusters=n_states, random_state=seed, n_init=10)
    return km.fit_predict(flat).reshape(activations.shape[:-1])


def best_label_match(pred_states: np.ndarray, true_states: np.ndarray, n_states: int) -> tuple[np.ndarray, float]:
    mapping = _confusion_mapping(pred_states, true_states, n_states)
    remapped = np.vectorize(lambda z: mapping.get(int(z), int(z)))(pred_states)
    return remapped, float((remapped == true_states).mean())

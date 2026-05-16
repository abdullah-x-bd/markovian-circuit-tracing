from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def flatten_time(activations: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = activations.reshape(-1, activations.shape[-1])
    y = labels.reshape(-1)
    return x, y


def probe_state_recovery(
    activations: np.ndarray,
    true_states: np.ndarray,
    test_size: float = 0.25,
    seed: int = 0,
) -> float:
    x, y = flatten_time(activations, true_states)
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )
    clf = LogisticRegression(max_iter=1000, multi_class="auto")
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    return float(accuracy_score(y_test, pred))


def cluster_internal_states(
    activations: np.ndarray,
    n_states: int,
    seed: int = 0,
) -> np.ndarray:
    flat = activations.reshape(-1, activations.shape[-1])
    km = KMeans(n_clusters=n_states, random_state=seed, n_init="auto")
    labels = km.fit_predict(flat)
    return labels.reshape(activations.shape[0], activations.shape[1])


def best_label_match(pred_states: np.ndarray, true_states: np.ndarray, n_states: int) -> tuple[np.ndarray, float]:
    confusion = np.zeros((n_states, n_states), dtype=np.int64)
    for p, t in zip(pred_states.reshape(-1), true_states.reshape(-1), strict=False):
        confusion[p, t] += 1
    row_ind, col_ind = linear_sum_assignment(confusion.max() - confusion)
    mapping = {int(r): int(c) for r, c in zip(row_ind, col_ind, strict=False)}
    remapped = np.vectorize(lambda z: mapping.get(int(z), int(z)))(pred_states)
    acc = float((remapped == true_states).mean())
    return remapped, acc


def state_centroids(activations: np.ndarray, states: np.ndarray, n_states: int) -> np.ndarray:
    flat_x, flat_s = flatten_time(activations, states)
    centroids = np.zeros((n_states, flat_x.shape[-1]), dtype=np.float32)
    for state_id in range(n_states):
        mask = flat_s == state_id
        if not mask.any():
            raise ValueError(f"No samples for state {state_id}")
        centroids[state_id] = flat_x[mask].mean(axis=0)
    return centroids

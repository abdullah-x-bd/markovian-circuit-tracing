from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection

from mct.states import best_label_match
from mct.transition import estimate_transition_matrix, transition_report


def flatten_representation(x: np.ndarray) -> np.ndarray:
    return x.reshape(-1, x.shape[-1])


def reshape_labels(labels: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    return labels.reshape(shape)


def kmeans_states(representation: np.ndarray, k: int, seed: int) -> np.ndarray:
    flat = flatten_representation(representation)
    km = KMeans(n_clusters=k, random_state=seed, n_init=10)
    labels = km.fit_predict(flat)
    return reshape_labels(labels, representation.shape[:2])


def pca_kmeans_states(representation: np.ndarray, k: int, pca_dim: int, seed: int) -> np.ndarray:
    flat = flatten_representation(representation)
    pca_dim = min(pca_dim, flat.shape[-1])
    reduced = PCA(n_components=pca_dim, random_state=seed).fit_transform(flat)
    km = KMeans(n_clusters=k, random_state=seed, n_init=10)
    labels = km.fit_predict(reduced)
    return reshape_labels(labels, representation.shape[:2])


def random_projection_kmeans_states(representation: np.ndarray, k: int, proj_dim: int, seed: int) -> np.ndarray:
    flat = flatten_representation(representation)
    proj_dim = min(proj_dim, flat.shape[-1])
    reduced = GaussianRandomProjection(n_components=proj_dim, random_state=seed).fit_transform(flat)
    km = KMeans(n_clusters=k, random_state=seed, n_init=10)
    labels = km.fit_predict(reduced)
    return reshape_labels(labels, representation.shape[:2])


def belief_cluster_states(beliefs: np.ndarray, k: int, seed: int) -> np.ndarray:
    flat = flatten_representation(beliefs)
    km = KMeans(n_clusters=k, random_state=seed, n_init=10)
    labels = km.fit_predict(flat)
    return reshape_labels(labels, beliefs.shape[:2])


def token_state(tokens: np.ndarray) -> np.ndarray:
    return tokens.copy()


def token_bigram_state(tokens: np.ndarray, vocab_size: int) -> np.ndarray:
    prev = np.zeros_like(tokens)
    prev[:, 1:] = tokens[:, :-1]
    return prev * vocab_size + tokens


def transition_if_same_k(
    true_transition: np.ndarray,
    states: np.ndarray,
    true_states: np.ndarray,
    k: int,
) -> dict[str, float]:
    if k != true_transition.shape[0]:
        return {}
    remapped, acc = best_label_match(states, true_states, n_states=k)
    estimated_t = estimate_transition_matrix(remapped, n_states=k)
    return {
        "cluster_accuracy": acc,
        **transition_report(true_transition, estimated_t),
    }


def cluster_belief_reconstruction(
    states: np.ndarray,
    beliefs: np.ndarray,
    eps: float = 1e-12,
) -> dict[str, float]:
    flat_states = states.reshape(-1)
    flat_beliefs = beliefs.reshape(-1, beliefs.shape[-1])
    k = int(flat_states.max()) + 1
    centroids = np.zeros((k, beliefs.shape[-1]), dtype=np.float64)
    counts = np.zeros(k, dtype=np.float64)

    for state, belief in zip(flat_states, flat_beliefs, strict=False):
        centroids[int(state)] += belief
        counts[int(state)] += 1

    for idx in range(k):
        if counts[idx] > 0:
            centroids[idx] /= counts[idx]
        else:
            centroids[idx] = np.ones(beliefs.shape[-1]) / beliefs.shape[-1]

    recon = centroids[flat_states]
    mse = float(np.mean((flat_beliefs - recon) ** 2))
    p = np.clip(flat_beliefs, eps, 1.0)
    q = np.clip(recon, eps, 1.0)
    kl = float(np.mean(np.sum(p * (np.log(p) - np.log(q)), axis=-1)))
    return {
        "belief_reconstruction_mse": mse,
        "belief_reconstruction_kl": kl,
    }


def next_true_state_nll(states: np.ndarray, true_states: np.ndarray, smoothing: float = 1e-6) -> float:
    flat_z = states[:, :-1].reshape(-1)
    flat_next_true = true_states[:, 1:].reshape(-1)
    k = int(flat_z.max()) + 1
    n_true = int(true_states.max()) + 1
    counts = np.full((k, n_true), smoothing, dtype=np.float64)
    for z, s_next in zip(flat_z, flat_next_true, strict=False):
        counts[int(z), int(s_next)] += 1.0
    probs = counts / counts.sum(axis=1, keepdims=True)
    selected = probs[flat_z, flat_next_true]
    return float(-np.log(np.clip(selected, 1e-12, 1.0)).mean())


def evaluate_representation_for_k(
    representation: np.ndarray,
    true_states: np.ndarray,
    beliefs: np.ndarray,
    true_transition: np.ndarray,
    k: int,
    seed: int,
    method: str,
    pca_dim: int = 8,
) -> dict[str, float]:
    if method == "residual_kmeans":
        states = kmeans_states(representation, k=k, seed=seed)
    elif method == "pca_kmeans":
        states = pca_kmeans_states(representation, k=k, pca_dim=pca_dim, seed=seed)
    elif method == "random_projection_kmeans":
        states = random_projection_kmeans_states(representation, k=k, proj_dim=pca_dim, seed=seed)
    elif method == "belief_kmeans":
        states = belief_cluster_states(beliefs, k=k, seed=seed)
    else:
        raise KeyError(method)

    metrics = {
        "method": method,
        "k": k,
        "next_true_state_nll": next_true_state_nll(states, true_states),
        **cluster_belief_reconstruction(states, beliefs),
    }
    same_k = transition_if_same_k(true_transition, states, true_states, k=k)
    metrics.update(same_k)
    return metrics

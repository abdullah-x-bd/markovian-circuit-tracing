"""Post-hoc robustness analyses for the Neural Processing Letters revision.

This script leaves results/v1 unchanged. It reproduces the canonical trained
models from configs/main_experiment.yaml, adds transition-specific untrained and
explicit-history controls, evaluates held-out probabilistic Markov NLL, and
repeats the transition/NLL analysis under a second cyclic latent transition law.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import torch
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment

from mct.data import HMM, bayes_predictive_distribution, make_hmm, make_lm_tensors, sample_hmm_sequences, sequence_cross_entropy
from mct.model import TinyTransformer, TransformerConfig
from mct.splits import split_sequence_indices
from mct.states import fit_state_abstraction
from mct.train import collect_activations, train_model
from mct.transition import estimate_transition_matrix, rowwise_kl

SEEDS = [7, 17, 29, 43, 71]
OBS = ["easy", "medium", "hard"]
CYCLIC_T = np.array([
    [0.58, 0.32, 0.07, 0.03],
    [0.05, 0.58, 0.32, 0.05],
    [0.03, 0.07, 0.58, 0.32],
    [0.32, 0.05, 0.05, 0.58],
], dtype=np.float64)

def cyclic_hmm(observability: str) -> HMM:
    base = make_hmm(observability)
    return HMM(transition=CYCLIC_T.copy(), emission=base.emission.copy(), initial=base.initial.copy(), name=f"cyclic_{observability}")

def history_features(tokens: np.ndarray, history: int = 4) -> np.ndarray:
    n, length = tokens.shape
    vocab_plus_bos = 7
    out = np.zeros((n, length, history * vocab_plus_bos), dtype=np.float32)
    for lag in range(1, history + 1):
        values = np.full((n, length), 6, dtype=np.int64)
        values[:, lag:] = tokens[:, :-lag]
        for value in range(vocab_plus_bos):
            out[:, :, (lag - 1) * vocab_plus_bos + value] = values == value
    return out

def aligned_kmeans(cal_x, cal_states, ev_x, seed):
    flat = cal_x.reshape(-1, cal_x.shape[-1])
    km = KMeans(n_clusters=4, random_state=seed, n_init=10).fit(flat)
    cal_raw = km.predict(flat).reshape(cal_states.shape)
    confusion = np.zeros((4, 4), dtype=np.int64)
    for pred, true in zip(cal_raw.ravel(), cal_states.ravel(), strict=False):
        confusion[int(pred), int(true)] += 1
    rows, cols = linear_sum_assignment(confusion.max() - confusion)
    mapping = {int(r): int(c) for r, c in zip(rows, cols, strict=True)}
    raw = km.predict(ev_x.reshape(-1, ev_x.shape[-1]))
    return np.array([mapping[int(v)] for v in raw], dtype=np.int64).reshape(ev_x.shape[:-1])

def markov_nll(cal_states: np.ndarray, ev_states: np.ndarray, smoothing: float = 1e-6):
    counts0 = np.full(4, smoothing)
    for value in cal_states.ravel():
        counts0[int(value)] += 1
    p0 = counts0 / counts0.sum()
    p1 = estimate_transition_matrix(cal_states, 4, smoothing=smoothing)
    counts2 = np.full((4, 4, 4), smoothing)
    for seq in cal_states:
        for a, b, c in zip(seq[:-2], seq[1:-1], seq[2:], strict=False):
            counts2[int(a), int(b), int(c)] += 1
    p2 = counts2 / counts2.sum(axis=-1, keepdims=True)
    loss0 = loss1 = loss2 = 0.0
    n01 = n2 = 0
    for seq in ev_states:
        for t in range(1, len(seq)):
            loss0 -= np.log(p0[seq[t]])
            loss1 -= np.log(p1[seq[t - 1], seq[t]])
            n01 += 1
        for t in range(2, len(seq)):
            loss2 -= np.log(p2[seq[t - 2], seq[t - 1], seq[t]])
            n2 += 1
    return {"nll0": float(loss0 / n01), "nll1": float(loss1 / n01), "nll2": float(loss2 / n2)}

def run_cell(observability: str, seed: int, family: str):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(2)
    hmm = make_hmm(observability) if family == "canonical" else cyclic_hmm(observability)
    train_tokens, _ = sample_hmm_sequences(hmm, 1200, 40, seed=seed)
    val_tokens, _ = sample_hmm_sequences(hmm, 250, 40, seed=seed + 1)
    analysis_tokens, analysis_states = sample_hmm_sequences(hmm, 800, 40, seed=seed + 2)
    train_x, train_y = make_lm_tensors(train_tokens, 6)
    val_x, val_y = make_lm_tensors(val_tokens, 6)
    analysis_x, _ = make_lm_tensors(analysis_tokens, 6)
    cfg = TransformerConfig(vocab_size=7, seq_len=40, d_model=64, n_layers=2, n_heads=4, d_mlp=128)
    model = TinyTransformer(cfg)
    bayes_val = sequence_cross_entropy(val_tokens, bayes_predictive_distribution(hmm, val_tokens))
    result = train_model(model, train_x, train_y, val_x, val_y, epochs=30 if family == "canonical" else 60,
                         batch_size=256, lr=3e-4, min_epochs=6, target_val_loss=bayes_val + 0.02)
    split = split_sequence_indices(800, seed=seed + 3)
    acts = collect_activations(model, analysis_x, "resid_post_1", 256).numpy()
    abstraction = fit_state_abstraction(acts[split.calibration], analysis_states[split.calibration], 4, seed=seed)
    cal_rec = abstraction.predict(acts[split.calibration])
    ev_rec = abstraction.predict(acts[split.evaluation])
    recovered_kl = rowwise_kl(hmm.transition, estimate_transition_matrix(ev_rec, 4))
    rng = np.random.default_rng(seed + 101)
    shuffled = ev_rec.copy().reshape(-1); rng.shuffle(shuffled); shuffled = shuffled.reshape(ev_rec.shape)
    rng = np.random.default_rng(seed + 202)
    random_states = rng.integers(0, 4, size=ev_rec.shape)
    row = {
        "family": family, "observability": observability, "seed": seed,
        "epochs": len(result.train_loss), "validation_gap": result.val_loss[-1] - bayes_val,
        "transition_kl": recovered_kl,
        "shuffled_transition_kl": rowwise_kl(hmm.transition, estimate_transition_matrix(shuffled, 4)),
        "random_transition_kl": rowwise_kl(hmm.transition, estimate_transition_matrix(random_states, 4)),
    }
    row.update(markov_nll(cal_rec, ev_rec))
    row["nll_gain_0_to_1"] = row["nll0"] - row["nll1"]
    row["nll_gain_1_to_2"] = row["nll1"] - row["nll2"]
    if family == "canonical":
        torch.manual_seed(seed + 999)
        untrained = TinyTransformer(cfg)
        uacts = collect_activations(untrained, analysis_x, "resid_post_1", 256).numpy()
        urec = aligned_kmeans(uacts[split.calibration], analysis_states[split.calibration], uacts[split.evaluation], seed)
        row["untrained_transition_kl"] = rowwise_kl(hmm.transition, estimate_transition_matrix(urec, 4))
        h = history_features(analysis_tokens, 4)
        hrec = aligned_kmeans(h[split.calibration], analysis_states[split.calibration], h[split.evaluation], seed)
        row["history4_kmeans_transition_kl"] = rowwise_kl(hmm.transition, estimate_transition_matrix(hrec, 4))
    return row

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--family", choices=["canonical", "cyclic"], required=True)
    p.add_argument("--observability", choices=OBS, required=True)
    p.add_argument("--seed", type=int, choices=SEEDS, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    row = run_cell(args.observability, args.seed, args.family)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(row, indent=2))
    print(json.dumps(row, indent=2))

if __name__ == "__main__":
    main()

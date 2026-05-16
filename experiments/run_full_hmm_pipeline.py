from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from mct.data import (
    default_hmm,
    forced_state_next_token_distribution,
    make_lm_tensors,
    sample_hmm_sequences,
)
from mct.interventions import state_forcing_kl
from mct.model import TinyTransformer, TransformerConfig, count_parameters
from mct.plots import save_loss_curve, save_matrix_heatmap
from mct.states import best_label_match, cluster_internal_states, probe_state_recovery, state_centroids
from mct.train import collect_activations, evaluate_loss, train_model
from mct.transition import estimate_transition_matrix, markov_order_accuracy, transition_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Markovian Circuit Tracing HMM pipeline")
    parser.add_argument("--output-dir", type=str, default="runs/hmm_4state_demo")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--train-sequences", type=int, default=8000)
    parser.add_argument("--val-sequences", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--d-mlp", type=int, default=256)
    parser.add_argument("--activation-name", type=str, default="resid_post_1")
    parser.add_argument("--forcing-position", type=int, default=20)
    parser.add_argument("--forcing-samples", type=int, default=512)
    parser.add_argument("--num-threads", type=int, default=2)
    return parser.parse_args()


def pad_visible_distribution_for_bos(visible_probs: np.ndarray, bos_token: int) -> np.ndarray:
    padded = np.zeros((visible_probs.shape[0], bos_token + 1), dtype=visible_probs.dtype)
    padded[:, :bos_token] = visible_probs
    return padded


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(args.num_threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    hmm = default_hmm()
    bos_token = hmm.vocab_size

    train_tokens, _ = sample_hmm_sequences(
        hmm,
        args.train_sequences,
        args.seq_len,
        seed=args.seed,
    )
    val_tokens, val_states = sample_hmm_sequences(
        hmm,
        args.val_sequences,
        args.seq_len,
        seed=args.seed + 1,
    )

    train_x, train_y = make_lm_tensors(train_tokens, bos_token=bos_token)
    val_x, val_y = make_lm_tensors(val_tokens, bos_token=bos_token)

    cfg = TransformerConfig(
        vocab_size=hmm.vocab_size + 1,
        seq_len=args.seq_len,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_mlp=args.d_mlp,
    )
    model = TinyTransformer(cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    result = train_model(
        model,
        train_x,
        train_y,
        val_x,
        val_y,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )

    acts = collect_activations(
        model,
        val_x,
        activation_name=args.activation_name,
        batch_size=args.batch_size,
    ).numpy()
    probe_acc = probe_state_recovery(acts, val_states, seed=args.seed)

    discovered = cluster_internal_states(acts, n_states=hmm.n_states, seed=args.seed)
    remapped, cluster_acc = best_label_match(discovered, val_states, n_states=hmm.n_states)
    estimated_t = estimate_transition_matrix(remapped, n_states=hmm.n_states)
    transition_metrics = transition_report(hmm.transition, estimated_t)
    markov_metrics = markov_order_accuracy(remapped, n_states=hmm.n_states)

    centroids = state_centroids(acts, remapped, n_states=hmm.n_states)
    ideal_visible_forced = np.stack(
        [forced_state_next_token_distribution(hmm, s) for s in range(hmm.n_states)],
        axis=0,
    )
    ideal_forced = pad_visible_distribution_for_bos(ideal_visible_forced, bos_token=bos_token)
    n_force = min(args.forcing_samples, val_x.shape[0])
    forced_kl = state_forcing_kl(
        model,
        val_x[:n_force],
        centroids,
        ideal_forced,
        activation_name=args.activation_name,
        position=args.forcing_position,
        batch_size=min(args.batch_size, 128),
    )

    metrics = {
        "device": device,
        "seed": args.seed,
        "seq_len": args.seq_len,
        "train_sequences": args.train_sequences,
        "val_sequences": args.val_sequences,
        "epochs": args.epochs,
        "parameter_count": count_parameters(model),
        "final_train_loss": result.train_loss[-1],
        "final_val_loss": evaluate_loss(model, val_x, val_y, batch_size=args.batch_size),
        "probe_state_recovery_accuracy": probe_acc,
        "cluster_state_recovery_accuracy": cluster_acc,
        **transition_metrics,
        **markov_metrics,
        **forced_kl,
    }

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    np.save(out_dir / "true_transition.npy", hmm.transition)
    np.save(out_dir / "estimated_transition.npy", estimated_t)
    save_matrix_heatmap(hmm.transition, "True HMM transition", out_dir / "true_transition.png")
    save_matrix_heatmap(estimated_t, "Recovered internal transition", out_dir / "estimated_transition.png")
    save_matrix_heatmap(hmm.transition - estimated_t, "Transition difference", out_dir / "transition_difference.png")
    save_loss_curve(result.train_loss, result.val_loss, out_dir / "loss_curve.png")

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

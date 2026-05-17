from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from mct.data import (
    bayes_filter,
    bayes_predictive_distribution,
    default_hmm,
    forced_state_next_token_distribution,
    make_lm_tensors,
    sample_hmm_sequences,
    sequence_cross_entropy,
    unigram_distribution,
)
from mct.interventions import state_forcing_kl
from mct.model import TinyTransformer, TransformerConfig, count_parameters
from mct.plots import save_loss_curve, save_matrix_heatmap
from mct.probes import belief_probe_metrics
from mct.sae import encode_activations, train_sae
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
    parser.add_argument("--run-sae", action="store_true")
    parser.add_argument("--run-sae-sweep", action="store_true")
    parser.add_argument("--sae-hidden-dim", type=int, default=256)
    parser.add_argument("--sae-epochs", type=int, default=5)
    parser.add_argument("--sae-l1-coef", type=float, default=1e-3)
    parser.add_argument("--sae-top-k", type=int, default=0)
    parser.add_argument("--sae-max-samples", type=int, default=50000)
    parser.add_argument("--sae-sweep-hidden-dims", type=str, default="128,256,512")
    parser.add_argument("--sae-sweep-l1-coefs", type=str, default="0.003,0.01,0.03")
    parser.add_argument("--sae-sweep-top-ks", type=str, default="0,8,16,32")
    return parser.parse_args()


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def pad_visible_distribution_for_bos(visible_probs: np.ndarray, bos_token: int) -> np.ndarray:
    padded = np.zeros((visible_probs.shape[0], bos_token + 1), dtype=visible_probs.dtype)
    padded[:, :bos_token] = visible_probs
    return padded


def repeated_distribution_loss(tokens: np.ndarray, distribution: np.ndarray) -> float:
    probs = np.broadcast_to(distribution, (*tokens.shape, distribution.shape[0]))
    return sequence_cross_entropy(tokens, probs)


def shuffled_state_baseline(true_transition: np.ndarray, states: np.ndarray, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    shuffled = states.copy().reshape(-1)
    rng.shuffle(shuffled)
    shuffled = shuffled.reshape(states.shape)
    shuffled_t = estimate_transition_matrix(shuffled, n_states=true_transition.shape[0])
    return {f"shuffled_{k}": v for k, v in transition_report(true_transition, shuffled_t).items()}


def random_state_baseline(true_transition: np.ndarray, states: np.ndarray, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    random_states = rng.integers(0, true_transition.shape[0], size=states.shape)
    random_t = estimate_transition_matrix(random_states, n_states=true_transition.shape[0])
    return {f"random_{k}": v for k, v in transition_report(true_transition, random_t).items()}


def true_state_sanity_check(true_transition: np.ndarray, states: np.ndarray) -> dict[str, float]:
    true_empirical_t = estimate_transition_matrix(states, n_states=true_transition.shape[0])
    return {f"true_state_empirical_{k}": v for k, v in transition_report(true_transition, true_empirical_t).items()}


def prefixed(prefix: str, values: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in values.items()}


def state_pipeline_metrics(
    representation: np.ndarray,
    true_states: np.ndarray,
    true_transition: np.ndarray,
    seed: int,
    prefix: str,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    probe_acc = probe_state_recovery(representation, true_states, seed=seed)
    discovered = cluster_internal_states(representation, n_states=true_transition.shape[0], seed=seed)
    remapped, cluster_acc = best_label_match(discovered, true_states, n_states=true_transition.shape[0])
    estimated_t = estimate_transition_matrix(remapped, n_states=true_transition.shape[0])

    metrics = {
        f"{prefix}_probe_state_recovery_accuracy": probe_acc,
        f"{prefix}_cluster_state_recovery_accuracy": cluster_acc,
        **prefixed(prefix, transition_report(true_transition, estimated_t)),
        **prefixed(prefix, markov_order_accuracy(remapped, n_states=true_transition.shape[0])),
    }
    return metrics, remapped, estimated_t


def run_sae_state_pipeline(
    activations: np.ndarray,
    true_states: np.ndarray,
    true_transition: np.ndarray,
    seed: int,
    hidden_dim: int,
    l1_coef: float,
    epochs: int,
    batch_size: int,
    max_samples: int,
    prefix: str,
    top_k: int | None = None,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    sae, sae_result = train_sae(
        activations,
        hidden_dim=hidden_dim,
        l1_coef=l1_coef,
        epochs=epochs,
        batch_size=batch_size,
        max_samples=max_samples,
        seed=seed,
        top_k=top_k,
    )
    sae_features = encode_activations(sae, activations)
    sae_metrics, sae_states, sae_t = state_pipeline_metrics(
        sae_features,
        true_states,
        true_transition,
        seed,
        prefix=prefix,
    )
    sae_metrics.update(
        {
            f"{prefix}_hidden_dim": hidden_dim,
            f"{prefix}_epochs": epochs,
            f"{prefix}_l1_coef": l1_coef,
            f"{prefix}_top_k": 0 if top_k is None else top_k,
            f"{prefix}_reconstruction_mse": sae_result.reconstruction_mse,
            f"{prefix}_mean_l1": sae_result.mean_l1,
            f"{prefix}_active_fraction": sae_result.active_fraction,
        }
    )
    return sae_metrics, sae_states, sae_t


def compact_sae_row(metrics: dict[str, float], prefix: str) -> dict[str, float]:
    keys = [
        "hidden_dim",
        "epochs",
        "l1_coef",
        "top_k",
        "reconstruction_mse",
        "mean_l1",
        "active_fraction",
        "probe_state_recovery_accuracy",
        "cluster_state_recovery_accuracy",
        "rowwise_kl",
        "frobenius_error",
        "stationary_l1",
        "spectral_error",
        "order0_accuracy",
        "order1_accuracy",
        "order2_accuracy",
        "order1_gain_over_order0",
        "order2_gain_over_order1",
    ]
    return {key: metrics[f"{prefix}_{key}"] for key in keys}


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

    final_val_loss = evaluate_loss(model, val_x, val_y, batch_size=args.batch_size)
    acts = collect_activations(
        model,
        val_x,
        activation_name=args.activation_name,
        batch_size=args.batch_size,
    ).numpy()

    residual_metrics, residual_states, residual_t = state_pipeline_metrics(
        acts,
        val_states,
        hmm.transition,
        args.seed,
        prefix="residual",
    )

    beliefs = bayes_filter(hmm, val_tokens)
    belief_metrics = belief_probe_metrics(acts, beliefs, val_states, seed=args.seed)

    bayes_probs = bayes_predictive_distribution(hmm, val_tokens)
    bayes_loss = sequence_cross_entropy(val_tokens, bayes_probs)
    uniform_loss = float(np.log(hmm.vocab_size))
    unigram_probs = unigram_distribution(train_tokens, vocab_size=hmm.vocab_size)
    unigram_loss = repeated_distribution_loss(val_tokens, unigram_probs)

    centroids = state_centroids(acts, residual_states, n_states=hmm.n_states)
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
        "final_val_loss": final_val_loss,
        "bayes_optimal_loss": bayes_loss,
        "uniform_baseline_loss": uniform_loss,
        "unigram_baseline_loss": unigram_loss,
        "model_excess_loss_over_bayes": final_val_loss - bayes_loss,
        **belief_metrics,
        **residual_metrics,
        **true_state_sanity_check(hmm.transition, val_states),
        **shuffled_state_baseline(hmm.transition, residual_states, args.seed + 101),
        **random_state_baseline(hmm.transition, residual_states, args.seed + 202),
        **forced_kl,
    }

    if args.run_sae:
        top_k = None if args.sae_top_k <= 0 else args.sae_top_k
        sae_metrics, _, sae_t = run_sae_state_pipeline(
            acts,
            val_states,
            hmm.transition,
            seed=args.seed,
            hidden_dim=args.sae_hidden_dim,
            l1_coef=args.sae_l1_coef,
            epochs=args.sae_epochs,
            batch_size=max(args.batch_size, 512),
            max_samples=args.sae_max_samples,
            prefix="sae",
            top_k=top_k,
        )
        metrics.update(sae_metrics)
        np.save(out_dir / "sae_estimated_transition.npy", sae_t)
        save_matrix_heatmap(sae_t, "SAE recovered transition", out_dir / "sae_estimated_transition.png")
        save_matrix_heatmap(hmm.transition - sae_t, "SAE transition difference", out_dir / "sae_transition_difference.png")

    if args.run_sae_sweep:
        sweep_rows = []
        best_row = None
        best_metrics = None
        best_t = None
        hidden_dims = parse_int_list(args.sae_sweep_hidden_dims)
        l1_coefs = parse_float_list(args.sae_sweep_l1_coefs)
        top_ks = parse_int_list(args.sae_sweep_top_ks)

        for hidden_dim in hidden_dims:
            for l1_coef in l1_coefs:
                for top_k_value in top_ks:
                    top_k = None if top_k_value <= 0 else top_k_value
                    safe_l1 = str(l1_coef).replace(".", "p").replace("-", "m")
                    prefix = f"sae_sweep_h{hidden_dim}_l1{safe_l1}_k{top_k_value}"
                    candidate_metrics, _, candidate_t = run_sae_state_pipeline(
                        acts,
                        val_states,
                        hmm.transition,
                        seed=args.seed + hidden_dim + int(l1_coef * 1_000_000) + top_k_value,
                        hidden_dim=hidden_dim,
                        l1_coef=l1_coef,
                        epochs=args.sae_epochs,
                        batch_size=max(args.batch_size, 512),
                        max_samples=args.sae_max_samples,
                        prefix=prefix,
                        top_k=top_k,
                    )
                    row = compact_sae_row(candidate_metrics, prefix)
                    sweep_rows.append(row)
                    if best_row is None or row["rowwise_kl"] < best_row["rowwise_kl"]:
                        best_row = row
                        best_metrics = candidate_metrics
                        best_t = candidate_t

        if best_row is not None and best_metrics is not None and best_t is not None:
            (out_dir / "sae_sweep_results.json").write_text(json.dumps(sweep_rows, indent=2))
            metrics.update(
                {
                    "best_sae_hidden_dim": best_row["hidden_dim"],
                    "best_sae_l1_coef": best_row["l1_coef"],
                    "best_sae_top_k": best_row["top_k"],
                    "best_sae_epochs": best_row["epochs"],
                    "best_sae_reconstruction_mse": best_row["reconstruction_mse"],
                    "best_sae_active_fraction": best_row["active_fraction"],
                    "best_sae_probe_state_recovery_accuracy": best_row["probe_state_recovery_accuracy"],
                    "best_sae_cluster_state_recovery_accuracy": best_row["cluster_state_recovery_accuracy"],
                    "best_sae_rowwise_kl": best_row["rowwise_kl"],
                    "best_sae_frobenius_error": best_row["frobenius_error"],
                    "best_sae_stationary_l1": best_row["stationary_l1"],
                    "best_sae_spectral_error": best_row["spectral_error"],
                    "best_sae_order1_gain_over_order0": best_row["order1_gain_over_order0"],
                    "best_sae_order2_gain_over_order1": best_row["order2_gain_over_order1"],
                }
            )
            np.save(out_dir / "best_sae_estimated_transition.npy", best_t)
            save_matrix_heatmap(best_t, "Best SAE recovered transition", out_dir / "best_sae_estimated_transition.png")
            save_matrix_heatmap(hmm.transition - best_t, "Best SAE transition difference", out_dir / "best_sae_transition_difference.png")

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    np.save(out_dir / "true_transition.npy", hmm.transition)
    np.save(out_dir / "estimated_transition.npy", residual_t)
    save_matrix_heatmap(hmm.transition, "True HMM transition", out_dir / "true_transition.png")
    save_matrix_heatmap(residual_t, "Residual recovered transition", out_dir / "estimated_transition.png")
    save_matrix_heatmap(hmm.transition - residual_t, "Residual transition difference", out_dir / "transition_difference.png")
    save_loss_curve(result.train_loss, result.val_loss, out_dir / "loss_curve.png")

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

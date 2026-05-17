from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from mct.baselines import (
    belief_cluster_states,
    cluster_belief_reconstruction,
    evaluate_representation_for_k,
    next_true_state_nll,
    token_bigram_state,
    token_state,
    transition_if_same_k,
)
from mct.data import (
    bayes_filter,
    bayes_predictive_distribution,
    make_lm_tensors,
    sample_hmm_sequences,
    sequence_cross_entropy,
    unigram_distribution,
)
from mct.hmm_families import FAMILIES, emission_entropy, make_hmm_family, transition_entropy
from mct.markov_models import bootstrap_markov_nll, markov_nll_report
from mct.model import TinyTransformer, TransformerConfig, count_parameters
from mct.probes import belief_probe_metrics
from mct.states import best_label_match, cluster_internal_states, probe_state_recovery
from mct.train import collect_activations, evaluate_loss, train_model
from mct.transition import estimate_transition_matrix, transition_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the TMLR-grade MCT benchmark suite")
    parser.add_argument("--output-dir", default="runs/tmlr_benchmark_suite")
    parser.add_argument("--families", default="easy_separable,ambiguous_emissions,persistent,high_entropy,three_state,six_state")
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--cluster-ks", default="2,3,4,5,6,8,10")
    parser.add_argument("--layers", default="embed,resid_post_0,resid_post_1,ln_final")
    parser.add_argument("--train-sequences", type=int, default=6000)
    parser.add_argument("--val-sequences", type=int, default=1500)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--d-mlp", type=int, default=256)
    parser.add_argument("--num-threads", type=int, default=2)
    parser.add_argument("--bootstrap", type=int, default=100)
    return parser.parse_args()


def parse_str_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def repeated_distribution_loss(tokens: np.ndarray, distribution: np.ndarray) -> float:
    probs = np.broadcast_to(distribution, (*tokens.shape, distribution.shape[0]))
    return sequence_cross_entropy(tokens, probs)


def flatten_json_rows(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    keys = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def state_metrics_for_true_k(
    activations: np.ndarray,
    true_states: np.ndarray,
    true_transition: np.ndarray,
    seed: int,
) -> tuple[dict[str, float], np.ndarray]:
    k = true_transition.shape[0]
    probe_acc = probe_state_recovery(activations, true_states, seed=seed)
    discovered = cluster_internal_states(activations, n_states=k, seed=seed)
    remapped, cluster_acc = best_label_match(discovered, true_states, n_states=k)
    estimated_t = estimate_transition_matrix(remapped, n_states=k)
    return {
        "probe_state_recovery_accuracy": probe_acc,
        "cluster_state_recovery_accuracy": cluster_acc,
        **transition_report(true_transition, estimated_t),
        **markov_nll_report(remapped, n_states=k),
        **bootstrap_markov_nll(remapped, n_states=k, n_bootstrap=100, seed=seed + 1000),
    }, remapped


def token_baseline_rows(
    tokens: np.ndarray,
    true_states: np.ndarray,
    beliefs: np.ndarray,
    true_transition: np.ndarray,
    family: str,
    seed: int,
) -> list[dict]:
    rows = []
    baselines = {
        "token_current": token_state(tokens),
        "token_bigram": token_bigram_state(tokens, vocab_size=int(tokens.max()) + 1),
    }
    for name, states in baselines.items():
        k = int(states.max()) + 1
        row = {
            "family": family,
            "seed": seed,
            "method": name,
            "k": k,
            "next_true_state_nll": next_true_state_nll(states, true_states),
            **cluster_belief_reconstruction(states, beliefs),
            **markov_nll_report(states, n_states=k),
        }
        rows.append(row)
    return rows


def run_one_family_seed(args: argparse.Namespace, family: str, seed: int, out_dir: Path) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    hmm = make_hmm_family(family, seed=seed)
    bos_token = hmm.vocab_size

    train_tokens, _ = sample_hmm_sequences(hmm, args.train_sequences, args.seq_len, seed=seed)
    val_tokens, val_states = sample_hmm_sequences(hmm, args.val_sequences, args.seq_len, seed=seed + 1)
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

    beliefs = bayes_filter(hmm, val_tokens)
    bayes_probs = bayes_predictive_distribution(hmm, val_tokens)
    bayes_loss = sequence_cross_entropy(val_tokens, bayes_probs)
    unigram_probs = unigram_distribution(train_tokens, vocab_size=hmm.vocab_size)
    unigram_loss = repeated_distribution_loss(val_tokens, unigram_probs)

    family_seed_dir = out_dir / family / f"seed_{seed}"
    family_seed_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "family": family,
        "seed": seed,
        "n_states": hmm.n_states,
        "vocab_size": hmm.vocab_size,
        "transition_entropy": transition_entropy(hmm),
        "emission_entropy": emission_entropy(hmm),
        "parameter_count": count_parameters(model),
        "final_train_loss": result.train_loss[-1],
        "final_val_loss": final_val_loss,
        "bayes_optimal_loss": bayes_loss,
        "unigram_baseline_loss": unigram_loss,
        "uniform_baseline_loss": float(np.log(hmm.vocab_size)),
        "model_excess_loss_over_bayes": final_val_loss - bayes_loss,
    }

    layer_rows = []
    k_rows = []
    baseline_rows = token_baseline_rows(val_tokens, val_states, beliefs, hmm.transition, family, seed)

    for layer in parse_str_list(args.layers):
        acts = collect_activations(model, val_x, activation_name=layer, batch_size=args.batch_size).numpy()
        belief_metrics = belief_probe_metrics(acts, beliefs, val_states, seed=seed)
        state_metrics, recovered_states = state_metrics_for_true_k(acts, val_states, hmm.transition, seed=seed)
        layer_row = {
            "family": family,
            "seed": seed,
            "layer": layer,
            **belief_metrics,
            **state_metrics,
        }
        layer_rows.append(layer_row)

        if layer == "resid_post_1":
            summary.update({f"main_{key}": value for key, value in belief_metrics.items()})
            summary.update({f"main_{key}": value for key, value in state_metrics.items()})

            for method in ["residual_kmeans", "pca_kmeans", "random_projection_kmeans", "belief_kmeans"]:
                for k in parse_int_list(args.cluster_ks):
                    row = evaluate_representation_for_k(
                        acts,
                        val_states,
                        beliefs,
                        hmm.transition,
                        k=k,
                        seed=seed,
                        method=method,
                        pca_dim=min(8, acts.shape[-1]),
                    )
                    row.update({"family": family, "seed": seed, "layer": layer})
                    k_rows.append(row)

            true_state_metrics = {
                "family": family,
                "seed": seed,
                "method": "true_state_oracle",
                "k": hmm.n_states,
                "next_true_state_nll": next_true_state_nll(val_states, val_states),
                **transition_if_same_k(hmm.transition, val_states, val_states, k=hmm.n_states),
                **markov_nll_report(val_states, n_states=hmm.n_states),
            }
            baseline_rows.append(true_state_metrics)

    (family_seed_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (family_seed_dir / "layer_rows.json").write_text(json.dumps(layer_rows, indent=2))
    (family_seed_dir / "k_sensitivity_rows.json").write_text(json.dumps(k_rows, indent=2))
    (family_seed_dir / "baseline_rows.json").write_text(json.dumps(baseline_rows, indent=2))

    return {
        "summary": summary,
        "layers": layer_rows,
        "k_rows": k_rows,
        "baselines": baseline_rows,
    }


def aggregate_numeric(rows: list[dict], group_keys: list[str]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        key = tuple(row.get(k) for k in group_keys)
        groups.setdefault(key, []).append(row)

    out = []
    for key, items in groups.items():
        base = {name: value for name, value in zip(group_keys, key, strict=False)}
        numeric_keys = sorted({k for item in items for k, v in item.items() if isinstance(v, (int, float)) and not isinstance(v, bool)})
        for metric in numeric_keys:
            values = [float(item[metric]) for item in items if metric in item and isinstance(item[metric], (int, float))]
            if not values:
                continue
            mean = float(np.mean(values))
            std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            out.append({**base, "metric": metric, "mean": mean, "std": std, "n": len(values)})
    return out


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.num_threads)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    families = parse_str_list(args.families)
    seeds = parse_int_list(args.seeds)
    for family in families:
        if family not in FAMILIES:
            raise KeyError(f"Unknown family {family}")

    summaries = []
    layer_rows = []
    k_rows = []
    baseline_rows = []

    for family in families:
        for seed in seeds:
            result = run_one_family_seed(args, family, seed, out_dir)
            summaries.append(result["summary"])
            layer_rows.extend(result["layers"])
            k_rows.extend(result["k_rows"])
            baseline_rows.extend(result["baselines"])

    (out_dir / "all_summaries.json").write_text(json.dumps(summaries, indent=2))
    (out_dir / "all_layer_rows.json").write_text(json.dumps(layer_rows, indent=2))
    (out_dir / "all_k_sensitivity_rows.json").write_text(json.dumps(k_rows, indent=2))
    (out_dir / "all_baseline_rows.json").write_text(json.dumps(baseline_rows, indent=2))

    flatten_json_rows(summaries, out_dir / "all_summaries.csv")
    flatten_json_rows(layer_rows, out_dir / "all_layer_rows.csv")
    flatten_json_rows(k_rows, out_dir / "all_k_sensitivity_rows.csv")
    flatten_json_rows(baseline_rows, out_dir / "all_baseline_rows.csv")

    summary_agg = aggregate_numeric(summaries, group_keys=["family"])
    layer_agg = aggregate_numeric(layer_rows, group_keys=["family", "layer"])
    k_agg = aggregate_numeric(k_rows, group_keys=["family", "method", "k"])
    baseline_agg = aggregate_numeric(baseline_rows, group_keys=["family", "method"])

    flatten_json_rows(summary_agg, out_dir / "summary_aggregate.csv")
    flatten_json_rows(layer_agg, out_dir / "layer_aggregate.csv")
    flatten_json_rows(k_agg, out_dir / "k_sensitivity_aggregate.csv")
    flatten_json_rows(baseline_agg, out_dir / "baseline_aggregate.csv")

    (out_dir / "summary_aggregate.json").write_text(json.dumps(summary_agg, indent=2))
    (out_dir / "layer_aggregate.json").write_text(json.dumps(layer_agg, indent=2))
    (out_dir / "k_sensitivity_aggregate.json").write_text(json.dumps(k_agg, indent=2))
    (out_dir / "baseline_aggregate.json").write_text(json.dumps(baseline_agg, indent=2))

    print(json.dumps({"n_summaries": len(summaries), "n_layer_rows": len(layer_rows), "n_k_rows": len(k_rows)}, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from mct.data import forced_state_next_token_distribution, make_lm_tensors, sample_hmm_sequences
from mct.hmm_families import FAMILIES, make_hmm_family
from mct.interventions import state_forcing_control_report
from mct.model import TinyTransformer, TransformerConfig
from mct.states import best_label_match, cluster_internal_states, state_centroids
from mct.train import collect_activations, train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run state-forcing controls for MCT")
    parser.add_argument("--output-dir", default="runs/state_forcing_controls")
    parser.add_argument("--families", default="easy_separable,ambiguous_emissions,persistent,high_entropy,three_state,six_state")
    parser.add_argument("--seeds", default="1,2,3")
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
    parser.add_argument("--activation-name", default="resid_post_1")
    parser.add_argument("--forcing-position", type=int, default=20)
    parser.add_argument("--forcing-samples", type=int, default=256)
    parser.add_argument("--num-threads", type=int, default=2)
    return parser.parse_args()


def parse_str_list(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def parse_int_list(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def pad_visible_distribution_for_bos(visible_probs: np.ndarray, bos_token: int) -> np.ndarray:
    padded = np.zeros((visible_probs.shape[0], bos_token + 1), dtype=visible_probs.dtype)
    padded[:, :bos_token] = visible_probs
    return padded


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def aggregate(rows: list[dict], group_keys: list[str]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        key = tuple(row.get(k) for k in group_keys)
        groups.setdefault(key, []).append(row)
    out = []
    for key, items in groups.items():
        base = {name: value for name, value in zip(group_keys, key, strict=False)}
        numeric_keys = sorted({k for item in items for k, v in item.items() if isinstance(v, (int, float)) and not isinstance(v, bool)})
        for metric in numeric_keys:
            values = [float(item[metric]) for item in items if metric in item]
            mean = float(np.mean(values))
            std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            out.append({**base, "metric": metric, "mean": mean, "std": std, "n": len(values)})
    return out


def run_one(args: argparse.Namespace, family: str, seed: int) -> list[dict]:
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

    train_model(
        model,
        train_x,
        train_y,
        val_x,
        val_y,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )

    acts = collect_activations(model, val_x, activation_name=args.activation_name, batch_size=args.batch_size).numpy()
    discovered = cluster_internal_states(acts, n_states=hmm.n_states, seed=seed)
    recovered_states, cluster_acc = best_label_match(discovered, val_states, n_states=hmm.n_states)

    recovered_centroids = state_centroids(acts, recovered_states, n_states=hmm.n_states)
    true_centroids = state_centroids(acts, val_states, n_states=hmm.n_states)
    ideal_visible = np.stack([forced_state_next_token_distribution(hmm, s) for s in range(hmm.n_states)], axis=0)
    ideal_forced = pad_visible_distribution_for_bos(ideal_visible, bos_token=bos_token)

    n_force = min(args.forcing_samples, val_x.shape[0])
    rows = state_forcing_control_report(
        model,
        val_x[:n_force],
        recovered_centroids,
        true_centroids,
        ideal_forced,
        activation_name=args.activation_name,
        position=min(args.forcing_position, args.seq_len - 1),
        seed=seed,
        batch_size=min(args.batch_size, 128),
    )
    for row in rows:
        row.update({
            "family": family,
            "seed": seed,
            "activation_name": args.activation_name,
            "cluster_accuracy": cluster_acc,
        })
    return rows


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.num_threads)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    families = parse_str_list(args.families)
    seeds = parse_int_list(args.seeds)
    for family in families:
        if family not in FAMILIES:
            raise KeyError(family)

    rows = []
    for family in families:
        for seed in seeds:
            rows.extend(run_one(args, family, seed))

    agg_family_patch = aggregate(rows, ["family", "patch_type"])
    agg_patch = aggregate(rows, ["patch_type"])
    (out_dir / "forcing_rows.json").write_text(json.dumps(rows, indent=2))
    (out_dir / "forcing_aggregate_by_family.json").write_text(json.dumps(agg_family_patch, indent=2))
    (out_dir / "forcing_aggregate_overall.json").write_text(json.dumps(agg_patch, indent=2))
    write_csv(rows, out_dir / "forcing_rows.csv")
    write_csv(agg_family_patch, out_dir / "forcing_aggregate_by_family.csv")
    write_csv(agg_patch, out_dir / "forcing_aggregate_overall.csv")
    print(json.dumps({"n_rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()

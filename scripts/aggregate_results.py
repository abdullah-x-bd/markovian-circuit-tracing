from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

METRIC_PATHS = {
    "actual_epochs": ("model_quality", "actual_training_epochs"),
    "validation_gap": ("model_quality", "model_validation_excess_over_bayes"),
    "eval_loss": ("model_quality", "untouched_evaluation_loss"),
    "bayes_loss": ("model_quality", "bayes_optimal_evaluation_loss"),
    "unigram_loss": ("model_quality", "unigram_baseline_loss"),
    "excess_loss": ("model_quality", "model_excess_loss_over_bayes"),
    "belief_kl": ("belief_recovery", "belief_probe_kl"),
    "belief_r2": ("belief_recovery", "belief_probe_r2"),
    "belief_argmax_acc": ("belief_recovery", "belief_probe_argmax_accuracy"),
    "state_probe_acc": ("sampled_state_recovery", "state_probe_accuracy"),
    "cluster_state_acc": ("sampled_state_recovery", "cluster_state_recovery_accuracy"),
    "bayes_state_ceiling": ("sampled_state_recovery", "bayes_observable_state_accuracy_ceiling"),
    "transition_kl": ("transition_recovery", "rowwise_kl"),
    "transition_fro": ("transition_recovery", "frobenius_error"),
    "shuffled_transition_kl": ("transition_recovery", "shuffled_rowwise_kl"),
    "random_transition_kl": ("transition_recovery", "random_rowwise_kl"),
    "order1_gain": ("markov_tests", "order1_gain_over_order0"),
    "order2_gain": ("markov_tests", "order2_gain_over_order1"),
    "history1_belief_kl": ("token_history_baselines", "history1_belief_probe_kl"),
    "history2_belief_kl": ("token_history_baselines", "history2_belief_probe_kl"),
    "history4_belief_kl": ("token_history_baselines", "history4_belief_probe_kl"),
    "untrained_belief_kl": ("untrained_transformer_baseline", "belief_probe_kl"),
    "same_scrub_kl": ("causal_scrubbing", "same_state_swap_kl_mean"),
    "different_scrub_kl": ("causal_scrubbing", "different_state_swap_kl_mean"),
    "scrub_gap": ("causal_scrubbing", "different_minus_same_kl"),
    "sae_transition_kl": ("sae", "heldout_evaluation", "rowwise_kl"),
    "sae_belief_kl": ("sae", "heldout_evaluation", "belief_probe_kl"),
    "sae_cluster_state_acc": ("sae", "heldout_evaluation", "cluster_state_recovery_accuracy"),
}


def nested(obj: dict, path: tuple[str, ...]):
    cur = obj
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def ci95(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return 1.96 * stdev(values) / (len(values) ** 0.5)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=Path, default=Path("results/v1/runs"))
    p.add_argument("--output", type=Path, default=Path("results/v1"))
    args = p.parse_args()
    run_rows = []
    forcing_rows = []
    for run_dir in sorted(args.runs.glob("*_seed*")):
        metrics_path = run_dir / "metrics.json"
        if not metrics_path.exists():
            continue
        m = json.loads(metrics_path.read_text())
        row = {"observability": m["run_metadata"]["hmm_observability"], "seed": int(m["run_metadata"]["seed"])}
        for name, path in METRIC_PATHS.items():
            value = nested(m, path)
            row[name] = "" if value is None else float(value)
        run_rows.append(row)
        controls = json.loads((run_dir / "forcing_controls.json").read_text())
        for control in controls:
            forcing_rows.append({
                "observability": row["observability"],
                "seed": row["seed"],
                "target_state": int(control["target_state"]),
                "patch_type": control["patch_type"],
                "kl_to_target": float(control["kl_to_target"]),
                "improvement_over_unpatched": float(control["improvement_over_unpatched"]),
            })

    expected = {(o, s) for o in ("easy", "medium", "hard") for s in (7, 17, 29, 43, 71)}
    found = {(r["observability"], r["seed"]) for r in run_rows}
    if found != expected:
        raise RuntimeError(f"Canonical run set mismatch. missing={sorted(expected-found)}, extra={sorted(found-expected)}")

    summary_rows = []
    by_obs = defaultdict(list)
    for row in run_rows:
        by_obs[row["observability"]].append(row)
    for obs in ("easy", "medium", "hard"):
        rows = by_obs[obs]
        summary = {"observability": obs, "n_seeds": len(rows)}
        for metric in METRIC_PATHS:
            vals = [float(r[metric]) for r in rows if r[metric] != ""]
            if vals:
                summary[f"{metric}_mean"] = mean(vals)
                summary[f"{metric}_sd"] = stdev(vals) if len(vals) > 1 else 0.0
                summary[f"{metric}_ci95"] = ci95(vals)
            else:
                summary[f"{metric}_mean"] = ""
                summary[f"{metric}_sd"] = ""
                summary[f"{metric}_ci95"] = ""
        summary_rows.append(summary)

    forcing_summary = []
    groups = defaultdict(list)
    for row in forcing_rows:
        groups[(row["observability"], row["patch_type"])].append(row["kl_to_target"])
    patch_types = sorted({r["patch_type"] for r in forcing_rows})
    for obs in ("easy", "medium", "hard"):
        for patch_type in patch_types:
            vals = groups[(obs, patch_type)]
            if not vals:
                continue
            forcing_summary.append({
                "observability": obs,
                "patch_type": patch_type,
                "n": len(vals),
                "kl_mean": mean(vals),
                "kl_sd": stdev(vals) if len(vals) > 1 else 0.0,
                "kl_ci95": ci95(vals),
            })

    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "tables" / "run_metrics.csv", run_rows)
    write_csv(args.output / "tables" / "summary_by_observability.csv", summary_rows)
    write_csv(args.output / "tables" / "forcing_controls.csv", forcing_rows)
    write_csv(args.output / "tables" / "forcing_summary.csv", forcing_summary)
    (args.output / "summary.json").write_text(json.dumps({
        "artifact_schema_version": "1.0",
        "canonical_runs": len(run_rows),
        "observability_summary": summary_rows,
        "forcing_summary": forcing_summary,
    }, indent=2))
    print(f"Aggregated {len(run_rows)} runs into {args.output}")


if __name__ == "__main__":
    main()

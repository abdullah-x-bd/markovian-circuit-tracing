from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    m = mean(values)
    return (sum((x - m) ** 2 for x in values) / (len(values) - 1)) ** 0.5


def main() -> None:
    args = parse_args()
    runs_dir = Path(args.runs_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for path in sorted(runs_dir.glob("seed_*/metrics.json")):
        item = json.loads(path.read_text())
        item["run_dir"] = str(path.parent)
        rows.append(item)

    if not rows:
        raise FileNotFoundError("No seed metrics found")

    numeric_keys = sorted({key for row in rows for key, value in row.items() if is_number(value)})
    summary = {}
    for key in numeric_keys:
        values = [float(row[key]) for row in rows if key in row and is_number(row[key])]
        summary[key] = {
            "mean": mean(values),
            "std": std(values),
            "min": min(values),
            "max": max(values),
            "n": len(values),
        }

    (out_dir / "seed_sweep_summary.json").write_text(json.dumps(summary, indent=2))

    with (out_dir / "seed_sweep_runs.csv").open("w", newline="") as f:
        fieldnames = ["run_dir", *numeric_keys]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    with (out_dir / "seed_sweep_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "mean", "std", "min", "max", "n"])
        writer.writeheader()
        for key, value in summary.items():
            writer.writerow({"metric": key, **value})

    important = [
        "final_val_loss",
        "bayes_optimal_loss",
        "model_excess_loss_over_bayes",
        "belief_probe_mse",
        "belief_probe_kl",
        "belief_probe_argmax_accuracy",
        "belief_probe_true_state_accuracy",
        "residual_probe_state_recovery_accuracy",
        "residual_cluster_state_recovery_accuracy",
        "residual_rowwise_kl",
        "residual_frobenius_error",
        "shuffled_rowwise_kl",
        "random_rowwise_kl",
        "residual_order1_gain_over_order0",
        "residual_order2_gain_over_order1",
    ]
    lines = ["# Seed sweep summary", "", "| Metric | Mean | Std | Min | Max | N |", "|---|---:|---:|---:|---:|---:|"]
    for key in important:
        if key in summary:
            s = summary[key]
            lines.append(f"| {key} | {s['mean']:.6f} | {s['std']:.6f} | {s['min']:.6f} | {s['max']:.6f} | {int(s['n'])} |")
    (out_dir / "seed_sweep_summary.md").write_text("\n".join(lines) + "\n")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

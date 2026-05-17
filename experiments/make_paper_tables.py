from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create paper-ready tables from MCT benchmark outputs")
    parser.add_argument("--benchmark-dir", required=True)
    parser.add_argument("--forcing-dir", default="")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def number(value: str) -> float:
    return float(value)


def fmt(mean: float, std: float) -> str:
    return f"{mean:.4f} ± {std:.4f}"


def lookup(rows: list[dict[str, str]], **filters: str) -> list[dict[str, str]]:
    out = []
    for row in rows:
        ok = True
        for key, value in filters.items():
            if row.get(key) != str(value):
                ok = False
                break
        if ok:
            out.append(row)
    return out


def metric_map(rows: list[dict[str, str]], group_key: str) -> dict[tuple[str, str], tuple[float, float]]:
    out = {}
    for row in rows:
        out[(row.get(group_key, ""), row["metric"])] = (number(row["mean"]), number(row["std"]))
    return out


def make_family_summary(benchmark_dir: Path) -> list[dict]:
    rows = read_csv(benchmark_dir / "summary_aggregate.csv")
    out = []
    for family in sorted({row["family"] for row in rows}):
        fam = {row["metric"]: row for row in rows if row["family"] == family}
        out.append(
            {
                "family": family,
                "excess_loss_over_bayes": fmt(number(fam["model_excess_loss_over_bayes"]["mean"]), number(fam["model_excess_loss_over_bayes"]["std"])),
                "belief_probe_kl": fmt(number(fam["main_belief_probe_kl"]["mean"]), number(fam["main_belief_probe_kl"]["std"])),
                "cluster_accuracy": fmt(number(fam["main_cluster_state_recovery_accuracy"]["mean"]), number(fam["main_cluster_state_recovery_accuracy"]["std"])),
                "transition_rowwise_kl": fmt(number(fam["main_rowwise_kl"]["mean"]), number(fam["main_rowwise_kl"]["std"])),
                "markov_nll_gain_0_to_1": fmt(number(fam["main_markov_nll_gain_0_to_1"]["mean"]), number(fam["main_markov_nll_gain_0_to_1"]["std"])),
                "markov_nll_gain_1_to_2": fmt(number(fam["main_markov_nll_gain_1_to_2"]["mean"]), number(fam["main_markov_nll_gain_1_to_2"]["std"])),
            }
        )
    return out


def make_true_k_baseline_table(benchmark_dir: Path) -> list[dict]:
    rows = read_csv(benchmark_dir / "k_sensitivity_aggregate.csv")
    out = []
    methods = ["belief_kmeans", "residual_kmeans", "pca_kmeans", "random_projection_kmeans"]
    for family in sorted({row["family"] for row in rows}):
        family_rows = [row for row in rows if row["family"] == family]
        true_k_candidates = [row for row in family_rows if row["metric"] == "cluster_accuracy"]
        true_ks = sorted({row["k"] for row in true_k_candidates}, key=lambda x: int(float(x)))
        if not true_ks:
            continue
        # The true-K rows are the only rows that have cluster_accuracy and transition KL.
        true_k = true_ks[0]
        for method in methods:
            row_kl = lookup(rows, family=family, method=method, k=true_k, metric="rowwise_kl")
            row_belief = lookup(rows, family=family, method=method, k=true_k, metric="belief_reconstruction_kl")
            row_next = lookup(rows, family=family, method=method, k=true_k, metric="next_true_state_nll")
            if row_kl:
                out.append(
                    {
                        "family": family,
                        "method": method,
                        "k": true_k,
                        "rowwise_kl": fmt(number(row_kl[0]["mean"]), number(row_kl[0]["std"])),
                        "belief_reconstruction_kl": fmt(number(row_belief[0]["mean"]), number(row_belief[0]["std"])) if row_belief else "",
                        "next_true_state_nll": fmt(number(row_next[0]["mean"]), number(row_next[0]["std"])) if row_next else "",
                    }
                )
    return out


def make_layer_table(benchmark_dir: Path) -> list[dict]:
    rows = read_csv(benchmark_dir / "layer_aggregate.csv")
    out = []
    for family in sorted({row["family"] for row in rows}):
        for layer in ["embed", "resid_post_0", "resid_post_1", "ln_final"]:
            row_kl = lookup(rows, family=family, layer=layer, metric="rowwise_kl")
            row_belief = lookup(rows, family=family, layer=layer, metric="belief_probe_kl")
            row_cluster = lookup(rows, family=family, layer=layer, metric="cluster_state_recovery_accuracy")
            if row_kl:
                out.append(
                    {
                        "family": family,
                        "layer": layer,
                        "belief_probe_kl": fmt(number(row_belief[0]["mean"]), number(row_belief[0]["std"])),
                        "cluster_accuracy": fmt(number(row_cluster[0]["mean"]), number(row_cluster[0]["std"])),
                        "transition_rowwise_kl": fmt(number(row_kl[0]["mean"]), number(row_kl[0]["std"])),
                    }
                )
    return out


def make_forcing_table(forcing_dir: Path) -> list[dict]:
    path = forcing_dir / "forcing_aggregate_overall.csv"
    if not path.exists():
        return []
    rows = read_csv(path)
    out = []
    for patch_type in sorted({row["patch_type"] for row in rows}):
        kl = lookup(rows, patch_type=patch_type, metric="kl_to_target")
        improvement = lookup(rows, patch_type=patch_type, metric="improvement_over_unpatched")
        if kl:
            out.append(
                {
                    "patch_type": patch_type,
                    "kl_to_target": fmt(number(kl[0]["mean"]), number(kl[0]["std"])),
                    "improvement_over_unpatched": fmt(number(improvement[0]["mean"]), number(improvement[0]["std"])) if improvement else "",
                }
            )
    return out


def markdown_table(rows: list[dict], title: str) -> str:
    if not rows:
        return f"## {title}\n\nNo rows available.\n"
    keys = list(rows[0].keys())
    lines = [f"## {title}", "", "| " + " | ".join(keys) + " |", "|" + "|".join(["---" for _ in keys]) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(key, "")) for key in keys) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    benchmark_dir = Path(args.benchmark_dir)
    forcing_dir = Path(args.forcing_dir) if args.forcing_dir else Path("")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    family_summary = make_family_summary(benchmark_dir)
    true_k_baselines = make_true_k_baseline_table(benchmark_dir)
    layer_table = make_layer_table(benchmark_dir)
    forcing_table = make_forcing_table(forcing_dir) if args.forcing_dir else []

    write_csv(family_summary, out_dir / "paper_table_family_summary.csv")
    write_csv(true_k_baselines, out_dir / "paper_table_true_k_baselines.csv")
    write_csv(layer_table, out_dir / "paper_table_layer_sweep.csv")
    write_csv(forcing_table, out_dir / "paper_table_forcing_controls.csv")

    md = "\n".join(
        [
            markdown_table(family_summary, "Family summary"),
            markdown_table(true_k_baselines, "True-K baseline comparison"),
            markdown_table(layer_table, "Layer sweep"),
            markdown_table(forcing_table, "State-forcing controls"),
        ]
    )
    (out_dir / "paper_tables.md").write_text(md)
    (out_dir / "paper_tables.json").write_text(json.dumps(
        {
            "family_summary": family_summary,
            "true_k_baselines": true_k_baselines,
            "layer_table": layer_table,
            "forcing_table": forcing_table,
        },
        indent=2,
    ))
    print(json.dumps({"family_rows": len(family_summary), "baseline_rows": len(true_k_baselines)}, indent=2))


if __name__ == "__main__":
    main()

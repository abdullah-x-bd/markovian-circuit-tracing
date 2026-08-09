from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml


def load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def command_for(config: dict, observability: str, seed: int, root: Path) -> list[str]:
    model = config["model"]
    training = config["training"]
    rep = config["representation"]
    inter = config["interventions"]
    sae = config["sae"]
    out = root / f"{observability}_seed{seed:02d}"
    cmd = [
        sys.executable,
        "experiments/run_full_hmm_pipeline.py",
        "--run-kind", "confirmatory",
        "--hmm-observability", observability,
        "--seed", str(seed),
        "--output-dir", str(out),
        "--seq-len", str(training["seq_len"]),
        "--train-sequences", str(training["train_sequences"]),
        "--model-val-sequences", str(training["model_val_sequences"]),
        "--analysis-sequences", str(training["analysis_sequences"]),
        "--epochs", str(training["max_epochs"]),
        "--min-epochs", str(training["min_epochs"]),
        "--bayes-gap-target", str(training["bayes_gap_target"]),
        "--batch-size", str(training["batch_size"]),
        "--lr", str(training["learning_rate"]),
        "--d-model", str(model["d_model"]),
        "--n-layers", str(model["n_layers"]),
        "--n-heads", str(model["n_heads"]),
        "--d-mlp", str(model["d_mlp"]),
        "--activation-name", str(rep["activation_name"]),
        "--forcing-position", str(inter["forcing_position"]),
        "--forcing-samples", str(inter["forcing_samples"]),
        "--scrubbing-pairs", str(inter["scrubbing_pairs"]),
        "--num-threads", "2",
    ]
    if sae.get("enabled", False):
        cmd += [
            "--run-sae",
            "--sae-hidden-dim", str(sae["hidden_dim"]),
            "--sae-epochs", str(sae["epochs"]),
            "--sae-l1-coef", str(sae["l1_coef"]),
            "--sae-top-k", str(sae["top_k"]),
            "--sae-max-samples", str(sae["max_samples"]),
        ]
    return cmd


def is_complete(run_dir: Path) -> bool:
    required = ["metrics.json", "config.json", "forcing_controls.json", "true_transition.npy", "recovered_transition.npy"]
    return all((run_dir / name).exists() for name in required)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the frozen MCT confirmatory benchmark suite")
    parser.add_argument("--config", type=Path, default=Path("configs/main_experiment.yaml"))
    parser.add_argument("--output-root", type=Path, default=Path("results/v1/runs"))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--only-observability", choices=("easy", "medium", "hard"), default=None)
    parser.add_argument("--only-seed", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    args.output_root.mkdir(parents=True, exist_ok=True)
    observabilities = config["observability_sweep"]
    seeds = config["seeds"]
    if args.only_observability:
        observabilities = [args.only_observability]
    if args.only_seed is not None:
        seeds = [args.only_seed]

    manifest = []
    for observability in observabilities:
        for seed in seeds:
            run_dir = args.output_root / f"{observability}_seed{seed:02d}"
            if is_complete(run_dir) and not args.force:
                print(f"SKIP complete {run_dir}", flush=True)
                manifest.append({"observability": observability, "seed": seed, "status": "existing"})
                continue
            print(f"RUN {observability} seed={seed}", flush=True)
            cmd = command_for(config, observability, seed, args.output_root)
            subprocess.run(cmd, check=True)
            if not is_complete(run_dir):
                raise RuntimeError(f"Run did not produce complete artifact: {run_dir}")
            manifest.append({"observability": observability, "seed": seed, "status": "completed"})

    (args.output_root.parent / "suite_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Completed {len(manifest)} canonical runs.")


if __name__ == "__main__":
    main()

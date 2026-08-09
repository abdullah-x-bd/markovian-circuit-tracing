import json
import subprocess
import sys
from pathlib import Path


def test_pipeline_smoke(tmp_path: Path):
    cmd = [
        sys.executable,
        "experiments/run_full_hmm_pipeline.py",
        "--output-dir", str(tmp_path / "out"),
        "--seed", "7",
        "--seq-len", "12",
        "--train-sequences", "120",
        "--model-val-sequences", "40",
        "--analysis-sequences", "60",
        "--epochs", "1",
        "--batch-size", "32",
        "--d-model", "16",
        "--n-layers", "1",
        "--n-heads", "2",
        "--d-mlp", "32",
        "--activation-name", "resid_post_0",
        "--forcing-position", "4",
        "--forcing-samples", "8",
        "--scrubbing-pairs", "8",
        "--num-threads", "1",
    ]
    subprocess.run(cmd, check=True)
    metrics = json.loads((tmp_path / "out" / "metrics.json").read_text())
    assert metrics["run_metadata"]["sequence_counts"]["evaluation"] == 12
    assert metrics["intervention_summary"]["target_semantics"] == "P(x_t | s_t=k) = E[k]"
    assert "belief_probe_kl" in metrics["belief_recovery"]
    assert "different_minus_same_kl" in metrics["causal_scrubbing"]

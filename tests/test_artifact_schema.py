import json
import subprocess
import sys
from pathlib import Path


def test_smoke_run_writes_versioned_environment_and_training_history(tmp_path: Path):
    out = tmp_path / "run"
    subprocess.run([
        sys.executable, "experiments/run_full_hmm_pipeline.py",
        "--output-dir", str(out), "--seq-len", "10", "--train-sequences", "80",
        "--model-val-sequences", "30", "--analysis-sequences", "50", "--epochs", "1", "--min-epochs", "1", "--bayes-gap-target", "10",
        "--batch-size", "32", "--d-model", "16", "--n-layers", "1", "--n-heads", "2",
        "--d-mlp", "32", "--activation-name", "resid_post_0", "--forcing-position", "3",
        "--forcing-samples", "6", "--scrubbing-pairs", "6", "--num-threads", "1",
    ], check=True)
    m = json.loads((out / "metrics.json").read_text())
    env = json.loads((out / "environment.json").read_text())
    hist = json.loads((out / "training_history.json").read_text())
    assert m["artifact_schema_version"] == "1.0"
    assert "python" in env and "torch" in env and "numpy" in env
    assert len(hist["train_loss"]) == 1

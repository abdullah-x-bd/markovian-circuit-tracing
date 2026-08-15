#!/usr/bin/env bash
set -euo pipefail

cd /workspace/mct
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export MCT_TORCH_THREADS="${MCT_TORCH_THREADS:-2}"

git config user.name "MCT RunPod Bot"
git config user.email "mct-runpod-bot@users.noreply.github.com"

sync_results() {
  set +e
  cd /workspace/mct
  if [ -d results/v2_mega ]; then
    python - <<'PY'
import json, os, platform, subprocess, time
from pathlib import Path
p = Path("results/v2_mega/metadata")
p.mkdir(parents=True, exist_ok=True)
gpu = ""
try:
    gpu = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
        text=True,
    ).strip()
except Exception as exc:
    gpu = f"unavailable: {exc}"
meta = {
    "sync_time_unix": time.time(),
    "gpu": gpu,
    "hostname": platform.node(),
    "runtime_budget_seconds": os.getenv("MCT_RUNTIME_BUDGET_SECONDS"),
    "git_ref": os.getenv("GIT_REF"),
}
(p / "runpod_runtime.json").write_text(json.dumps(meta, indent=2))
PY
    git add results/v2_mega
    if ! git diff --cached --quiet; then
      git commit -m "Add RunPod MCT mega-suite evidence [skip ci]" || true
      git remote set-url origin "https://x-access-token:${GH_TOKEN}@github.com/${GITHUB_REPOSITORY}.git"
      git pull --rebase origin "${GIT_REF}" || true
      git push origin "HEAD:${GIT_REF}" || true
    fi
  fi
}
trap sync_results EXIT

python -m pip install --upgrade pip
pip install -e .
python -m compileall -q src tests experiments scripts
pytest -q

python experiments/run_mega_suite.py \
  --config configs/mega_experiment.yaml \
  --output-dir results/v2_mega \
  --cache-dir /workspace/mct-cache \
  --deadline-seconds "${MCT_RUNTIME_BUDGET_SECONDS}"

sync_results
trap - EXIT

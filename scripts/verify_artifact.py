from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

EXPECTED_SEEDS = (7, 17, 29, 43, 71)
EXPECTED_OBS = ("easy", "medium", "hard")
REQUIRED_AGGREGATES = (
    "raw_runs.json", "summary.json", "claims.json",
    "tables/run_metrics.csv", "tables/summary_by_observability.csv",
    "tables/forcing_controls.csv", "tables/forcing_summary.csv",
    "figures/figure_1_main.svg", "figures/figure_2_belief_recovery.svg",
    "figures/figure_3_sae_comparison.svg",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("results/v1"))
    args = p.parse_args()
    errors = []
    for name in REQUIRED_AGGREGATES:
        if not (args.root / name).exists():
            errors.append(f"missing {name}")
    raw_path = args.root / "raw_runs.json"
    if raw_path.exists():
        raw = json.loads(raw_path.read_text())
        runs = raw.get("runs", {})
        expected = {f"{o}_seed{s:02d}" for o in EXPECTED_OBS for s in EXPECTED_SEEDS}
        if set(runs) != expected:
            errors.append(f"raw run grid mismatch: missing={sorted(expected-set(runs))}, extra={sorted(set(runs)-expected)}")
        for name, payload in runs.items():
            if payload.get("metrics", {}).get("artifact_schema_version") != "1.0":
                errors.append(f"bad schema in raw run {name}")
    if errors:
        raise SystemExit("Artifact verification failed:\n" + "\n".join(errors))
    manifest = {}
    for path in sorted(p for p in args.root.rglob("*") if p.is_file() and p.name != "MANIFEST.json" and "runs" not in p.parts):
        manifest[str(path.relative_to(args.root))] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    (args.root / "MANIFEST.json").write_text(json.dumps({
        "artifact_schema_version": "1.0",
        "canonical_run_count": 15,
        "files": manifest,
    }, indent=2))
    print(f"Committed artifact verified: 15 canonical runs represented, {len(manifest)} files.")


if __name__ == "__main__":
    main()

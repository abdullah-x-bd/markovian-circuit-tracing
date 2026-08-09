from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

EXPECTED_SEEDS = (7, 17, 29, 43, 71)
EXPECTED_OBS = ("easy", "medium", "hard")
REQUIRED_AGGREGATES = (
    "raw_runs.json", "raw/easy.json", "raw/medium.json", "raw/hard.json",
    "summary.json", "claims.json",
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

    expected = {f"{o}_seed{s:02d}" for o in EXPECTED_OBS for s in EXPECTED_SEEDS}
    found = set()
    index_path = args.root / "raw_runs.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())
        if index.get("artifact_schema_version") != "1.0":
            errors.append("bad raw evidence index schema")
        for _, rel in index.get("files", {}).items():
            pth = args.root / rel
            if not pth.exists():
                errors.append(f"missing indexed raw evidence {rel}")
                continue
            raw = json.loads(pth.read_text())
            if raw.get("artifact_schema_version") != "1.0":
                errors.append(f"bad schema in {rel}")
            for name, payload in raw.get("runs", {}).items():
                found.add(name)
                if payload.get("metrics", {}).get("artifact_schema_version") != "1.0":
                    errors.append(f"bad metric schema in raw run {name}")
    if found != expected:
        errors.append(f"raw run grid mismatch: missing={sorted(expected-found)}, extra={sorted(found-expected)}")

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

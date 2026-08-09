# Reproducibility

## One-command canonical suite

```bash
python scripts/run_benchmark_suite.py
```

This reads `configs/main_experiment.yaml` and executes the full 3-regime × 5-seed matrix.

## Build the evidence artifact

```bash
python scripts/aggregate_results.py
python scripts/analyze_claims.py
python scripts/make_figures.py
python scripts/verify_artifact.py
```

## Canonical configuration

The configuration file fixes:

* seeds
* observability regimes
* model architecture
* train/validation/analysis sizes
* Bayes-gap stopping rule
* analysis split proportions
* intervention position and sample counts
* token-history baselines
* fixed SAE configuration

## Output schema

Every run writes:

* `metrics.json`
* `config.json`
* `environment.json`
* `training_history.json`
* `forcing_controls.json`
* `true_transition.npy`
* `recovered_transition.npy`
* `true_empirical_transition.npy`
* `sae_recovered_transition.npy` when SAE evaluation is enabled

`metrics.json` has `artifact_schema_version = "1.0"`.

## Committed evidence

`results/v1/raw_runs.json` combines the text-based run evidence for all 15 canonical cells. Aggregate CSVs, figures, claim tests, and representative matrices are also committed.

The generated run directories can be reconstructed from the frozen config and are validated by `scripts/verify_artifact.py`.

## Integrity

The verifier checks the canonical run grid, required files, metadata consistency, schema version, aggregate outputs, and figures. It then writes SHA-256 hashes to `results/v1/MANIFEST.json`.

## CI

The standard CI workflow runs tests and a complete tiny end-to-end smoke experiment. A separate canonical workflow exposes the full 15-cell benchmark as a GitHub Actions matrix for independent cloud reproduction.

# Canonical evidence v1

This directory contains the completed confirmatory evidence package for Markovian Circuit Tracing.

The canonical grid is 3 observability regimes (`easy`, `medium`, `hard`) by 5 fixed seeds (`7`, `17`, `29`, `43`, `71`), for 15 runs.

Key files:

* `raw_runs.json` - compact 15-cell index with extracted per-run metrics
* `summary.json` - aggregate statistics
* `claims.json` - machine-readable claim statuses and paired cross-seed tests
* `tables/run_metrics.csv` - one row per canonical cell
* `tables/summary_by_observability.csv` - mean, SD, and approximate 95% CI
* `tables/forcing_controls.csv` - all forcing-control rows
* `tables/forcing_summary.csv` - forcing summary by regime and control
* `figures/figure_1_main.svg` - canonical five-panel figure
* `figures/figure_2_belief_recovery.svg` - trained vs history/untrained belief recovery
* `figures/figure_3_sae_comparison.svg` - raw vs fixed-SAE transition recovery
* `MANIFEST.json` - SHA-256 integrity manifest

Complete generated per-run directories are reproducible with `python scripts/run_benchmark_suite.py`.

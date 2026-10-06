# NPL robustness artifact

This directory contains post-hoc robustness analyses added for the Neural Processing Letters revision.

The canonical confirmatory evidence remains unchanged under results/v1 and version 0.2.0. These additional analyses reproduce the canonical trained models from the frozen configuration, then add transition-specific controls and held-out probabilistic Markov-order NLL.

The canonical robustness analysis adds an identically configured untrained-transformer transition control and a four-token causal-history K-means control. The cyclic robustness family replaces only the latent transition matrix and keeps the three canonical emission matrices. Cyclic-family training permits up to 60 epochs so every run reaches the same 0.02-nat Bayes-gap stopping target.

The fixed seeds are 7, 17, 29, 43, and 71. All final metrics use the untouched evaluation split.

Key files:

* posthoc_robustness_summary.csv - aggregate means and 95 percent confidence-interval half widths
* scripts/run_npl_robustness.py - reproducible cell runner

The manuscript treats these analyses as post-hoc and does not fold them into the frozen v0.2.0 confirmatory claim ledger.

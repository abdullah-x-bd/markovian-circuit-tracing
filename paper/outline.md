# Paper-facing outline

## Working title

**Markovian Circuit Tracing: Ground-Truth Tests of State-Machine Abstractions in Transformers**

## One-line claim

Controlled HMM experiments show that transformer activations can support recoverable and causally relevant discrete transition structure when latent states are sufficiently observable, while the same abstraction degrades toward controls as observability falls.

## Evidence-backed abstract skeleton

Mechanistic interpretability often describes model computation through prompt-local features and circuits, but some behaviors may be better characterized as trajectories through reusable internal states. We introduce Markovian Circuit Tracing (MCT), a controlled benchmark for testing whether transformer activations admit compact probabilistic state abstractions. Small causal transformers are trained on sequences from known Hidden Markov Models, giving exact access to latent dynamics, Bayesian predictive beliefs, and counterfactual emission targets. Across five fixed seeds and three levels of hidden-state observability, all models are trained to near-Bayes predictive performance. In easy and medium regimes, held-out state abstractions recover transition matrices substantially closer to the generating process than shuffled and random controls, correct state-centroid interventions move outputs toward exact state-conditioned targets, and same-state activation swaps perturb predictions less than different-state swaps. These effects shrink sharply under hard observability, where transition recovery is no better than controls and state forcing loses state selectivity. Linear predictive-belief probes remain highly decodable but do not outperform strong short-history and untrained-transformer baselines, and a pre-specified sparse autoencoder does not improve transition recovery. The results support MCT as a ground-truth test for when state-machine circuit abstractions are warranted and, equally, when they fail.

## Sections

1. Motivation: from prompt-local circuits to reusable dynamical abstractions
2. Exact HMM benchmark and temporal semantics
3. Held-out state abstraction and transition recovery
4. Baselines and observability sweep
5. Causal state forcing
6. Same-state versus different-state scrubbing
7. Predictive-belief probes and negative specificity result
8. SAE comparison and negative result
9. Limitations and scope
10. Implications for future natural-language validation

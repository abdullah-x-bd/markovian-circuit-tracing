# Completed experiment protocol

This document replaces the early development plan. The canonical benchmark is complete.

## Question

When does a transformer trained on a partially observable Markov process admit a compact internal state abstraction that recovers known transition dynamics and has causal predictive relevance?

## Pre-specified structure

* Three HMM emission-observability regimes with fixed latent transition law
* Five fixed seeds `[7, 17, 29, 43, 71]`
* Exact Bayes training criterion
* Whole-sequence calibration/selection/evaluation split
* Predictive-belief and sampled-state probes
* Discrete held-out transition recovery
* Token-history, untrained-model, shuffled-state, and random-state controls
* State forcing with exact HMM targets and multiple intervention controls
* Same-state versus different-state causal scrubbing
* Fixed SAE comparison

## Completed result

Transition recovery and state-specific causal evidence are strong under easy observability, remain detectable under medium observability, and degrade sharply under hard observability. The fixed SAE does not improve transition recovery. Belief decodability is not specific to learned activations because explicit history and untrained-transformer baselines are competitive or stronger.

The earlier planned natural-language demonstration is intentionally excluded from this artifact. It remains future work and is not needed for the completed HMM validation benchmark.

# Experiment plan

## Goal

Show that Markovian Circuit Tracing can recover a probabilistic state-transition structure from transformer activations.

The first validation task uses a known Hidden Markov Model. This gives us exact ground truth for latent states, transition dynamics, emission dynamics, and counterfactual next-token predictions under forced states.

## Main hypothesis

A small transformer trained on HMM-generated sequences will represent enough latent state information for next-token prediction. If we extract the right internal state representation, its transition matrix should approximate the real HMM transition matrix.

## Exactness tests

### Known transition matrix

The true transition matrix is fixed before training. The recovered internal transition matrix is compared to it with KL, Frobenius error, stationary distribution error, and spectral error.

### Bayes optimal target

The HMM filtering equations give the exact Bayes optimal next-token distribution at every position.

### Counterfactual state forcing

If an internal activation is forced to a centroid representing state k, the model should predict the exact next-token distribution implied by state k.

This target is

```text
T[k] @ E
```

where T is the transition matrix and E is the emission matrix.

## Phases

### Phase 1

Run the four-state HMM benchmark with residual-stream states.

Expected result

- State information is linearly recoverable.
- Clustered internal states recover part of the true state structure.
- The recovered transition matrix is closer than random baselines.
- Forced-state prediction moves toward the exact HMM counterfactual.

### Phase 2

Add sparse autoencoder features.

Expected result

- SAE features give cleaner states than raw residual vectors.
- Transition recovery improves.
- State forcing becomes more stable.

### Phase 3

Add causal scrubbing.

Same-state swaps should preserve output behavior. Different-state swaps should change output behavior in the direction predicted by the HMM.

### Phase 4

Add a small language-model demonstration.

Candidate tasks

- Bracket completion
- Indirect object identification
- Refusal versus compliance
- Quote or parenthesis closing

## Paper figure

The main figure should show

1. True HMM transition graph
2. Recovered internal transition graph
3. Difference heatmap
4. Forced-state prediction matrix
5. Causal scrubbing same-state and different-state results

## Reviewer-facing claim

Markovian Circuit Tracing is not a claim that all transformer behavior is Markovian. It is a method for testing when an interpretable internal state abstraction has Markovian transition structure, and whether that structure is causally used by the model.

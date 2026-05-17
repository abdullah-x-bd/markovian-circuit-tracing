# TMLR-grade experiment plan

## Aim

The revised paper should not claim exact hidden-state recovery from one toy HMM. The stronger and safer claim is that Markovian Circuit Tracing tests whether transformer activations contain coarse latent transition structure, and whether that structure is closer to known HMM belief or state dynamics than appropriate controls.

## Main claim

Markovian Circuit Tracing is a diagnostic framework for latent transition structure in neural activations. On controlled HMM benchmarks, small transformers trained for next-token prediction learn near-Bayes predictors, encode Bayesian belief information in residual activations, and yield recovered internal state transitions that are closer to true or belief-cluster transition structure than shuffled, random, token-only, and dimensionality-reduction controls.

## Experiments required

### Experiment 1: HMM benchmark suite

Run multiple HMM families rather than one hand-picked HMM.

Families:

- easy_separable: low emission overlap, moderate transitions
- ambiguous_emissions: high emission overlap, belief inference required
- persistent: diagonal-heavy transitions
- high_entropy: near-uniform transitions, transition signal should be weak
- three_state: smaller latent system
- six_state: larger latent system

Report per family and aggregate results over seeds.

Core metrics:

- model validation loss
- Bayes optimal loss
- unigram baseline loss
- excess loss over Bayes
- belief probe MSE and KL
- state probe accuracy
- cluster recovery if K equals true K
- transition rowwise KL
- transition Frobenius error
- Markov NLL gain

### Experiment 2: K sensitivity

Do not rely only on oracle K. Run cluster counts:

```text
K = 2, 3, 4, 5, 6, 8, 10
```

Report:

- transition NLL over recovered states
- next true-state NLL from recovered states
- belief reconstruction error from recovered clusters
- cluster purity and entropy

When K does not equal the true number of latent states, avoid direct matrix comparison to the true transition matrix. Use predictive and belief-reconstruction metrics.

### Experiment 3: Belief-state target

The transformer is solving a partially observed problem. The sufficient statistic is the Bayesian belief vector, not the hidden state label.

Add a belief-cluster upper bound:

1. Compute exact Bayesian belief vectors.
2. Cluster belief vectors using the same K.
3. Estimate belief-cluster transition dynamics.
4. Compare residual MCT transition dynamics to belief-cluster dynamics.

This reframes smoothed transition matrices as coarse belief-state dynamics rather than failed hidden-state recovery.

### Experiment 4: Markov NLL tests

Replace accuracy-only Markov tests with held-out negative log-likelihood.

Fit:

- order 0: P(z_t)
- order 1: P(z_t | z_{t-1})
- order 2: P(z_t | z_{t-1}, z_{t-2})

Report:

- NLL0
- NLL1
- NLL2
- NLL0 minus NLL1
- NLL1 minus NLL2
- bootstrap confidence intervals over sequences

Expected result:

Order 1 should improve strongly over order 0. Order 2 should add little beyond order 1.

### Experiment 5: State-forcing controls

Current state forcing must be expanded.

Controls:

- unpatched model
- correct recovered-state centroid
- wrong recovered-state centroid
- mean activation patch
- random Gaussian activation patch
- shuffled-label centroid
- true-state centroid upper bound

Report KL to exact HMM counterfactual target and improvement over unpatched.

### Experiment 6: Baselines

Add baselines beyond random and shuffled.

Baselines:

- token-only current-token state
- token bigram state
- PCA plus KMeans
- random projection plus KMeans
- exact belief clustering
- true-state oracle

The key comparison is whether residual MCT beats token-only and PCA baselines.

### Experiment 7: Layer sweep

Run MCT at:

- embed
- resid_post_0
- resid_post_1
- ln_final

Report belief probe, transition recovery, and Markov NLL by layer.

This makes the paper mechanistic rather than only statistical.

### Experiment 8: Training dynamics

Optional but strong.

Save checkpoints at epochs:

```text
0, 1, 2, 4, 8
```

Run MCT on each checkpoint. Show whether belief recovery and transition recovery emerge as loss approaches Bayes optimal.

## Figures for the revised paper

### Figure 1

Architecture overview.

### Figure 2

Markovian state-transition conceptual figure.

### Figure 3

Benchmark suite summary across HMM families.

### Figure 4

K sensitivity curves.

### Figure 5

Layer sweep.

### Figure 6

State-forcing controls.

### Figure 7

Belief-cluster target comparison.

## Paper framing

Do not write that MCT exactly recovers latent HMM transitions.

Use:

```text
coarse transition signal
belief-state dynamics
diagnostic for transition-structured internal representations
partial latent dynamics recovery
```

Avoid:

```text
exact recovery
full mechanistic explanation
proof that transformers are Markovian
```

## Minimum acceptance-grade result

The revised paper becomes credible if it shows:

1. Near-Bayes model learning across HMM families.
2. Belief vectors are recoverable from activations.
3. Residual MCT beats token-only, PCA, shuffled, and random baselines on transition metrics.
4. Results are stable across seeds and HMM families.
5. Markov NLL tests support first-order structure.
6. State-forcing controls show state-specific causal effect.
7. The code and workflows reproduce all tables and figures.

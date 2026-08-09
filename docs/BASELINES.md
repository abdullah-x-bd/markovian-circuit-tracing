# Baselines and controls

## Predictive-loss controls

* Exact Bayes-optimal predictor
* Uniform token predictor
* Empirical unigram predictor

The transformer must approach the Bayes oracle before interpretability results are considered canonical.

## Representation controls

### Explicit history

One-hot features encode only observations causally available before the current target. Canonical histories use the previous 1, 2, and 4 tokens.

### Untrained transformer

An identically configured transformer with random initialization is evaluated with the same probe protocol. This controls for information made linearly available by embeddings, positional structure, dimensionality, and architecture without task learning.

### Bayes observable-state ceiling

The argmax of the exact predictive hidden-state belief gives the best classification available from the causal observation history under 0-1 loss.

## Transition controls

* Recovered state sequence with labels shuffled globally
* Random four-state sequence
* Empirical transition matrix of the actual sampled hidden states, which estimates finite-sample error in the ground-truth evaluation set

## Intervention controls

State forcing compares the correct recovered centroid against:

* unpatched model
* wrong recovered centroid
* mean activation
* random activation
* deranged shuffled-label centroid
* true-state centroid oracle

The shuffled-label control is implemented as a derangement, ensuring that it never accidentally uses the correct state centroid.

## Causal-scrubbing control

Same-recovered-state activation swaps are the preservation condition. Different-recovered-state swaps are the causal contrast.

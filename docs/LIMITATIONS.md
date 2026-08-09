# Limitations

## Synthetic benchmark

The canonical evidence uses four-state HMMs with six visible symbols and small two-layer transformers. Exact ground truth is the central advantage of the benchmark, but it also limits direct ecological validity for natural-language systems.

## Partial observability and identifiability

The sampled latent state is not always inferable from visible history. The theoretically available object is the predictive belief `P(s_t | x_<t)`. A transformer's internal representation need not choose the same discrete coordinates as the data-generating HMM, and multiple internal abstractions may be behaviorally equivalent.

## Discrete clustering is an imposed abstraction

KMeans requests exactly four discrete clusters because the synthetic generator has four latent states. This is a diagnostic abstraction, not proof that the neural network itself implements four literal symbolic states. The hard-observability failure is an important demonstration of this limitation.

## Belief probes are not mechanistic proof

Linear predictive-belief probes can recover Bayesian beliefs from trained activations, but they can also recover comparable information from untrained transformer activations and explicit short histories. Probe success therefore establishes accessibility of information, not that the trained model uses a distinct Bayesian-belief variable internally.

## Centroid interventions may be off-distribution

Replacing a residual-stream activation with a state centroid is a strong intervention. Centroids can lie in regions the model rarely visits naturally. The benchmark includes mean, random, wrong-state, shuffled-label, and true-state-oracle controls, but centroid patching remains an approximation to causal state manipulation.

## Causal scrubbing tests one activation site

The canonical experiment intervenes at `resid_post_1` and one fixed sequence position. A complete mechanistic account would examine layer, position, head, and feature dependence. The current result demonstrates state-sensitive behavior at the chosen site, not a unique causal locus.

## SAE scope

Only one fixed SAE configuration is used in the confirmatory artifact. This was deliberate to prevent post-hoc hyperparameter selection. The negative SAE result therefore rejects the claim that this pre-specified sparse representation improves the benchmark, not the broader possibility that another SAE architecture or training procedure could perform better.

## Small number of training seeds

Five predetermined seeds provide paired robustness checks but are not a large-sample population study. Reported t-tests should be read together with raw per-seed effects and effect magnitudes.

## Training criterion

Models stop when the model-validation loss approaches the Bayes oracle within a pre-specified gap. This ensures the underlying prediction task is learned, but different architectures or optimization schedules could learn different internal solutions at similar predictive performance.

## No natural-language extrapolation

The earlier roadmap mentioned bracket completion, indirect-object identification, refusal/compliance, and other language tasks. Those experiments are deliberately excluded from the completed canonical artifact. The present repository makes no empirical claim about frontier models or natural-language circuits.

## Markov abstraction, not Markov transformer

MCT tests whether a chosen internal abstraction has useful approximately Markovian transition structure. It does not claim that the full transformer computation is first-order Markovian, nor that all behavior can be represented by a finite-state machine.

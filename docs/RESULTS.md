# Results

## Canonical evidence

The canonical evidence comprises 15 confirmatory runs: three observability regimes (`easy`, `medium`, `hard`) crossed with five fixed seeds (`7, 17, 29, 43, 71`). All inferential comparisons below use the same five paired seeds within a regime.

## 1. The models solve the HMM prediction task

Training uses the model-validation set only and stops after at least six epochs when validation cross-entropy is within `0.02` nats/token of the exact Bayes-optimal loss, or at 30 epochs. All 15 runs reached the stopping condition.

| Regime | Mean epochs | Mean held-out excess loss over Bayes |
|---|---:|---:|
| Easy | 22.4 | 0.0198 |
| Medium | 16.2 | 0.0198 |
| Hard | 10.4 | 0.0190 |

Interpretability results are therefore measured on models that have learned the predictive task rather than on the underfit pilot used only to diagnose compute requirements.

## 2. Predictive beliefs are decodable, but decodability is not specific to learned activations

A ridge probe trained only on calibration sequences reconstructs the exact Bayesian predictive belief `P(s_t | x_<t)` from held-out trained-transformer activations.

| Regime | Belief KL ↓ | Belief R² ↑ | 4-token history KL ↓ | Untrained-transformer KL ↓ |
|---|---:|---:|---:|---:|
| Easy | 0.0291 | 0.925 | 0.0272 | 0.0319 |
| Medium | 0.0246 | 0.849 | 0.0114 | 0.0247 |
| Hard | 0.00294 | 0.777 | 0.000401 | 0.00293 |

The trained representation is not reliably better than the untrained-transformer control. The mean KL advantage of trained over untrained activations is `0.00285` in easy (`p=0.079`), `0.00006` in medium (`p=0.863`), and `-0.000009` in hard (`p=0.886`).

A four-token explicit history baseline is also at least as strong in easy and substantially stronger in medium and hard. Consequently, **belief decodability is treated as a descriptive diagnostic, not evidence of a uniquely learned belief representation**.

## 3. Transition recovery depends strongly on observability

The discrete state abstraction is fit and aligned only on calibration sequences, then frozen and evaluated on untouched sequences.

| Regime | Recovered transition KL ↓ | Shuffled KL ↓ | Random KL ↓ |
|---|---:|---:|---:|
| Easy | 0.114 | 0.379 | 0.359 |
| Medium | 0.221 | 0.355 | 0.359 |
| Hard | 0.360 | 0.367 | 0.359 |

Against the shuffled control, the paired cross-seed advantage is:

* Easy: `0.264`, `p=6.33e-7`
* Medium: `0.133`, `p=3.41e-4`
* Hard: `0.00698`, `p=0.183`

Against the random-state control, recovery is also clearly better in easy and medium, but not in hard.

The result supports recoverable transition structure under easy and medium observability and rejects a broad claim that the same discrete abstraction remains informative under severe emission overlap.

## 4. Markov order tests show the same degradation

Mean gain of an order-1 predictor over an order-0 majority baseline:

* Easy: `+0.134`
* Medium: `+0.043`
* Hard: `-0.0006`

The additional gain from order 2 over order 1 is small in all regimes (`+0.011`, `+0.002`, `+0.004`).

The recovered state sequence therefore looks most like a useful first-order abstraction in the easy regime, less so in medium, and not meaningfully so in hard.

## 5. Causal scrubbing supports state-sensitive internal structure

For each held-out recipient sequence, MCT swaps the activation at the intervention position with a donor from either the same recovered state or a different recovered state. It then measures KL divergence between the original and patched model output distributions.

| Regime | Different-state minus same-state KL ↑ | Paired p-value |
|---|---:|---:|
| Easy | 0.2269 | 0.00131 |
| Medium | 0.0722 | 0.00225 |
| Hard | 0.00203 | 0.00117 |

The sign is consistent across the five seeds in each regime. The hard-regime result is statistically consistent but extremely small in magnitude, so it is reported as a **small-effect** result rather than evidence for a strong discrete circuit.

## 6. State forcing is selective in easy and medium, not hard

The exact same-position target for forcing hidden state `k` is `P(x_t | s_t=k) = E[k]`. Lower KL means the post-intervention prediction is closer to that exact target.

| Regime | Unpatched | Correct recovered centroid | Wrong recovered centroid |
|---|---:|---:|---:|
| Easy | 0.785 | 0.323 | 0.938 |
| Medium | 0.385 | 0.188 | 0.394 |
| Hard | 0.0393 | 0.0223 | 0.0228 |

Correct recovered forcing improves over the unpatched model in all regimes. However, selectivity relative to the wrong recovered centroid is significant only in easy (`p=4.97e-9`) and medium (`p=9.19e-4`). In hard, correct versus wrong forcing is not distinguishable (`p=0.568`).

Thus state-specific causal forcing is supported for easy and medium but not hard.

## 7. The fixed SAE does not improve the abstraction

A fixed SAE configuration was selected before confirmatory evidence: hidden width 128, two epochs, L1 coefficient `0.003`, top-k 16. It was fit only on calibration activations.

| Regime | Raw residual abstraction | SAE abstraction |
|---|---:|---:|
| Easy | 0.114 | 0.117 |
| Medium | 0.221 | 0.258 |
| Hard | 0.360 | 0.416 |

The SAE difference is near zero in easy (`p=0.741`), significantly worse in medium (`p=0.0337`), and worse but marginal under the five-seed test in hard (`p=0.0510`).

The pre-specified hypothesis that sparse features would cleanly improve transition recovery is therefore **not supported**.

## 8. Overall conclusion

The completed benchmark supports a constrained claim:

> When the hidden state of a controlled sequence process is sufficiently observable, transformer activations admit a compact discrete abstraction whose held-out transitions approximate the generating HMM and whose state assignments have causal predictive relevance. As observability decreases, transition recovery and state-specific interventions degrade toward control performance.

The experiment does **not** establish that transformers generally implement discrete Markov state machines, nor that linear belief decodability identifies a learned internal Bayesian belief representation.

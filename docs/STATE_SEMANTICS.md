# State semantics and temporal alignment

Markovian Circuit Tracing uses the convention

1. hidden state `s_t` exists at time `t`;
2. observation `x_t` is emitted from `E[s_t]`;
3. the process transitions to `s_{t+1}` according to `T[s_t]`.

The causal language model input at position `t` contains only `x_<t` and the target is `x_t`. Therefore the primary latent quantity aligned with the transformer activation at position `t` is the predictive belief

`b^-_t = P(s_t | x_<t)`.

The filtering posterior

`b^+_t = P(s_t | x_<=t)`

is a different quantity and is not used as the primary activation target at position `t`.

For a pure intervention that represents `s_t = k`, the exact target distribution for the model output at the same position is `E[k] = P(x_t | s_t=k)`. The distribution `T[k] @ E` is instead the next-step distribution `P(x_{t+1} | s_t=k)` and must not be used as the same-position intervention target.

This convention is used throughout probes, state abstraction, transition analysis, and causal interventions.

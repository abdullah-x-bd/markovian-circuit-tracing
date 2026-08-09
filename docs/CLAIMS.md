# Claim ledger

Every public claim is tied to a committed metric and an explicit status. Machine-readable statistics are in `results/v1/claims.json`.

| Claim | Status | Primary evidence |
|---|---|---|
| Canonical transformers learn the HMM prediction task near Bayes | **Supported** | All 15 validation stopping targets reached; mean evaluation excess loss about 0.019 nats/token |
| Predictive beliefs are linearly decodable from trained activations | **Supported descriptively** | Held-out belief-probe R² = 0.925 / 0.849 / 0.777 |
| Belief decodability is specific to learned transformer representations | **Not supported** | Untrained activations are comparable; 4-token history beats trained activations in medium/hard |
| Recovered discrete states approximate the true transition law in easy HMMs | **Supported** | Transition KL 0.114 vs shuffled 0.379, paired p=6.33e-7 |
| Recovered discrete states approximate the true transition law in medium HMMs | **Supported** | Transition KL 0.221 vs shuffled 0.355, paired p=3.41e-4 |
| Recovered discrete states approximate the true transition law in hard HMMs | **Not supported** | Transition KL 0.360 vs shuffled 0.367, p=0.183; no advantage over random |
| Same-state swaps preserve behavior better than different-state swaps in easy | **Supported** | Scrubbing gap 0.227, p=0.00131 |
| Same-state swaps preserve behavior better than different-state swaps in medium | **Supported** | Scrubbing gap 0.072, p=0.00225 |
| Same-state swaps preserve behavior better than different-state swaps in hard | **Supported, small effect** | Gap 0.0020, p=0.00117 |
| Correct recovered-state forcing is state-selective in easy | **Supported** | Better than unpatched and wrong-centroid controls |
| Correct recovered-state forcing is state-selective in medium | **Supported** | Better than unpatched and wrong-centroid controls |
| Correct recovered-state forcing is state-selective in hard | **Not supported** | Correct vs wrong centroid p=0.568 |
| SAE features improve transition recovery | **Not supported** | Neutral in easy, worse in medium/hard |
| MCT performance depends on latent-state observability | **Supported** | Transition KL rises 0.114 → 0.221 → 0.360 while scrubbing gap falls 0.227 → 0.072 → 0.002 |
| All transformer behavior is Markovian | **Not claimed** | Outside scope by design |
| Results generalize to frontier language models | **Not claimed** | No natural-language/frontier-model study in canonical artifact |

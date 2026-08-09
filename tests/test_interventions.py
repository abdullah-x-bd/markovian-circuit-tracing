import numpy as np
import torch

from mct.interventions import causal_scrubbing_report
from mct.model import TinyTransformer, TransformerConfig


def test_causal_scrubbing_report_runs_with_batch_specific_donors():
    torch.manual_seed(0)
    cfg = TransformerConfig(vocab_size=5, seq_len=5, d_model=8, n_layers=1, n_heads=2, d_mlp=16)
    model = TinyTransformer(cfg)
    x = torch.randint(0, 5, (12, 5))
    with torch.no_grad():
        _, acts = model(x, return_activations=True)
    a = acts["resid_post_0"].numpy()
    states = np.tile(np.array([[0, 1, 0, 1, 0]]), (12, 1))
    states[6:, 2] = 1
    report = causal_scrubbing_report(
        model,
        x,
        a,
        states,
        "resid_post_0",
        position=2,
        seed=1,
        max_pairs=8,
    )
    assert report["scrubbing_pairs"] > 0
    assert report["same_state_swap_kl_mean"] >= 0
    assert report["different_state_swap_kl_mean"] >= 0

import numpy as np
import torch

from mct.interventions import state_forcing_control_report
from mct.model import TinyTransformer, TransformerConfig


def test_shuffled_label_control_is_always_wrong_state():
    torch.manual_seed(0)
    cfg = TransformerConfig(vocab_size=5, seq_len=4, d_model=8, n_layers=1, n_heads=2, d_mlp=16)
    model = TinyTransformer(cfg)
    x = torch.randint(0, 5, (8, 4))
    recovered = np.eye(4, 8, dtype=np.float32)
    oracle = recovered.copy()
    targets = np.full((4, 5), 0.2, dtype=np.float32)
    rows = state_forcing_control_report(model, x, recovered, oracle, targets, "resid_post_0", 2, seed=0, batch_size=8)
    assert sum(r["patch_type"] == "shuffled_label_centroid" for r in rows) == 4

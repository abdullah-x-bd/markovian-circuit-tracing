from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from mct.data import (
    bayes_filter,
    default_hmm,
    forced_state_next_token_distribution,
    make_lm_tensors,
    sample_hmm_sequences,
)
from mct.interventions import state_forcing_kl
from mct.model import TinyTransformer, TransformerConfig, count_parameters
from mct.plots import save_loss_curve, save_matrix_heatmap
from mct.states import best_label_match, cluster_internal_states, probe_state_recovery, state_centroids
from mct.train import collect_activations, evaluate_loss, train_model
from mct.transition import estimate_transition_matrix, markov_order_accuracy, transition_report


def main() -> None:
    out_dir = Path("runs/hmm_4state_demo")
    out_dir.mkdir(parents=True, exist_ok=True)

    seed = 7
    torch.manual_seed(seed)
    np.random.seed(seed)

    hmm = default_hmm()
    bos_token = hmm.vocab_size
    seq_len = 64

    train_tokens, train_states = sample_hmm_sequences(hmm, 8000, seq_len, seed=seed)
    val_tokens, val_states = sample_hmm_sequences(hmm, 2000, seq_len, seed=seed + 1)

    train_x, train_y = make_lm_tensors(train_tokens, bos_token=bos_token)
    val_x, val_y = make_lm_tensors(val_tokens, bos_token=bos_token)

    cfg = TransformerConfig(vocab_size=hmm.vocab_size + 1, seq_len=seq_len)
    model = TinyTransformer(cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    result = train_model(
        model,
        train_x,
        train_y,
        val_x,
        val_y,
        epochs=8,
        batch_size=256,
        lr=3e-4,
    )

    activation_name = "resid_post_1"
    acts = collect_activations(model, val_x, activation_name=activation_name).numpy()
    probe_acc = probe_state_recovery(acts, val_states, seed=seed)

    discovered = cluster_internal_states(acts, n_states=hmm.n_states, seed=seed)
    remapped, cluster_acc = best_label_match(discovered, val_states, n_states=hmm.n_states)
    estimated_t = estimate_transition_matrix(remapped, n_states=hmm.n_states)
    transition_metrics = transition_report(hmm.transition, estimated_t)
    markov_metrics = markov_order_accuracy(remapped, n_states=hmm.n_states)

    centroids = state_centroids(acts, remapped, n_states=hmm.n_states)
    ideal_forced = np.stack(
        [forced_state_next_token_distribution(hmm, s) for s in range(hmm.n_states)],
        axis=0,
    )
    forced_kl = state_forcing_kl(
        model,
        val_x[:512],
        centroids,
        ideal_forced,
        activation_name=activation_name,
        position=20,
        batch_size=128,
    )

    metrics = {
        "parameter_count": count_parameters(model),
        "final_train_loss": result.train_loss[-1],
        "final_val_loss": evaluate_loss(model, val_x, val_y),
        "probe_state_recovery_accuracy": probe_acc,
        "cluster_state_recovery_accuracy": cluster_acc,
        **transition_metrics,
        **markov_metrics,
        **forced_kl,
    }

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    np.save(out_dir / "true_transition.npy", hmm.transition)
    np.save(out_dir / "estimated_transition.npy", estimated_t)
    save_matrix_heatmap(hmm.transition, "True HMM transition", out_dir / "true_transition.png")
    save_matrix_heatmap(estimated_t, "Recovered internal transition", out_dir / "estimated_transition.png")
    save_matrix_heatmap(hmm.transition - estimated_t, "Transition difference", out_dir / "transition_difference.png")
    save_loss_curve(result.train_loss, result.val_loss, out_dir / "loss_curve.png")

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

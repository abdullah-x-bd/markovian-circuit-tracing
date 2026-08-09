from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np
import torch
import sklearn

from mct.data import (
    bayes_predictive_distribution,
    bayes_predictive_state_beliefs,
    make_hmm,
    make_lm_tensors,
    sample_hmm_sequences,
    sequence_cross_entropy,
    unigram_distribution,
)
from mct.history_baselines import history_baseline_report
from mct.interventions import causal_scrubbing_report, state_forcing_control_report
from mct.model import TinyTransformer, TransformerConfig, count_parameters
from mct.probes import bayes_state_classification_ceiling, belief_probe_metrics, state_probe_metrics
from mct.sae_eval import run_sae_evaluation
from mct.splits import split_sequence_indices
from mct.states import fit_state_abstraction, state_centroids, state_recovery_accuracy
from mct.train import collect_activations, evaluate_loss, train_model
from mct.transition import estimate_transition_matrix, markov_order_report, transition_report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the Markovian Circuit Tracing HMM benchmark")
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--run-kind", choices=("exploratory", "confirmatory"), default="exploratory")
    p.add_argument("--hmm-observability", choices=("easy", "medium", "hard"), default="medium")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--train-sequences", type=int, default=8000)
    p.add_argument("--model-val-sequences", type=int, default=1000)
    p.add_argument("--analysis-sequences", type=int, default=3000)
    p.add_argument("--val-sequences", type=int, default=None, help="Deprecated alias for --analysis-sequences")
    p.add_argument("--epochs", type=int, default=30, help="Maximum training epochs")
    p.add_argument("--min-epochs", type=int, default=6)
    p.add_argument("--bayes-gap-target", type=float, default=0.02)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--d-mlp", type=int, default=256)
    p.add_argument("--activation-name", type=str, default="resid_post_1")
    p.add_argument("--forcing-position", type=int, default=20)
    p.add_argument("--forcing-samples", type=int, default=512)
    p.add_argument("--scrubbing-pairs", type=int, default=256)
    p.add_argument("--num-threads", type=int, default=2)
    p.add_argument("--run-sae", action="store_true")
    p.add_argument("--run-sae-sweep", action="store_true")
    p.add_argument("--sae-hidden-dim", type=int, default=256)
    p.add_argument("--sae-epochs", type=int, default=5)
    p.add_argument("--sae-l1-coef", type=float, default=1e-3)
    p.add_argument("--sae-top-k", type=int, default=0)
    p.add_argument("--sae-max-samples", type=int, default=50000)
    p.add_argument("--sae-sweep-hidden-dims", type=str, default="128,256,512")
    p.add_argument("--sae-sweep-l1-coefs", type=str, default="0.003,0.01")
    p.add_argument("--sae-sweep-top-ks", type=str, default="0,8,16,32")
    return p.parse_args()


def pad_visible(values: np.ndarray, bos_token: int) -> np.ndarray:
    padded = np.zeros((*values.shape[:-1], bos_token + 1), dtype=values.dtype)
    padded[..., :bos_token] = values
    return padded


def repeated_loss(tokens: np.ndarray, distribution: np.ndarray) -> float:
    probs = np.broadcast_to(distribution, (*tokens.shape, distribution.shape[0]))
    return sequence_cross_entropy(tokens, probs)


def transition_baseline(true_t: np.ndarray, states: np.ndarray, seed: int, kind: str) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    if kind == "shuffled":
        candidate = states.copy().reshape(-1)
        rng.shuffle(candidate)
        candidate = candidate.reshape(states.shape)
    elif kind == "random":
        candidate = rng.integers(0, true_t.shape[0], size=states.shape)
    else:
        raise KeyError(kind)
    estimated = estimate_transition_matrix(candidate, n_states=true_t.shape[0])
    return {f"{kind}_{k}": v for k, v in transition_report(true_t, estimated).items()}


def json_ready(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_ready(v) for v in value]
    return value


def main() -> None:
    args = parse_args()
    if args.val_sequences is not None:
        args.analysis_sequences = args.val_sequences
    if args.forcing_position >= args.seq_len:
        raise ValueError("forcing_position must be smaller than seq_len")

    torch.set_num_threads(args.num_threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    hmm = make_hmm(args.hmm_observability)
    bos_token = hmm.vocab_size
    out_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path("runs") / args.run_kind / f"{args.hmm_observability}_seed{args.seed:02d}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    train_tokens, _ = sample_hmm_sequences(hmm, args.train_sequences, args.seq_len, seed=args.seed)
    model_val_tokens, _ = sample_hmm_sequences(hmm, args.model_val_sequences, args.seq_len, seed=args.seed + 1)
    analysis_tokens, analysis_states = sample_hmm_sequences(hmm, args.analysis_sequences, args.seq_len, seed=args.seed + 2)
    train_x, train_y = make_lm_tensors(train_tokens, bos_token)
    model_val_x, model_val_y = make_lm_tensors(model_val_tokens, bos_token)
    analysis_x, _ = make_lm_tensors(analysis_tokens, bos_token)

    cfg = TransformerConfig(
        vocab_size=hmm.vocab_size + 1,
        seq_len=args.seq_len,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_mlp=args.d_mlp,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TinyTransformer(cfg).to(device)
    model_val_bayes_loss = sequence_cross_entropy(
        model_val_tokens, bayes_predictive_distribution(hmm, model_val_tokens)
    )
    target_val_loss = model_val_bayes_loss + args.bayes_gap_target
    training = train_model(
        model,
        train_x,
        train_y,
        model_val_x,
        model_val_y,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        min_epochs=args.min_epochs,
        target_val_loss=target_val_loss,
    )

    split = split_sequence_indices(args.analysis_sequences, seed=args.seed + 3)
    cal, sel, ev = split.calibration, split.selection, split.evaluation
    acts = collect_activations(model, analysis_x, args.activation_name, args.batch_size).numpy()
    beliefs = bayes_predictive_state_beliefs(hmm, analysis_tokens)
    cal_acts, sel_acts, ev_acts = acts[cal], acts[sel], acts[ev]
    cal_tokens, ev_tokens = analysis_tokens[cal], analysis_tokens[ev]
    cal_states, sel_states, ev_states = analysis_states[cal], analysis_states[sel], analysis_states[ev]
    cal_beliefs, ev_beliefs = beliefs[cal], beliefs[ev]
    ev_x = analysis_x[ev]

    belief_metrics = belief_probe_metrics(cal_acts, cal_beliefs, ev_acts, ev_beliefs, ev_states)
    state_metrics = state_probe_metrics(cal_acts, cal_states, ev_acts, ev_states)
    abstraction = fit_state_abstraction(cal_acts, cal_states, n_states=hmm.n_states, seed=args.seed)
    cal_recovered = abstraction.predict(cal_acts)
    ev_recovered = abstraction.predict(ev_acts)
    recovered_t = estimate_transition_matrix(ev_recovered, n_states=hmm.n_states)
    true_empirical_t = estimate_transition_matrix(ev_states, n_states=hmm.n_states)

    history_metrics = history_baseline_report(
        cal_tokens,
        ev_tokens,
        cal_beliefs,
        ev_beliefs,
        cal_states,
        ev_states,
        vocab_size=hmm.vocab_size,
    )
    torch.manual_seed(args.seed + 999)
    untrained = TinyTransformer(cfg).to(device)
    untrained_cal = collect_activations(untrained, analysis_x[cal], args.activation_name, args.batch_size).numpy()
    untrained_ev = collect_activations(untrained, ev_x, args.activation_name, args.batch_size).numpy()
    untrained_metrics = {
        **belief_probe_metrics(untrained_cal, cal_beliefs, untrained_ev, ev_beliefs, ev_states),
        **state_probe_metrics(untrained_cal, cal_states, untrained_ev, ev_states),
    }

    recovered_centroids = state_centroids(cal_acts, cal_recovered, hmm.n_states)
    true_centroids = state_centroids(cal_acts, cal_states, hmm.n_states)
    forcing_n = min(args.forcing_samples, len(ev_x))
    forcing_controls = state_forcing_control_report(
        model,
        ev_x[:forcing_n],
        recovered_centroids,
        true_centroids,
        pad_visible(hmm.emission, bos_token),
        args.activation_name,
        args.forcing_position,
        seed=args.seed + 200,
        batch_size=min(args.batch_size, 128),
    )
    scrubbing = causal_scrubbing_report(
        model,
        ev_x,
        ev_acts,
        ev_recovered,
        args.activation_name,
        args.forcing_position,
        seed=args.seed + 300,
        max_pairs=args.scrubbing_pairs,
    )

    sae_metrics: dict[str, object] = {}
    if args.run_sae or args.run_sae_sweep:
        sae_metrics, sae_t = run_sae_evaluation(
            calibration_activations=cal_acts,
            selection_activations=sel_acts,
            evaluation_activations=ev_acts,
            calibration_states=cal_states,
            selection_states=sel_states,
            evaluation_states=ev_states,
            calibration_beliefs=cal_beliefs,
            evaluation_beliefs=ev_beliefs,
            true_transition=hmm.transition,
            seed=args.seed,
            epochs=args.sae_epochs,
            max_samples=args.sae_max_samples,
            batch_size=args.batch_size,
            run_sweep=args.run_sae_sweep,
            hidden_dim=args.sae_hidden_dim,
            l1_coef=args.sae_l1_coef,
            top_k=args.sae_top_k,
            sweep_hidden_dims=args.sae_sweep_hidden_dims,
            sweep_l1_coefs=args.sae_sweep_l1_coefs,
            sweep_top_ks=args.sae_sweep_top_ks,
        )
        np.save(out_dir / "sae_recovered_transition.npy", sae_t)

    bayes_loss = sequence_cross_entropy(ev_tokens, bayes_predictive_distribution(hmm, ev_tokens))
    eval_loss = evaluate_loss(model, ev_x, torch.from_numpy(ev_tokens), batch_size=args.batch_size)
    unigram = unigram_distribution(train_tokens, hmm.vocab_size)
    true_empirical_report = transition_report(hmm.transition, true_empirical_t)
    metrics = {
        "artifact_schema_version": "1.0",
        "run_metadata": {
            "run_kind": args.run_kind,
            "hmm_observability": args.hmm_observability,
            "seed": args.seed,
            "device": device,
            "parameter_count": count_parameters(model),
            "activation_name": args.activation_name,
            "sequence_counts": {
                "train": args.train_sequences,
                "model_validation": args.model_val_sequences,
                "analysis_total": args.analysis_sequences,
                "calibration": int(len(cal)),
                "selection": int(len(sel)),
                "evaluation": int(len(ev)),
            },
        },
        "model_quality": {
            "final_train_loss": training.train_loss[-1],
            "final_model_validation_loss": training.val_loss[-1],
            "bayes_optimal_model_validation_loss": model_val_bayes_loss,
            "model_validation_excess_over_bayes": training.val_loss[-1] - model_val_bayes_loss,
            "actual_training_epochs": len(training.train_loss),
            "maximum_training_epochs": args.epochs,
            "minimum_training_epochs": args.min_epochs,
            "bayes_gap_target": args.bayes_gap_target,
            "stopping_target_reached": bool(training.val_loss[-1] <= target_val_loss),
            "untouched_evaluation_loss": eval_loss,
            "bayes_optimal_evaluation_loss": bayes_loss,
            "model_excess_loss_over_bayes": eval_loss - bayes_loss,
            "uniform_baseline_loss": float(np.log(hmm.vocab_size)),
            "unigram_baseline_loss": repeated_loss(ev_tokens, unigram),
        },
        "belief_recovery": belief_metrics,
        "sampled_state_recovery": {
            **state_metrics,
            "cluster_state_recovery_accuracy": state_recovery_accuracy(abstraction, ev_acts, ev_states),
            "bayes_observable_state_accuracy_ceiling": bayes_state_classification_ceiling(ev_beliefs, ev_states),
        },
        "transition_recovery": {
            **transition_report(hmm.transition, recovered_t),
            **{f"true_empirical_{k}": v for k, v in true_empirical_report.items()},
            **transition_baseline(hmm.transition, ev_recovered, args.seed + 101, "shuffled"),
            **transition_baseline(hmm.transition, ev_recovered, args.seed + 202, "random"),
        },
        "markov_tests": markov_order_report(cal_recovered, ev_recovered, hmm.n_states),
        "token_history_baselines": history_metrics,
        "untrained_transformer_baseline": untrained_metrics,
        "causal_scrubbing": scrubbing,
        "sae": sae_metrics,
        "intervention_summary": {
            "forcing_position": args.forcing_position,
            "forcing_sequences": forcing_n,
            "target_semantics": "P(x_t | s_t=k) = E[k]",
        },
    }
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2))
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
    }
    (out_dir / "environment.json").write_text(json.dumps(environment, indent=2))
    (out_dir / "training_history.json").write_text(json.dumps({"train_loss": training.train_loss, "model_validation_loss": training.val_loss}, indent=2))
    (out_dir / "metrics.json").write_text(json.dumps(json_ready(metrics), indent=2))
    (out_dir / "forcing_controls.json").write_text(json.dumps(json_ready(forcing_controls), indent=2))
    np.save(out_dir / "true_transition.npy", hmm.transition)
    np.save(out_dir / "recovered_transition.npy", recovered_t)
    np.save(out_dir / "true_empirical_transition.npy", true_empirical_t)
    print(json.dumps(json_ready(metrics), indent=2))


if __name__ == "__main__":
    main()

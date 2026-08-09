from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from mct.baselines import history_baseline_report
from mct.data import (
    bayes_predictive_distribution,
    bayes_predictive_state_beliefs,
    make_hmm,
    make_lm_tensors,
    sample_hmm_sequences,
    sequence_cross_entropy,
    unigram_distribution,
)
from mct.interventions import causal_scrubbing_report, state_forcing_control_report
from mct.model import TinyTransformer, TransformerConfig, count_parameters
from mct.probes import bayes_state_classification_ceiling, belief_probe_metrics, state_probe_metrics
from mct.sae import encode_activations, train_sae
from mct.splits import split_sequence_indices
from mct.states import fit_state_abstraction, state_centroids, state_recovery_accuracy
from mct.train import collect_activations, evaluate_loss, train_model
from mct.transition import estimate_transition_matrix, markov_order_report, transition_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Markovian Circuit Tracing HMM benchmark")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--run-kind", choices=("exploratory", "confirmatory"), default="exploratory")
    parser.add_argument("--hmm-observability", choices=("easy", "medium", "hard"), default="medium")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--train-sequences", type=int, default=8000)
    parser.add_argument("--model-val-sequences", type=int, default=1000)
    parser.add_argument("--analysis-sequences", type=int, default=3000)
    parser.add_argument("--val-sequences", type=int, default=None, help="Deprecated alias for --analysis-sequences")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--d-mlp", type=int, default=256)
    parser.add_argument("--activation-name", type=str, default="resid_post_1")
    parser.add_argument("--forcing-position", type=int, default=20)
    parser.add_argument("--forcing-samples", type=int, default=512)
    parser.add_argument("--scrubbing-pairs", type=int, default=256)
    parser.add_argument("--num-threads", type=int, default=2)
    parser.add_argument("--run-sae", action="store_true")
    parser.add_argument("--run-sae-sweep", action="store_true")
    parser.add_argument("--sae-hidden-dim", type=int, default=256)
    parser.add_argument("--sae-epochs", type=int, default=5)
    parser.add_argument("--sae-l1-coef", type=float, default=1e-3)
    parser.add_argument("--sae-top-k", type=int, default=0)
    parser.add_argument("--sae-max-samples", type=int, default=50000)
    parser.add_argument("--sae-sweep-hidden-dims", type=str, default="128,256,512")
    parser.add_argument("--sae-sweep-l1-coefs", type=str, default="0.003,0.01")
    parser.add_argument("--sae-sweep-top-ks", type=str, default="0,8,16,32")
    return parser.parse_args()


def _pad_visible_distribution(values: np.ndarray, bos_token: int) -> np.ndarray:
    padded = np.zeros((*values.shape[:-1], bos_token + 1), dtype=values.dtype)
    padded[..., :bos_token] = values
    return padded


def _repeated_distribution_loss(tokens: np.ndarray, distribution: np.ndarray) -> float:
    probs = np.broadcast_to(distribution, (*tokens.shape, distribution.shape[0]))
    return sequence_cross_entropy(tokens, probs)


def _shuffled_transition_baseline(true_transition: np.ndarray, states: np.ndarray, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    shuffled = states.copy().reshape(-1)
    rng.shuffle(shuffled)
    shuffled = shuffled.reshape(states.shape)
    estimated = estimate_transition_matrix(shuffled, n_states=true_transition.shape[0])
    return {f"shuffled_{k}": v for k, v in transition_report(true_transition, estimated).items()}


def _random_transition_baseline(true_transition: np.ndarray, states: np.ndarray, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    random_states = rng.integers(0, true_transition.shape[0], size=states.shape)
    estimated = estimate_transition_matrix(random_states, n_states=true_transition.shape[0])
    return {f"random_{k}": v for k, v in transition_report(true_transition, estimated).items()}


def _json_ready(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: _json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_ready(v) for v in value]
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

    if args.output_dir is None:
        out_dir = Path("runs") / args.run_kind / f"{args.hmm_observability}_seed{args.seed:02d}"
    else:
        out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_tokens, _ = sample_hmm_sequences(hmm, args.train_sequences, args.seq_len, seed=args.seed)
    model_val_tokens, _ = sample_hmm_sequences(hmm, args.model_val_sequences, args.seq_len, seed=args.seed + 1)
    analysis_tokens, analysis_states = sample_hmm_sequences(
        hmm, args.analysis_sequences, args.seq_len, seed=args.seed + 2
    )

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
    model = TinyTransformer(cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    training = train_model(
        model,
        train_x,
        train_y,
        model_val_x,
        model_val_y,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )

    split = split_sequence_indices(args.analysis_sequences, seed=args.seed + 3)
    cal_idx, sel_idx, eval_idx = split.calibration, split.selection, split.evaluation

    analysis_acts = collect_activations(
        model, analysis_x, activation_name=args.activation_name, batch_size=args.batch_size
    ).numpy()
    predictive_beliefs = bayes_predictive_state_beliefs(hmm, analysis_tokens)

    cal_acts, sel_acts, eval_acts = analysis_acts[cal_idx], analysis_acts[sel_idx], analysis_acts[eval_idx]
    cal_tokens, sel_tokens, eval_tokens = analysis_tokens[cal_idx], analysis_tokens[sel_idx], analysis_tokens[eval_idx]
    cal_states, sel_states, eval_states = analysis_states[cal_idx], analysis_states[sel_idx], analysis_states[eval_idx]
    cal_beliefs, sel_beliefs, eval_beliefs = predictive_beliefs[cal_idx], predictive_beliefs[sel_idx], predictive_beliefs[eval_idx]
    eval_x = analysis_x[eval_idx]

    belief_metrics = belief_probe_metrics(cal_acts, cal_beliefs, eval_acts, eval_beliefs, eval_states)
    state_probe = state_probe_metrics(cal_acts, cal_states, eval_acts, eval_states)
    bayes_ceiling = bayes_state_classification_ceiling(eval_beliefs, eval_states)

    abstraction = fit_state_abstraction(cal_acts, cal_states, n_states=hmm.n_states, seed=args.seed)
    cal_recovered = abstraction.predict(cal_acts)
    eval_recovered = abstraction.predict(eval_acts)
    recovered_accuracy = state_recovery_accuracy(abstraction, eval_acts, eval_states)
    recovered_t = estimate_transition_matrix(eval_recovered, n_states=hmm.n_states)

    transition_metrics = transition_report(hmm.transition, recovered_t)
    markov_metrics = markov_order_report(cal_recovered, eval_recovered, n_states=hmm.n_states)
    true_empirical_t = estimate_transition_matrix(eval_states, n_states=hmm.n_states)
    true_empirical_report = transition_report(hmm.transition, true_empirical_t)

    history_metrics = history_baseline_report(
        cal_tokens,
        eval_tokens,
        cal_beliefs,
        eval_beliefs,
        cal_states,
        eval_states,
        vocab_size=hmm.vocab_size,
    )
    torch.manual_seed(args.seed + 999)
    untrained = TinyTransformer(cfg).to(device)
    untrained_cal_acts = collect_activations(
        untrained, analysis_x[cal_idx], activation_name=args.activation_name, batch_size=args.batch_size
    ).numpy()
    untrained_eval_acts = collect_activations(
        untrained, eval_x, activation_name=args.activation_name, batch_size=args.batch_size
    ).numpy()
    untrained_metrics = {
        **belief_probe_metrics(untrained_cal_acts, cal_beliefs, untrained_eval_acts, eval_beliefs, eval_states),
        **state_probe_metrics(untrained_cal_acts, cal_states, untrained_eval_acts, eval_states),
    }

    recovered_centroids = state_centroids(cal_acts, cal_recovered, n_states=hmm.n_states)
    true_centroids = state_centroids(cal_acts, cal_states, n_states=hmm.n_states)
    current_state_targets = _pad_visible_distribution(hmm.emission, bos_token)
    forcing_n = min(args.forcing_samples, len(eval_x))
    forcing_controls = state_forcing_control_report(
        model,
        eval_x[:forcing_n],
        recovered_centroids,
        true_centroids,
        current_state_targets,
        activation_name=args.activation_name,
        position=args.forcing_position,
        seed=args.seed + 200,
        batch_size=min(args.batch_size, 128),
    )
    scrubbing = causal_scrubbing_report(
        model,
        eval_x,
        eval_acts,
        eval_recovered,
        activation_name=args.activation_name,
        position=args.forcing_position,
        seed=args.seed + 300,
        max_pairs=args.scrubbing_pairs,
    )

    sae_metrics: dict[str, object] = {}
    if args.run_sae or args.run_sae_sweep:
        def evaluate_sae_candidate(hidden_dim: int, l1_coef: float, top_k_value: int, use_selection: bool, candidate_seed: int):
            top_k = None if top_k_value <= 0 else top_k_value
            sae, sae_result = train_sae(
                cal_acts,
                hidden_dim=hidden_dim,
                l1_coef=l1_coef,
                epochs=args.sae_epochs,
                batch_size=max(args.batch_size, 256),
                max_samples=args.sae_max_samples,
                seed=candidate_seed,
                top_k=top_k,
            )
            cal_z = encode_activations(sae, cal_acts)
            target_acts = sel_acts if use_selection else eval_acts
            target_states = sel_states if use_selection else eval_states
            target_z = encode_activations(sae, target_acts)
            sae_abs = fit_state_abstraction(cal_z, cal_states, n_states=hmm.n_states, seed=candidate_seed)
            target_recovered = sae_abs.predict(target_z)
            target_t = estimate_transition_matrix(target_recovered, n_states=hmm.n_states)
            rep = transition_report(hmm.transition, target_t)
            return sae, sae_result, cal_z, target_z, target_recovered, target_t, rep

        if args.run_sae_sweep:
            hidden_dims = [int(x) for x in args.sae_sweep_hidden_dims.split(",") if x.strip()]
            l1s = [float(x) for x in args.sae_sweep_l1_coefs.split(",") if x.strip()]
            topks = [int(x) for x in args.sae_sweep_top_ks.split(",") if x.strip()]
            rows = []
            best = None
            for hidden_dim in hidden_dims:
                for l1_coef in l1s:
                    for top_k_value in topks:
                        cand_seed = args.seed + hidden_dim + int(l1_coef * 1_000_000) + top_k_value
                        _, res, _, _, _, _, rep = evaluate_sae_candidate(
                            hidden_dim, l1_coef, top_k_value, True, cand_seed
                        )
                        row = {
                            "hidden_dim": hidden_dim,
                            "l1_coef": l1_coef,
                            "top_k": top_k_value,
                            "reconstruction_mse": res.reconstruction_mse,
                            "active_fraction": res.active_fraction,
                            **rep,
                        }
                        rows.append(row)
                        if best is None or row["rowwise_kl"] < best["rowwise_kl"]:
                            best = row
            assert best is not None
            best_seed = args.seed + int(best["hidden_dim"]) + int(float(best["l1_coef"]) * 1_000_000) + int(best["top_k"])
            _, res, cal_z, eval_z, eval_rec, eval_t, rep = evaluate_sae_candidate(
                int(best["hidden_dim"]), float(best["l1_coef"]), int(best["top_k"]), False, best_seed
            )
            sae_metrics = {
                "selection_rule": "minimum selection-set transition rowwise KL",
                "sweep": rows,
                "best_hyperparameters": best,
                "heldout_evaluation": {
                    "reconstruction_mse": res.reconstruction_mse,
                    "active_fraction": res.active_fraction,
                    **rep,
                    **belief_probe_metrics(cal_z, cal_beliefs, eval_z, eval_beliefs, eval_states),
                    "cluster_state_recovery_accuracy": float((eval_rec == eval_states).mean()),
                },
            }
            np.save(out_dir / "sae_recovered_transition.npy", eval_t)
        elif args.run_sae:
            _, res, cal_z, eval_z, eval_rec, eval_t, rep = evaluate_sae_candidate(
                args.sae_hidden_dim, args.sae_l1_coef, args.sae_top_k, False, args.seed + 500
            )
            sae_metrics = {
                "heldout_evaluation": {
                    "hidden_dim": args.sae_hidden_dim,
                    "l1_coef": args.sae_l1_coef,
                    "top_k": args.sae_top_k,
                    "reconstruction_mse": res.reconstruction_mse,
                    "active_fraction": res.active_fraction,
                    **rep,
                    **belief_probe_metrics(cal_z, cal_beliefs, eval_z, eval_beliefs, eval_states),
                    "cluster_state_recovery_accuracy": float((eval_rec == eval_states).mean()),
                }
            }
            np.save(out_dir / "sae_recovered_transition.npy", eval_t)

    eval_probs = bayes_predictive_distribution(hmm, eval_tokens)
    bayes_loss = sequence_cross_entropy(eval_tokens, eval_probs)
    final_eval_loss = evaluate_loss(model, eval_x, torch.from_numpy(eval_tokens), batch_size=args.batch_size)
    unigram = unigram_distribution(train_tokens, hmm.vocab_size)

    metrics = {
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
                "calibration": int(len(cal_idx)),
                "selection": int(len(sel_idx)),
                "evaluation": int(len(eval_idx)),
            },
        },
        "model_quality": {
            "final_train_loss": training.train_loss[-1],
            "final_model_validation_loss": training.val_loss[-1],
            "untouched_evaluation_loss": final_eval_loss,
            "bayes_optimal_evaluation_loss": bayes_loss,
            "model_excess_loss_over_bayes": final_eval_loss - bayes_loss,
            "uniform_baseline_loss": float(np.log(hmm.vocab_size)),
            "unigram_baseline_loss": _repeated_distribution_loss(eval_tokens, unigram),
        },
        "belief_recovery": belief_metrics,
        "sampled_state_recovery": {
            **state_probe,
            "cluster_state_recovery_accuracy": recovered_accuracy,
            "bayes_observable_state_accuracy_ceiling": bayes_ceiling,
        },
        "transition_recovery": {
            **transition_metrics,
            **{f"true_empirical_{k}": v for k, v in true_empirical_report.items()},
            **_shuffled_transition_baseline(hmm.transition, eval_recovered, args.seed + 101),
            **_random_transition_baseline(hmm.transition, eval_recovered, args.seed + 202),
        },
        "markov_tests": markov_metrics,
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
    (out_dir / "metrics.json").write_text(json.dumps(_json_ready(metrics), indent=2))
    (out_dir / "forcing_controls.json").write_text(json.dumps(_json_ready(forcing_controls), indent=2))
    np.save(out_dir / "true_transition.npy", hmm.transition)
    np.save(out_dir / "recovered_transition.npy", recovered_t)
    np.save(out_dir / "true_empirical_transition.npy", true_empirical_t)
    print(json.dumps(_json_ready(metrics), indent=2))


if __name__ == "__main__":
    main()

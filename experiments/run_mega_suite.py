from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import random
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

from mct.data import (
    HMM,
    bayes_predictive_distribution,
    bayes_predictive_state_beliefs,
    make_hmm,
    make_lm_tensors,
    sample_hmm_sequences,
    sequence_cross_entropy,
)
from mct.interventions import causal_scrubbing_report, kl_divergence_torch
from mct.model import TinyTransformer, TransformerConfig, count_parameters
from mct.probes import belief_probe_metrics, bayes_state_classification_ceiling
from mct.splits import split_sequence_indices
from mct.states import fit_state_abstraction, state_centroids, state_recovery_accuracy
from mct.train import collect_activations, evaluate_loss, train_model
from mct.transition import estimate_transition_matrix, markov_order_report, transition_report


@dataclass(frozen=True)
class Arch:
    name: str
    d_model: int
    n_layers: int
    n_heads: int
    d_mlp: int


@dataclass
class RunBundle:
    label: str
    seed: int
    hmm: HMM
    model: TinyTransformer
    cfg: TransformerConfig
    tokens: np.ndarray
    states: np.ndarray
    x: torch.Tensor
    beliefs: np.ndarray
    split: object
    activations: dict[str, np.ndarray]
    epochs: int
    val_loss: float
    bayes_val_loss: float


class Deadline:
    def __init__(self, seconds: int):
        self.start = time.monotonic()
        self.seconds = int(seconds)

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.start

    @property
    def remaining(self) -> float:
        return self.seconds - self.elapsed

    def can_start(self, reserve: float = 180.0) -> bool:
        return self.remaining > reserve

    def require(self, reserve: float = 180.0) -> None:
        if not self.can_start(reserve):
            raise TimeoutError(f"MCT mega-suite deadline reached with {self.remaining:.1f}s remaining")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Budget-aware MCT robustness mega-suite")
    p.add_argument("--config", type=Path, default=Path("configs/mega_experiment.yaml"))
    p.add_argument("--output-dir", type=Path, default=Path("results/v2_mega"))
    p.add_argument("--cache-dir", type=Path, default=Path("/workspace/mct-cache"))
    p.add_argument("--deadline-seconds", type=int, default=int(os.getenv("MCT_RUNTIME_BUDGET_SECONDS", "14400")))
    p.add_argument("--quick", action="store_true")
    return p.parse_args()


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def ensure_dirs(root: Path) -> None:
    for sub in ["tables", "figures", "raw", "metadata"]:
        (root / sub).mkdir(parents=True, exist_ok=True)


def json_ready(x):
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, dict):
        return {k: json_ready(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [json_ready(v) for v in x]
    return x


def json_scalar(x):
    if isinstance(x, (list, tuple, dict)):
        return json.dumps(json_ready(x), separators=(",", ":"))
    if isinstance(x, np.generic):
        return x.item()
    return x


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(obj), indent=2))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    keys = sorted({k for r in rows for k in r})
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows([{k: json_scalar(r.get(k)) for k in keys} for r in rows])


def arch_from_dict(d: dict) -> Arch:
    return Arch(
        name=str(d["name"]),
        d_model=int(d["d_model"]),
        n_layers=int(d["n_layers"]),
        n_heads=int(d["n_heads"]),
        d_mlp=int(d["d_mlp"]),
    )


def make_interpolated_hmm(alpha: float) -> HMM:
    easy = make_hmm("easy")
    hard = make_hmm("hard")
    emission = (1.0 - alpha) * easy.emission + alpha * hard.emission
    emission = emission / emission.sum(axis=1, keepdims=True)
    return HMM(
        transition=easy.transition.copy(),
        emission=emission,
        initial=easy.initial.copy(),
        name=f"obs_alpha_{alpha:.3f}",
    )


def make_random_hmm(seed: int, n_states: int, vocab_size: int, persistence: float, observability: float) -> HMM:
    rng = np.random.default_rng(seed)
    transition = np.zeros((n_states, n_states), dtype=np.float64)
    for i in range(n_states):
        off = rng.dirichlet(np.ones(n_states))
        off[i] = 0.0
        if off.sum() <= 0:
            off[(i + 1) % n_states] = 1.0
        off /= off.sum()
        transition[i] = (1.0 - persistence) * off
        transition[i, i] += persistence
    anchors = rng.dirichlet(np.full(vocab_size, 0.25), size=n_states)
    uniform = np.full((n_states, vocab_size), 1.0 / vocab_size)
    emission = observability * anchors + (1.0 - observability) * uniform
    emission /= emission.sum(axis=1, keepdims=True)
    initial = np.full(n_states, 1.0 / n_states)
    return HMM(
        transition=transition,
        emission=emission,
        initial=initial,
        name=f"random_k{n_states}_p{persistence:.2f}_o{observability:.2f}_seed{seed}",
    )


def model_config(hmm: HMM, seq_len: int, arch: Arch) -> TransformerConfig:
    return TransformerConfig(
        vocab_size=hmm.vocab_size + 1,
        seq_len=seq_len,
        d_model=arch.d_model,
        n_layers=arch.n_layers,
        n_heads=arch.n_heads,
        d_mlp=arch.d_mlp,
    )


def train_bundle(*, hmm: HMM, label: str, seed: int, arch: Arch, cfg: dict, deadline: Deadline, collect_all_layers: bool = False) -> RunBundle:
    deadline.require(120)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    tcfg = cfg["training"]
    seq_len = int(tcfg["seq_len"])
    n_train = int(tcfg["train_sequences"])
    n_val = int(tcfg["model_val_sequences"])
    n_analysis = int(tcfg["analysis_sequences"])
    batch_size = int(tcfg["batch_size"])
    train_tokens, _ = sample_hmm_sequences(hmm, n_train, seq_len, seed=seed)
    val_tokens, _ = sample_hmm_sequences(hmm, n_val, seq_len, seed=seed + 1)
    tokens, states = sample_hmm_sequences(hmm, n_analysis, seq_len, seed=seed + 2)
    bos = hmm.vocab_size
    train_x, train_y = make_lm_tensors(train_tokens, bos)
    val_x, val_y = make_lm_tensors(val_tokens, bos)
    x, _ = make_lm_tensors(tokens, bos)
    mcfg = model_config(hmm, seq_len, arch)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TinyTransformer(mcfg).to(device)
    bayes_val_loss = sequence_cross_entropy(val_tokens, bayes_predictive_distribution(hmm, val_tokens))
    target = bayes_val_loss + float(tcfg["bayes_gap_target"])
    result = train_model(
        model,
        train_x,
        train_y,
        val_x,
        val_y,
        epochs=int(tcfg["max_epochs"]),
        batch_size=batch_size,
        lr=float(tcfg["learning_rate"]),
        min_epochs=int(tcfg["min_epochs"]),
        target_val_loss=target,
    )
    val_loss = evaluate_loss(model, val_x, val_y, batch_size=batch_size)
    beliefs = bayes_predictive_state_beliefs(hmm, tokens)
    split = split_sequence_indices(n_analysis, seed=seed + 3)
    names = [f"resid_post_{arch.n_layers - 1}"]
    if collect_all_layers:
        names = [f"resid_post_{i}" for i in range(arch.n_layers)]
    activations = {name: collect_activations(model, x, name, batch_size=batch_size).numpy() for name in names}
    return RunBundle(
        label=label,
        seed=seed,
        hmm=hmm,
        model=model,
        cfg=mcfg,
        tokens=tokens,
        states=states,
        x=x,
        beliefs=beliefs,
        split=split,
        activations=activations,
        epochs=len(result.train_loss),
        val_loss=val_loss,
        bayes_val_loss=bayes_val_loss,
    )


def core_metrics(bundle: RunBundle, activation_name: str | None = None) -> dict:
    if activation_name is None:
        activation_name = f"resid_post_{bundle.cfg.n_layers - 1}"
    acts = bundle.activations[activation_name]
    cal, ev = bundle.split.calibration, bundle.split.evaluation
    cal_acts, ev_acts = acts[cal], acts[ev]
    cal_states, ev_states = bundle.states[cal], bundle.states[ev]
    cal_beliefs, ev_beliefs = bundle.beliefs[cal], bundle.beliefs[ev]
    abstraction = fit_state_abstraction(cal_acts, cal_states, bundle.hmm.n_states, seed=bundle.seed)
    cal_rec = abstraction.predict(cal_acts)
    ev_rec = abstraction.predict(ev_acts)
    recovered_t = estimate_transition_matrix(ev_rec, bundle.hmm.n_states)
    tr = transition_report(bundle.hmm.transition, recovered_t)
    belief = belief_probe_metrics(cal_acts, cal_beliefs, ev_acts, ev_beliefs, ev_states)
    mk = markov_order_report(cal_rec, ev_rec, bundle.hmm.n_states)
    rng = np.random.default_rng(bundle.seed + 7001)
    flat = ev_rec.reshape(-1).copy()
    rng.shuffle(flat)
    shuf = flat.reshape(ev_rec.shape)
    shuf_t = estimate_transition_matrix(shuf, bundle.hmm.n_states)
    shuf_report = transition_report(bundle.hmm.transition, shuf_t)
    eval_x = bundle.x[ev]
    scrubbing = causal_scrubbing_report(
        bundle.model,
        eval_x,
        ev_acts,
        ev_rec,
        activation_name,
        position=min(12, bundle.cfg.seq_len - 1),
        seed=bundle.seed + 400,
        max_pairs=min(48, len(eval_x)),
    )
    return {
        "label": bundle.label,
        "seed": bundle.seed,
        "n_states": bundle.hmm.n_states,
        "vocab_size": bundle.hmm.vocab_size,
        "d_model": bundle.cfg.d_model,
        "n_layers": bundle.cfg.n_layers,
        "n_heads": bundle.cfg.n_heads,
        "parameters": count_parameters(bundle.model),
        "epochs": bundle.epochs,
        "validation_excess_over_bayes": bundle.val_loss - bundle.bayes_val_loss,
        "state_recovery_accuracy": state_recovery_accuracy(abstraction, ev_acts, ev_states),
        "bayes_state_accuracy_ceiling": bayes_state_classification_ceiling(ev_beliefs, ev_states),
        "transition_kl": tr["rowwise_kl"],
        "shuffled_transition_kl": shuf_report["rowwise_kl"],
        "transition_advantage_over_shuffle": shuf_report["rowwise_kl"] - tr["rowwise_kl"],
        **belief,
        **mk,
        **scrubbing,
    }


def _target_kl(logits: torch.Tensor, target_visible: np.ndarray, visible_vocab: int) -> torch.Tensor:
    pred = F.softmax(logits[..., :visible_vocab], dim=-1)
    pred = pred / pred.sum(dim=-1, keepdim=True)
    target = torch.tensor(target_visible, dtype=pred.dtype, device=pred.device)
    if target.ndim == 1:
        target = target.unsqueeze(0).expand_as(pred)
    return kl_divergence_torch(target, pred)


def forcing_kl(bundle: RunBundle, activation_name: str, position: int, patch_values: np.ndarray | None, target_state: int, sample_idx: np.ndarray) -> float:
    device = next(bundle.model.parameters()).device
    xb = bundle.x[sample_idx].to(device)
    bundle.model.eval()
    with torch.no_grad():
        if patch_values is None:
            logits = bundle.model(xb)[:, position, :]
        else:
            value = torch.tensor(patch_values, dtype=torch.float32, device=device)
            logits = bundle.model(xb, intervention={"name": activation_name, "position": position, "value": value})[:, position, :]
        losses = _target_kl(logits, bundle.hmm.emission[target_state], bundle.hmm.vocab_size)
    return float(losses.mean().cpu())


def layer_position_atlas(bundle: RunBundle, positions: list[int], samples: int) -> list[dict]:
    rows = []
    cal, ev = bundle.split.calibration, bundle.split.evaluation
    rng = np.random.default_rng(bundle.seed + 909)
    sample_idx = rng.choice(ev, size=min(samples, len(ev)), replace=False)
    for layer in range(bundle.cfg.n_layers):
        name = f"resid_post_{layer}"
        acts = bundle.activations[name]
        abstraction = fit_state_abstraction(acts[cal], bundle.states[cal], bundle.hmm.n_states, bundle.seed + layer)
        cal_rec = abstraction.predict(acts[cal])
        centroids = state_centroids(acts[cal], cal_rec, bundle.hmm.n_states)
        for pos in positions:
            if pos >= bundle.cfg.seq_len:
                continue
            for target in range(bundle.hmm.n_states):
                correct = forcing_kl(bundle, name, pos, centroids[target], target, sample_idx)
                wrong_state = (target + 1) % bundle.hmm.n_states
                wrong = forcing_kl(bundle, name, pos, centroids[wrong_state], target, sample_idx)
                unpatched = forcing_kl(bundle, name, pos, None, target, sample_idx)
                rows.append({
                    "label": bundle.label,
                    "seed": bundle.seed,
                    "layer": layer,
                    "position": pos,
                    "target_state": target,
                    "unpatched_kl": unpatched,
                    "correct_kl": correct,
                    "wrong_kl": wrong,
                    "selectivity_wrong_minus_correct": wrong - correct,
                    "improvement_unpatched_minus_correct": unpatched - correct,
                })
    return rows


def dose_response_and_natural_donors(bundle: RunBundle, lambdas: list[float], position: int, samples: int) -> tuple[list[dict], list[dict]]:
    layer = f"resid_post_{bundle.cfg.n_layers - 1}"
    acts = bundle.activations[layer]
    cal, ev = bundle.split.calibration, bundle.split.evaluation
    abstraction = fit_state_abstraction(acts[cal], bundle.states[cal], bundle.hmm.n_states, bundle.seed)
    cal_rec = abstraction.predict(acts[cal])
    centroids = state_centroids(acts[cal], cal_rec, bundle.hmm.n_states)
    rng = np.random.default_rng(bundle.seed + 12345)
    rec_idx = rng.choice(ev, size=min(samples, len(ev)), replace=False)
    recipient_acts = acts[rec_idx, position, :]
    dose_rows = []
    donor_rows = []
    cal_flat_acts = acts[cal, position, :]
    cal_flat_states = cal_rec[:, position]
    for target in range(bundle.hmm.n_states):
        for lam in lambdas:
            patches = recipient_acts + float(lam) * (centroids[target][None, :] - recipient_acts)
            kl = forcing_kl(bundle, layer, position, patches, target, rec_idx)
            dose_rows.append({
                "label": bundle.label,
                "seed": bundle.seed,
                "target_state": target,
                "lambda": float(lam),
                "kl_to_target": kl,
            })
        correct_pool = np.where(cal_flat_states == target)[0]
        wrong = (target + 1) % bundle.hmm.n_states
        wrong_pool = np.where(cal_flat_states == wrong)[0]
        n = len(rec_idx)
        correct_pick = rng.choice(correct_pool, size=n, replace=len(correct_pool) < n)
        wrong_pick = rng.choice(wrong_pool, size=n, replace=len(wrong_pool) < n)
        correct_values = cal_flat_acts[correct_pick]
        wrong_values = cal_flat_acts[wrong_pick]
        donor_rows.append({
            "label": bundle.label,
            "seed": bundle.seed,
            "target_state": target,
            "natural_correct_donor_kl": forcing_kl(bundle, layer, position, correct_values, target, rec_idx),
            "natural_wrong_donor_kl": forcing_kl(bundle, layer, position, wrong_values, target, rec_idx),
            "centroid_correct_kl": forcing_kl(bundle, layer, position, centroids[target], target, rec_idx),
            "unpatched_kl": forcing_kl(bundle, layer, position, None, target, rec_idx),
        })
    return dose_rows, donor_rows


def unsupervised_k_selection(bundle: RunBundle, ks: list[int]) -> tuple[list[dict], dict]:
    layer = f"resid_post_{bundle.cfg.n_layers - 1}"
    acts = bundle.activations[layer]
    cal, sel, ev = bundle.split.calibration, bundle.split.selection, bundle.split.evaluation
    vocab = bundle.hmm.vocab_size
    rows = []
    models = {}
    for k in ks:
        flat_cal = acts[cal].reshape(-1, acts.shape[-1])
        flat_sel = acts[sel].reshape(-1, acts.shape[-1])
        km = KMeans(n_clusters=k, random_state=bundle.seed + k, n_init=10)
        z_cal = km.fit_predict(flat_cal).reshape(acts[cal].shape[:-1])
        z_sel = km.predict(flat_sel).reshape(acts[sel].shape[:-1])
        token_counts = np.full((k, vocab), 0.5, dtype=np.float64)
        for z, tok in zip(z_cal.reshape(-1), bundle.tokens[cal].reshape(-1), strict=False):
            token_counts[int(z), int(tok)] += 1
        token_probs = token_counts / token_counts.sum(axis=1, keepdims=True)
        probs = token_probs[z_sel]
        sel_nll = sequence_cross_entropy(bundle.tokens[sel], probs)
        rows.append({"k": k, "selection_token_nll": sel_nll})
        models[k] = km
    chosen = min(rows, key=lambda r: r["selection_token_nll"])["k"]
    km = models[chosen]
    z_eval = km.predict(acts[ev].reshape(-1, acts.shape[-1])).reshape(acts[ev].shape[:-1])
    ari = adjusted_rand_score(bundle.states[ev].reshape(-1), z_eval.reshape(-1))
    result = {
        "label": bundle.label,
        "seed": bundle.seed,
        "true_k": bundle.hmm.n_states,
        "selected_k": int(chosen),
        "selected_matches_true": int(chosen == bundle.hmm.n_states),
        "evaluation_adjusted_rand_index": float(ari),
    }
    for row in rows:
        row.update({"label": bundle.label, "seed": bundle.seed, "true_k": bundle.hmm.n_states})
    return rows, result


def evaluate_checkpoint(model: TinyTransformer, hmm: HMM, tokens: np.ndarray, states: np.ndarray, x: torch.Tensor, split, seed: int, batch_size: int, epoch: int, val_loss: float, bayes_val_loss: float) -> dict:
    name = f"resid_post_{model.cfg.n_layers - 1}"
    acts = collect_activations(model, x, name, batch_size=batch_size).numpy()
    beliefs = bayes_predictive_state_beliefs(hmm, tokens)
    b = RunBundle(
        label=hmm.name,
        seed=seed,
        hmm=hmm,
        model=model,
        cfg=model.cfg,
        tokens=tokens,
        states=states,
        x=x,
        beliefs=beliefs,
        split=split,
        activations={name: acts},
        epochs=epoch,
        val_loss=val_loss,
        bayes_val_loss=bayes_val_loss,
    )
    row = core_metrics(b, name)
    row["checkpoint_epoch"] = epoch
    return row


def training_emergence(hmm: HMM, seed: int, arch: Arch, cfg: dict, checkpoints: list[int], deadline: Deadline) -> list[dict]:
    deadline.require(180)
    torch.manual_seed(seed)
    np.random.seed(seed)
    tcfg = cfg["training"]
    seq_len = int(tcfg["seq_len"])
    n_train = int(tcfg["train_sequences"])
    n_val = int(tcfg["model_val_sequences"])
    n_analysis = int(tcfg["analysis_sequences"])
    bs = int(tcfg["batch_size"])
    tr_tok, _ = sample_hmm_sequences(hmm, n_train, seq_len, seed=seed)
    va_tok, _ = sample_hmm_sequences(hmm, n_val, seq_len, seed=seed + 1)
    an_tok, an_states = sample_hmm_sequences(hmm, n_analysis, seq_len, seed=seed + 2)
    bos = hmm.vocab_size
    tr_x, tr_y = make_lm_tensors(tr_tok, bos)
    va_x, va_y = make_lm_tensors(va_tok, bos)
    an_x, _ = make_lm_tensors(an_tok, bos)
    split = split_sequence_indices(n_analysis, seed=seed + 3)
    mcfg = model_config(hmm, seq_len, arch)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TinyTransformer(mcfg).to(device)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(tr_x, tr_y), batch_size=bs, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=float(tcfg["learning_rate"]), weight_decay=0.01)
    bayes_val = sequence_cross_entropy(va_tok, bayes_predictive_distribution(hmm, va_tok))
    rows = []
    max_epoch = max(checkpoints)
    for epoch in range(0, max_epoch + 1):
        if epoch in checkpoints:
            val_loss = evaluate_loss(model, va_x, va_y, batch_size=bs)
            rows.append(evaluate_checkpoint(model, hmm, an_tok, an_states, an_x, split, seed, bs, epoch, val_loss, bayes_val))
        if epoch == max_epoch:
            break
        deadline.require(120)
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), yb.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    return rows


def aggregate(rows: list[dict], group_keys: list[str], metrics: list[str]) -> list[dict]:
    groups = {}
    for r in rows:
        key = tuple(r.get(k) for k in group_keys)
        groups.setdefault(key, []).append(r)
    out = []
    for key, items in groups.items():
        base = dict(zip(group_keys, key, strict=True))
        for m in metrics:
            vals = [float(x[m]) for x in items if m in x and x[m] is not None and np.isfinite(float(x[m]))]
            if vals:
                out.append({
                    **base,
                    "metric": m,
                    "mean": float(np.mean(vals)),
                    "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                    "n": len(vals),
                })
    return out


def save_phase_figures(rows: list[dict], out: Path) -> None:
    if not rows:
        return
    alphas = sorted(set(float(r["alpha"]) for r in rows))
    metrics = [
        ("transition_kl", "Transition KL"),
        ("transition_advantage_over_shuffle", "Transition advantage over shuffled control"),
        ("different_minus_same_kl", "Causal scrubbing gap"),
        ("state_recovery_accuracy", "State recovery accuracy"),
        ("belief_probe_r2", "Predictive-belief probe R²"),
    ]
    for metric, ylabel in metrics:
        xs, ys = [], []
        for a in alphas:
            vals = [float(r[metric]) for r in rows if float(r["alpha"]) == a and metric in r]
            if vals:
                xs.append(a)
                ys.append(float(np.mean(vals)))
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        ax.plot(xs, ys, marker="o")
        ax.set_xlabel("Observability interpolation α (easy → hard)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.2)
        fig.tight_layout()
        fig.savefig(out / f"phase_{metric}.svg")
        plt.close(fig)


def save_atlas_figure(rows: list[dict], out: Path) -> None:
    if not rows:
        return
    grouped = {}
    for r in rows:
        key = (int(r["layer"]), int(r["position"]))
        grouped.setdefault(key, []).append(float(r["selectivity_wrong_minus_correct"]))
    layers = sorted({k[0] for k in grouped})
    positions = sorted({k[1] for k in grouped})
    mat = np.full((len(layers), len(positions)), np.nan)
    for i, layer in enumerate(layers):
        for j, pos in enumerate(positions):
            vals = grouped.get((layer, pos), [])
            if vals:
                mat[i, j] = np.mean(vals)
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    im = ax.imshow(mat, aspect="auto")
    ax.set_xticks(range(len(positions)), positions)
    ax.set_yticks(range(len(layers)), [f"resid_post_{x}" for x in layers])
    ax.set_xlabel("Sequence position")
    ax.set_ylabel("Layer")
    fig.colorbar(im, ax=ax, label="Wrong minus correct forcing KL")
    fig.tight_layout()
    fig.savefig(out / "layer_position_causal_atlas.svg")
    plt.close(fig)


def run_suite(args: argparse.Namespace) -> dict:
    cfg = load_yaml(args.config)
    out = args.output_dir
    ensure_dirs(out)
    deadline = Deadline(args.deadline_seconds)
    base_arch = arch_from_dict(cfg["base_architecture"])
    status = {"started_at_unix": time.time(), "deadline_seconds": args.deadline_seconds, "stages": {}}
    write_json(out / "metadata" / "config_snapshot.json", cfg)

    phase_rows = []
    phase_alphas = cfg["continuous_observability"]["alphas"]
    phase_seeds = cfg["continuous_observability"]["seeds_quick" if args.quick else "seeds"]
    try:
        for alpha in phase_alphas:
            for seed in phase_seeds:
                if not deadline.can_start(240):
                    raise TimeoutError
                hmm = make_interpolated_hmm(float(alpha))
                b = train_bundle(hmm=hmm, label=hmm.name, seed=int(seed), arch=base_arch, cfg=cfg, deadline=deadline)
                row = core_metrics(b)
                row["alpha"] = float(alpha)
                phase_rows.append(row)
                write_csv(out / "tables" / "continuous_observability.csv", phase_rows)
        status["stages"]["continuous_observability"] = "complete"
    except TimeoutError:
        status["stages"]["continuous_observability"] = "partial_deadline"
    save_phase_figures(phase_rows, out / "figures")

    atlas_rows, dose_rows, donor_rows, k_rows, k_summary = [], [], [], [], []
    try:
        for regime in cfg["mechanistic_deep_dive"]["regimes"]:
            if not deadline.can_start(300):
                raise TimeoutError
            hmm = make_hmm(regime)
            seed = int(cfg["mechanistic_deep_dive"]["seed"])
            b = train_bundle(
                hmm=hmm,
                label=f"{regime}_mechanistic",
                seed=seed,
                arch=base_arch,
                cfg=cfg,
                deadline=deadline,
                collect_all_layers=True,
            )
            atlas_rows.extend(layer_position_atlas(b, [int(x) for x in cfg["mechanistic_deep_dive"]["positions"]], int(cfg["mechanistic_deep_dive"]["samples"])))
            dr, nd = dose_response_and_natural_donors(
                b,
                [float(x) for x in cfg["mechanistic_deep_dive"]["lambdas"]],
                int(cfg["mechanistic_deep_dive"]["forcing_position"]),
                int(cfg["mechanistic_deep_dive"]["samples"]),
            )
            dose_rows.extend(dr)
            donor_rows.extend(nd)
            kr, ks = unsupervised_k_selection(b, [int(x) for x in cfg["k_selection"]["ks"]])
            k_rows.extend(kr)
            k_summary.append(ks)
            write_csv(out / "tables" / "layer_position_atlas.csv", atlas_rows)
            write_csv(out / "tables" / "dose_response.csv", dose_rows)
            write_csv(out / "tables" / "natural_donor_interventions.csv", donor_rows)
            write_csv(out / "tables" / "k_selection_curve.csv", k_rows)
            write_csv(out / "tables" / "k_selection_summary.csv", k_summary)
        status["stages"]["mechanistic_deep_dive"] = "complete"
    except TimeoutError:
        status["stages"]["mechanistic_deep_dive"] = "partial_deadline"
    save_atlas_figure(atlas_rows, out / "figures")

    arch_rows = []
    try:
        arch_seeds = cfg["architecture_sweep"]["seeds_quick" if args.quick else "seeds"]
        for ad in cfg["architecture_sweep"]["architectures"]:
            arch = arch_from_dict(ad)
            for regime in cfg["architecture_sweep"]["regimes"]:
                for seed in arch_seeds:
                    if not deadline.can_start(240):
                        raise TimeoutError
                    b = train_bundle(hmm=make_hmm(regime), label=f"arch={arch.name}|obs={regime}", seed=int(seed), arch=arch, cfg=cfg, deadline=deadline)
                    row = core_metrics(b)
                    row["architecture"] = arch.name
                    row["observability"] = regime
                    arch_rows.append(row)
                    write_csv(out / "tables" / "architecture_sweep.csv", arch_rows)
        status["stages"]["architecture_sweep"] = "complete"
    except TimeoutError:
        status["stages"]["architecture_sweep"] = "partial_deadline"

    world_rows = []
    try:
        worlds = int(cfg["random_hmm_worlds"]["worlds_quick" if args.quick else "worlds"])
        ks = [int(x) for x in cfg["random_hmm_worlds"]["state_counts"]]
        persistence_vals = [float(x) for x in cfg["random_hmm_worlds"]["persistence"]]
        obs_vals = [float(x) for x in cfg["random_hmm_worlds"]["observability"]]
        for i in range(worlds):
            if not deadline.can_start(240):
                raise TimeoutError
            seed = int(cfg["random_hmm_worlds"]["seed_base"]) + i
            k = ks[i % len(ks)]
            p = persistence_vals[(i // len(ks)) % len(persistence_vals)]
            o = obs_vals[(i // (len(ks) * len(persistence_vals))) % len(obs_vals)]
            vocab = max(6, k + 2)
            hmm = make_random_hmm(seed, k, vocab, p, o)
            b = train_bundle(hmm=hmm, label=hmm.name, seed=seed, arch=base_arch, cfg=cfg, deadline=deadline)
            row = core_metrics(b)
            row.update({"world_index": i, "persistence": p, "observability_strength": o})
            world_rows.append(row)
            write_csv(out / "tables" / "random_hmm_worlds.csv", world_rows)
        status["stages"]["random_hmm_worlds"] = "complete"
    except TimeoutError:
        status["stages"]["random_hmm_worlds"] = "partial_deadline"

    emergence_rows = []
    try:
        for regime in cfg["training_emergence"]["regimes"]:
            if not deadline.can_start(360):
                raise TimeoutError
            rows = training_emergence(
                make_hmm(regime),
                int(cfg["training_emergence"]["seed"]),
                base_arch,
                cfg,
                [int(x) for x in cfg["training_emergence"]["checkpoints"]],
                deadline,
            )
            for r in rows:
                r["observability"] = regime
            emergence_rows.extend(rows)
            write_csv(out / "tables" / "training_emergence.csv", emergence_rows)
        status["stages"]["training_emergence"] = "complete"
    except TimeoutError:
        status["stages"]["training_emergence"] = "partial_deadline"

    seed_rows = []
    try:
        seeds = cfg["seed_stress"]["seeds_quick" if args.quick else "seeds"]
        for regime in cfg["seed_stress"]["regimes"]:
            for seed in seeds:
                if not deadline.can_start(240):
                    raise TimeoutError
                b = train_bundle(hmm=make_hmm(regime), label=f"seed_stress_{regime}", seed=int(seed), arch=base_arch, cfg=cfg, deadline=deadline)
                row = core_metrics(b)
                row["observability"] = regime
                seed_rows.append(row)
                write_csv(out / "tables" / "seed_stress.csv", seed_rows)
        status["stages"]["seed_stress"] = "complete"
    except TimeoutError:
        status["stages"]["seed_stress"] = "partial_deadline"

    combined = phase_rows + arch_rows + world_rows + seed_rows + emergence_rows
    agg = aggregate(
        combined,
        ["label"],
        ["transition_kl", "transition_advantage_over_shuffle", "state_recovery_accuracy", "belief_probe_r2", "different_minus_same_kl", "order1_gain_over_order0"],
    )
    write_csv(out / "tables" / "aggregate_metrics.csv", agg)
    status["finished_at_unix"] = time.time()
    status["elapsed_seconds"] = deadline.elapsed
    status["remaining_seconds"] = deadline.remaining
    status["row_counts"] = {
        "continuous_observability": len(phase_rows),
        "atlas": len(atlas_rows),
        "dose_response": len(dose_rows),
        "natural_donors": len(donor_rows),
        "k_selection": len(k_summary),
        "architecture": len(arch_rows),
        "random_worlds": len(world_rows),
        "training_emergence": len(emergence_rows),
        "seed_stress": len(seed_rows),
    }
    write_json(out / "STATUS.json", status)
    return status


def main() -> None:
    args = parse_args()
    torch.set_num_threads(int(os.getenv("MCT_TORCH_THREADS", "2")))
    ensure_dirs(args.output_dir)
    metadata = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "started_at_unix": time.time(),
    }
    write_json(args.output_dir / "metadata" / "environment.json", metadata)
    try:
        status = run_suite(args)
    except Exception as exc:
        write_json(args.output_dir / "FAILURE.json", {"type": type(exc).__name__, "message": str(exc), "time": time.time()})
        raise
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()

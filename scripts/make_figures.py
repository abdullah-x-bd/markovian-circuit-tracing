from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("results/v1")
FIG = ROOT / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def main_figure() -> None:
    run = ROOT / "runs" / "medium_seed07"
    true_t = np.load(run / "true_transition.npy")
    rec_t = np.load(run / "recovered_transition.npy")
    forcing = load_csv(ROOT / "tables" / "forcing_summary.csv")
    run_metrics = load_csv(ROOT / "tables" / "run_metrics.csv")

    fig, axes = plt.subplots(1, 5, figsize=(21, 3.8))
    im0 = axes[0].imshow(true_t, vmin=0, vmax=max(true_t.max(), rec_t.max()))
    axes[0].set_title("A  True transition")
    axes[0].set_xlabel("to state"); axes[0].set_ylabel("from state")
    axes[1].imshow(rec_t, vmin=0, vmax=max(true_t.max(), rec_t.max()))
    axes[1].set_title("B  Recovered transition")
    axes[1].set_xlabel("to state")
    diff = true_t - rec_t
    lim = max(abs(diff.min()), abs(diff.max()))
    im2 = axes[2].imshow(diff, vmin=-lim, vmax=lim)
    axes[2].set_title("C  True minus recovered")
    axes[2].set_xlabel("to state")

    patch_order = ["unpatched", "recovered_centroid", "wrong_recovered_centroid", "mean_activation", "random_activation", "true_state_centroid_oracle"]
    medium = {r["patch_type"]: float(r["kl_mean"]) for r in forcing if r["observability"] == "medium"}
    vals = [medium[p] for p in patch_order if p in medium]
    labels = [p.replace("_", "\n") for p in patch_order if p in medium]
    axes[3].bar(range(len(vals)), vals)
    axes[3].set_xticks(range(len(vals)), labels, rotation=70, ha="right", fontsize=7)
    axes[3].set_ylabel("KL(target || model)")
    axes[3].set_title("D  State-forcing controls")

    medium_runs = [r for r in run_metrics if r["observability"] == "medium"]
    same = [float(r["same_scrub_kl"]) for r in medium_runs]
    diffk = [float(r["different_scrub_kl"]) for r in medium_runs]
    means = [np.mean(same), np.mean(diffk)]
    errs = [1.96 * np.std(same, ddof=1) / np.sqrt(len(same)), 1.96 * np.std(diffk, ddof=1) / np.sqrt(len(diffk))]
    axes[4].bar([0, 1], means, yerr=errs, capsize=4)
    axes[4].set_xticks([0, 1], ["same state", "different state"])
    axes[4].set_ylabel("Output KL after swap")
    axes[4].set_title("E  Causal scrubbing")
    fig.colorbar(im0, ax=axes[:2], fraction=0.02, pad=0.01)
    fig.colorbar(im2, ax=axes[2], fraction=0.05, pad=0.03)
    fig.subplots_adjust(left=0.04, right=0.99, bottom=0.29, top=0.82, wspace=0.45)
    fig.savefig(FIG / "figure_1_main.png", dpi=220)
    fig.savefig(FIG / "figure_1_main.pdf")
    fig.savefig(FIG / "figure_1_main.svg")
    plt.close(fig)


def observability_figure() -> None:
    rows = load_csv(ROOT / "tables" / "summary_by_observability.csv")
    order = ["easy", "medium", "hard"]
    lookup = {r["observability"]: r for r in rows}
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    x = np.arange(3)
    belief = [float(lookup[o]["belief_kl_mean"]) for o in order]
    hist4 = [float(lookup[o]["history4_belief_kl_mean"]) for o in order]
    untrained = [float(lookup[o]["untrained_belief_kl_mean"]) for o in order]
    width = 0.25
    ax.bar(x - width, belief, width, label="trained transformer")
    ax.bar(x, hist4, width, label="4-token history")
    ax.bar(x + width, untrained, width, label="untrained transformer")
    ax.set_xticks(x, order)
    ax.set_ylabel("Held-out predictive-belief KL")
    ax.set_xlabel("HMM observability")
    ax.set_title("Predictive-belief recovery across observability")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "figure_2_belief_recovery.png", dpi=220)
    fig.savefig(FIG / "figure_2_belief_recovery.pdf")
    fig.savefig(FIG / "figure_2_belief_recovery.svg")
    plt.close(fig)


def sae_figure() -> None:
    rows = load_csv(ROOT / "tables" / "summary_by_observability.csv")
    order = ["easy", "medium", "hard"]
    lookup = {r["observability"]: r for r in rows}
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    x = np.arange(3)
    raw = [float(lookup[o]["transition_kl_mean"]) for o in order]
    sae = [float(lookup[o]["sae_transition_kl_mean"]) for o in order]
    width = 0.34
    ax.bar(x - width / 2, raw, width, label="raw residual state abstraction")
    ax.bar(x + width / 2, sae, width, label="fixed SAE state abstraction")
    ax.set_xticks(x, order)
    ax.set_ylabel("Transition rowwise KL")
    ax.set_xlabel("HMM observability")
    ax.set_title("Does a fixed SAE improve transition recovery?")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "figure_3_sae_comparison.png", dpi=220)
    fig.savefig(FIG / "figure_3_sae_comparison.pdf")
    fig.savefig(FIG / "figure_3_sae_comparison.svg")
    plt.close(fig)


if __name__ == "__main__":
    main_figure()
    observability_figure()
    sae_figure()
    print(f"Wrote figures to {FIG}")

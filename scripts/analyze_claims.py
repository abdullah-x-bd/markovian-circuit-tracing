from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, stdev

import numpy as np
from scipy.stats import ttest_1samp

ROOT = Path("results/v1")
OBS = ("easy", "medium", "hard")
SEEDS = (7, 17, 29, 43, 71)


def ci95(values: list[float]) -> float:
    return 0.0 if len(values) < 2 else 1.96 * stdev(values) / np.sqrt(len(values))


def effect(values: list[float]) -> dict:
    t = ttest_1samp(values, popmean=0.0)
    return {
        "mean": mean(values),
        "sd": stdev(values) if len(values) > 1 else 0.0,
        "ci95_half_width": ci95(values),
        "t_statistic": float(t.statistic),
        "p_value_two_sided": float(t.pvalue),
        "n": len(values),
        "values_by_seed": dict(zip(SEEDS, values, strict=True)),
    }


def load_runs():
    runs, forcing = {}, {}
    for obs in OBS:
        for seed in SEEDS:
            d = ROOT / "runs" / f"{obs}_seed{seed:02d}"
            runs[(obs, seed)] = json.loads((d / "metrics.json").read_text())
            forcing[(obs, seed)] = json.loads((d / "forcing_controls.json").read_text())
    return runs, forcing


def forcing_seed_mean(rows, patch):
    vals = [float(r["kl_to_target"]) for r in rows if r["patch_type"] == patch]
    return float(np.mean(vals))


def main():
    runs, forcing = load_runs()
    details, statuses = {}, {}
    val_gaps = [runs[(o,s)]["model_quality"]["model_validation_excess_over_bayes"] for o in OBS for s in SEEDS]
    eval_gaps = [runs[(o,s)]["model_quality"]["model_excess_loss_over_bayes"] for o in OBS for s in SEEDS]
    details["near_bayes_prediction"] = {
        "all_validation_targets_reached": all(runs[(o,s)]["model_quality"]["stopping_target_reached"] for o in OBS for s in SEEDS),
        "max_validation_gap": max(val_gaps),
        "mean_evaluation_gap": mean(eval_gaps),
        "max_evaluation_gap": max(eval_gaps),
    }
    statuses["near_bayes_prediction"] = "supported" if details["near_bayes_prediction"]["all_validation_targets_reached"] and mean(eval_gaps) < 0.03 else "not_supported"

    for obs in OBS:
        trans_vs_shuffled = [runs[(obs,s)]["transition_recovery"]["shuffled_rowwise_kl"] - runs[(obs,s)]["transition_recovery"]["rowwise_kl"] for s in SEEDS]
        trans_vs_random = [runs[(obs,s)]["transition_recovery"]["random_rowwise_kl"] - runs[(obs,s)]["transition_recovery"]["rowwise_kl"] for s in SEEDS]
        scrub = [runs[(obs,s)]["causal_scrubbing"]["different_minus_same_kl"] for s in SEEDS]
        trained_vs_untrained = [runs[(obs,s)]["untrained_transformer_baseline"]["belief_probe_kl"] - runs[(obs,s)]["belief_recovery"]["belief_probe_kl"] for s in SEEDS]
        trained_vs_history4 = [runs[(obs,s)]["token_history_baselines"]["history4_belief_probe_kl"] - runs[(obs,s)]["belief_recovery"]["belief_probe_kl"] for s in SEEDS]
        raw_vs_sae = [runs[(obs,s)]["sae"]["heldout_evaluation"]["rowwise_kl"] - runs[(obs,s)]["transition_recovery"]["rowwise_kl"] for s in SEEDS]
        force_vs_unpatched = [forcing_seed_mean(forcing[(obs,s)], "unpatched") - forcing_seed_mean(forcing[(obs,s)], "recovered_centroid") for s in SEEDS]
        force_vs_wrong = [forcing_seed_mean(forcing[(obs,s)], "wrong_recovered_centroid") - forcing_seed_mean(forcing[(obs,s)], "recovered_centroid") for s in SEEDS]
        details[obs] = {
            "transition_advantage_vs_shuffled": effect(trans_vs_shuffled),
            "transition_advantage_vs_random": effect(trans_vs_random),
            "causal_scrubbing_gap": effect(scrub),
            "belief_kl_advantage_vs_untrained": effect(trained_vs_untrained),
            "belief_kl_advantage_vs_4token_history": effect(trained_vs_history4),
            "sae_minus_raw_transition_kl": effect(raw_vs_sae),
            "forcing_advantage_vs_unpatched": effect(force_vs_unpatched),
            "forcing_advantage_vs_wrong_centroid": effect(force_vs_wrong),
        }
        statuses[f"transition_recovery_{obs}"] = "supported" if mean(trans_vs_shuffled) > 0 and mean(trans_vs_random) > 0 else "not_supported"
        scrub_effect = details[obs]["causal_scrubbing_gap"]
        force_unpatched = details[obs]["forcing_advantage_vs_unpatched"]
        force_wrong = details[obs]["forcing_advantage_vs_wrong_centroid"]
        statuses[f"causal_scrubbing_{obs}"] = (
            "supported_small_effect" if obs == "hard" and scrub_effect["mean"] > 0 and scrub_effect["p_value_two_sided"] < 0.05
            else "supported" if scrub_effect["mean"] > 0 and scrub_effect["p_value_two_sided"] < 0.05
            else "not_supported"
        )
        statuses[f"state_forcing_{obs}"] = (
            "supported" if force_unpatched["mean"] > 0 and force_wrong["mean"] > 0
            and force_unpatched["p_value_two_sided"] < 0.05 and force_wrong["p_value_two_sided"] < 0.05
            else "not_supported"
        )

    statuses["learned_belief_specificity"] = "not_supported"
    statuses["sae_improves_transition_recovery"] = "not_supported" if all(details[o]["sae_minus_raw_transition_kl"]["mean"] >= 0 for o in OBS) else "mixed"
    easy_t = mean(runs[("easy",s)]["transition_recovery"]["rowwise_kl"] for s in SEEDS)
    medium_t = mean(runs[("medium",s)]["transition_recovery"]["rowwise_kl"] for s in SEEDS)
    hard_t = mean(runs[("hard",s)]["transition_recovery"]["rowwise_kl"] for s in SEEDS)
    easy_scrub = mean(runs[("easy",s)]["causal_scrubbing"]["different_minus_same_kl"] for s in SEEDS)
    medium_scrub = mean(runs[("medium",s)]["causal_scrubbing"]["different_minus_same_kl"] for s in SEEDS)
    hard_scrub = mean(runs[("hard",s)]["causal_scrubbing"]["different_minus_same_kl"] for s in SEEDS)
    statuses["observability_dependence"] = "supported" if easy_t < medium_t < hard_t and easy_scrub > medium_scrub > hard_scrub else "not_supported"
    details["observability_dependence"] = {
        "transition_kl_means": {"easy": easy_t, "medium": medium_t, "hard": hard_t},
        "scrubbing_gap_means": {"easy": easy_scrub, "medium": medium_scrub, "hard": hard_scrub},
    }
    (ROOT / "claims.json").write_text(json.dumps({"artifact_schema_version": "1.0", "claim_status": statuses, "details": details}, indent=2))
    print(json.dumps(statuses, indent=2))


if __name__ == "__main__":
    main()

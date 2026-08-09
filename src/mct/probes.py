from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score


def _flatten(x: np.ndarray) -> np.ndarray:
    return x.reshape(-1, x.shape[-1])


def _flatten_labels(y: np.ndarray) -> np.ndarray:
    return y.reshape(-1)


def _row_normalize_positive(values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    values = np.clip(values, eps, None)
    return values / values.sum(axis=-1, keepdims=True)


def _kl(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return float(np.mean(np.sum(p * (np.log(p) - np.log(q)), axis=-1)))


def expected_calibration_error(probs: np.ndarray, targets: np.ndarray, n_bins: int = 10) -> float:
    confidences = probs.max(axis=-1)
    predictions = probs.argmax(axis=-1)
    labels = targets.argmax(axis=-1)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:], strict=True):
        mask = (confidences > lo) & (confidences <= hi)
        if not mask.any():
            continue
        accuracy = (predictions[mask] == labels[mask]).mean()
        confidence = confidences[mask].mean()
        ece += mask.mean() * abs(float(accuracy - confidence))
    return float(ece)


def belief_probe_metrics(
    calibration_activations: np.ndarray,
    calibration_beliefs: np.ndarray,
    evaluation_activations: np.ndarray,
    evaluation_beliefs: np.ndarray,
    evaluation_true_states: np.ndarray | None = None,
    alpha: float = 1.0,
) -> dict[str, float]:
    """Fit on calibration sequences and score only on held-out evaluation sequences."""
    x_cal = _flatten(calibration_activations)
    y_cal = _flatten(calibration_beliefs)
    x_eval = _flatten(evaluation_activations)
    y_eval = _flatten(evaluation_beliefs)
    reg = Ridge(alpha=alpha)
    reg.fit(x_cal, y_cal)
    pred = _row_normalize_positive(reg.predict(x_eval))
    result = {
        "belief_probe_mse": float(mean_squared_error(y_eval, pred)),
        "belief_probe_kl": _kl(y_eval, pred),
        "belief_probe_r2": float(r2_score(y_eval, pred, multioutput="variance_weighted")),
        "belief_probe_argmax_accuracy": float((pred.argmax(axis=-1) == y_eval.argmax(axis=-1)).mean()),
        "belief_probe_ece": expected_calibration_error(pred, y_eval),
    }
    if evaluation_true_states is not None:
        result["belief_probe_true_state_accuracy"] = float(
            (pred.argmax(axis=-1) == _flatten_labels(evaluation_true_states)).mean()
        )
    return result


def state_probe_metrics(
    calibration_activations: np.ndarray,
    calibration_states: np.ndarray,
    evaluation_activations: np.ndarray,
    evaluation_states: np.ndarray,
) -> dict[str, float]:
    clf = LogisticRegression(max_iter=1000, solver="lbfgs")
    clf.fit(_flatten(calibration_activations), _flatten_labels(calibration_states))
    pred = clf.predict(_flatten(evaluation_activations))
    return {"state_probe_accuracy": float(accuracy_score(_flatten_labels(evaluation_states), pred))}


def bayes_state_classification_ceiling(predictive_beliefs: np.ndarray, true_states: np.ndarray) -> float:
    """Best sampled-state classification available from observations x_<t alone."""
    return float((predictive_beliefs.argmax(axis=-1) == true_states).mean())

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


def _flatten_xy(activations: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = activations.reshape(-1, activations.shape[-1])
    y = targets.reshape(-1, targets.shape[-1])
    return x, y


def _row_normalize_positive(values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    values = np.clip(values, eps, None)
    return values / values.sum(axis=-1, keepdims=True)


def _kl(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return float(np.mean(np.sum(p * (np.log(p) - np.log(q)), axis=-1)))


def belief_probe_metrics(
    activations: np.ndarray,
    belief_targets: np.ndarray,
    true_states: np.ndarray,
    test_size: float = 0.25,
    seed: int = 0,
) -> dict[str, float]:
    """Fit a linear probe from activations to exact Bayesian belief vectors."""
    x, y = _flatten_xy(activations, belief_targets)
    states = true_states.reshape(-1)
    x_train, x_test, y_train, y_test, _, states_test = train_test_split(
        x,
        y,
        states,
        test_size=test_size,
        random_state=seed,
        stratify=states,
    )

    reg = Ridge(alpha=1.0)
    reg.fit(x_train, y_train)
    raw_pred = reg.predict(x_test)
    pred = _row_normalize_positive(raw_pred)

    belief_argmax = y_test.argmax(axis=-1)
    pred_argmax = pred.argmax(axis=-1)

    return {
        "belief_probe_mse": float(mean_squared_error(y_test, pred)),
        "belief_probe_kl": _kl(y_test, pred),
        "belief_probe_argmax_accuracy": float((pred_argmax == belief_argmax).mean()),
        "belief_probe_true_state_accuracy": float((pred_argmax == states_test).mean()),
    }

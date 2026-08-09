import numpy as np

from mct.history_baselines import token_history_features
from mct.probes import bayes_state_classification_ceiling, belief_probe_metrics, state_probe_metrics


def test_belief_and_state_probes_fit_calibration_then_evaluate():
    rng = np.random.default_rng(1)
    cal_b = rng.dirichlet([2, 2, 2], size=(20, 5))
    ev_b = rng.dirichlet([2, 2, 2], size=(8, 5))
    projection = rng.normal(size=(3, 10))
    cal_a = cal_b @ projection
    ev_a = ev_b @ projection
    cal_s = cal_b.argmax(-1)
    ev_s = ev_b.argmax(-1)
    belief = belief_probe_metrics(cal_a, cal_b, ev_a, ev_b, ev_s)
    state = state_probe_metrics(cal_a, cal_s, ev_a, ev_s)
    assert belief["belief_probe_mse"] < 0.02
    assert state["state_probe_accuracy"] > 0.75
    assert bayes_state_classification_ceiling(ev_b, ev_s) == 1.0


def test_token_history_features_do_not_include_current_token():
    tokens = np.array([[1, 2, 0]])
    feats = token_history_features(tokens, vocab_size=3, history=1)
    assert feats[0, 0, 3] == 1.0
    assert feats[0, 1, 1] == 1.0
    assert feats[0, 1, 2] == 0.0

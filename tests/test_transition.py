import numpy as np

from mct.transition import estimate_transition_matrix, markov_order_report


def test_transition_estimator_normalizes():
    states = np.array([[0, 1, 1, 2], [2, 1, 0, 0]])
    t = estimate_transition_matrix(states, n_states=3)
    assert np.allclose(t.sum(axis=-1), 1.0)


def test_markov_order_report_is_fit_on_calibration_and_scored_on_eval():
    cal = np.tile(np.array([[0, 1, 0, 1, 0, 1]]), (20, 1))
    ev = np.tile(np.array([[0, 1, 0, 1, 0, 1]]), (5, 1))
    report = markov_order_report(cal, ev, 2)
    assert report["order1_accuracy"] > 0.99

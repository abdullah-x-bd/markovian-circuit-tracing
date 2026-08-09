import numpy as np

from mct.splits import split_sequence_indices
from mct.states import fit_state_abstraction, state_recovery_accuracy


def test_sequence_splits_are_disjoint_and_complete():
    split = split_sequence_indices(100, seed=3)
    sets = [set(split.calibration), set(split.selection), set(split.evaluation)]
    assert sets[0].isdisjoint(sets[1])
    assert sets[0].isdisjoint(sets[2])
    assert sets[1].isdisjoint(sets[2])
    assert set.union(*sets) == set(range(100))
    assert [len(split.calibration), len(split.selection), len(split.evaluation)] == [60, 20, 20]


def test_state_abstraction_uses_calibration_mapping_and_predicts_heldout():
    rng = np.random.default_rng(0)
    centers = np.array([[-3.0, 0.0], [0.0, 3.0], [3.0, 0.0]])
    cal_states = np.tile(np.array([[0, 1, 2, 0, 1, 2]]), (30, 1))
    ev_states = np.tile(np.array([[2, 1, 0, 2, 1, 0]]), (10, 1))
    cal = centers[cal_states] + rng.normal(0, 0.1, size=(*cal_states.shape, 2))
    ev = centers[ev_states] + rng.normal(0, 0.1, size=(*ev_states.shape, 2))
    abstraction = fit_state_abstraction(cal, cal_states, n_states=3, seed=0)
    assert state_recovery_accuracy(abstraction, ev, ev_states) > 0.98

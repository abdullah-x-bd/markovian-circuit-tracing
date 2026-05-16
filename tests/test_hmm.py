import numpy as np

from mct.data import bayes_filter, default_hmm, sample_hmm_sequences
from mct.transition import estimate_transition_matrix


def test_hmm_shapes():
    hmm = default_hmm()
    tokens, states = sample_hmm_sequences(hmm, n_sequences=5, seq_len=11, seed=0)
    assert tokens.shape == (5, 11)
    assert states.shape == (5, 11)
    assert tokens.max() < hmm.vocab_size
    assert states.max() < hmm.n_states


def test_bayes_filter_normalizes():
    hmm = default_hmm()
    tokens, _ = sample_hmm_sequences(hmm, n_sequences=5, seq_len=11, seed=0)
    beliefs = bayes_filter(hmm, tokens)
    assert beliefs.shape == (5, 11, hmm.n_states)
    assert np.allclose(beliefs.sum(axis=-1), 1.0)


def test_transition_estimator_normalizes():
    states = np.array([[0, 1, 1, 2], [2, 1, 0, 0]])
    t = estimate_transition_matrix(states, n_states=3)
    assert t.shape == (3, 3)
    assert np.allclose(t.sum(axis=-1), 1.0)

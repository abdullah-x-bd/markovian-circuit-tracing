import numpy as np

from mct.data import (
    HMM,
    bayes_filter,
    bayes_predictive_state_beliefs,
    current_state_emission_distribution,
    next_state_predictive_distribution,
)


def tiny_hmm():
    return HMM(
        transition=np.array([[0.8, 0.2], [0.3, 0.7]], dtype=float),
        emission=np.array([[0.9, 0.1], [0.2, 0.8]], dtype=float),
        initial=np.array([0.6, 0.4], dtype=float),
    )


def test_predictive_prior_is_before_current_observation():
    hmm = tiny_hmm()
    tokens = np.array([[0, 1]])
    priors = bayes_predictive_state_beliefs(hmm, tokens)
    assert np.allclose(priors[0, 0], hmm.initial)
    posterior0 = hmm.initial * hmm.emission[:, 0]
    posterior0 /= posterior0.sum()
    assert np.allclose(priors[0, 1], posterior0 @ hmm.transition)


def test_filter_is_after_current_observation():
    hmm = tiny_hmm()
    tokens = np.array([[0]])
    posterior = hmm.initial * hmm.emission[:, 0]
    posterior /= posterior.sum()
    assert np.allclose(bayes_filter(hmm, tokens)[0, 0], posterior)


def test_current_and_next_state_targets_are_distinct():
    hmm = tiny_hmm()
    assert np.allclose(current_state_emission_distribution(hmm, 0), hmm.emission[0])
    assert np.allclose(next_state_predictive_distribution(hmm, 0), hmm.transition[0] @ hmm.emission)
    assert not np.allclose(
        current_state_emission_distribution(hmm, 0),
        next_state_predictive_distribution(hmm, 0),
    )

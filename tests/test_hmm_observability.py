import numpy as np

from mct.data import make_hmm


def mean_pairwise_emission_distance(hmm):
    rows = hmm.emission
    values = []
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            values.append(np.abs(rows[i] - rows[j]).sum())
    return float(np.mean(values))


def test_observability_family_keeps_latent_dynamics_fixed():
    easy = make_hmm("easy")
    medium = make_hmm("medium")
    hard = make_hmm("hard")
    assert np.allclose(easy.transition, medium.transition)
    assert np.allclose(medium.transition, hard.transition)
    assert mean_pairwise_emission_distance(easy) > mean_pairwise_emission_distance(medium)
    assert mean_pairwise_emission_distance(medium) > mean_pairwise_emission_distance(hard)

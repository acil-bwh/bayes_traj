from bayes_traj.utils import sample_cos, sample_precs
import numpy as np
import pdb

def test_sample_cos():
    """
    """
    M = 2
    D = 1
    w_mu0 = np.atleast_2d(np.array([3, 7])).T
    w_var0 = np.atleast_2d(np.array([.1, 2])).T

    w = sample_cos(w_mu0, w_var0, num_samples=100000)

    assert np.sum(np.isclose(w_mu0, np.mean(w, 2), atol=.01)) == 2, \
        "Unexpected mean"
    assert np.sum(np.isclose(w_var0, np.var(w, 2), atol=.1)) == 2, \
        "Unexpected variance"

def test_sample_precs():
    lambda_a0 = np.array([1, 5])
    lambda_b0 = np.array([2, 13])
    precs = sample_precs(lambda_a0, lambda_b0, num_samples=100000)

    assert np.sum(np.isclose(lambda_a0/lambda_b0, \
                             np.mean(precs, 1), atol=.01)) == 2, \
        "Unexpected precision"

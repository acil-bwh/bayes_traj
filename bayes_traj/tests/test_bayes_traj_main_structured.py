"""Focused tests for structured-DP main-program helpers."""

import numpy as np

from bayes_traj.bayes_traj_main import _expand_structured_prior_to_truncation


def test_expand_structured_prior_preserves_existing_and_pads_new_slots():
    prior = {
        'alpha': 2.5,
        'v_a': np.array([3.0, 2.0]),
        'v_b': np.array([4.0, 5.0]),
        'w_mu': np.arange(8.0).reshape(2, 2, 2),
        'w_var': np.ones((2, 2, 2)),
        'lambda_a': np.ones((2, 2)) * 3.0,
        'lambda_b': np.ones((2, 2)) * 4.0,
        'traj_probs': np.array([0.7, 0.3]),
        'R': np.array([[0.8, 0.2], [0.1, 0.9]]),
    }
    old_w = prior['w_mu'].copy()
    out = _expand_structured_prior_to_truncation(prior, 2, 5)

    assert out['v_a'].shape == (5,)
    assert out['w_mu'].shape == (2, 2, 5)
    assert out['lambda_a'].shape == (2, 5)
    assert out['R'].shape == (2, 5)
    np.testing.assert_allclose(out['w_mu'][:, :, :2], old_w)
    assert np.all(np.isnan(out['w_mu'][:, :, 2:]))
    np.testing.assert_allclose(out['v_a'][2:], 1.0)
    np.testing.assert_allclose(out['v_b'][2:], 2.5)
    np.testing.assert_allclose(out['traj_probs'][2:], 0.0)
    np.testing.assert_allclose(out['R'][:, 2:], 0.0)


def test_expand_structured_prior_handles_absent_optional_initializers():
    prior = {
        'alpha': 1.0,
        'v_a': None,
        'v_b': None,
        'w_mu': None,
        'w_var': None,
        'lambda_a': None,
        'lambda_b': None,
        'traj_probs': None,
        'R': None,
    }
    out = _expand_structured_prior_to_truncation(prior, 2, 5)
    for key in ('v_a', 'v_b', 'w_mu', 'w_var', 'lambda_a',
                'lambda_b', 'traj_probs', 'R'):
        assert out[key] is None

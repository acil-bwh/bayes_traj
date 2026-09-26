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



def test_structured_main_exposes_resume_cli_contract():
    import inspect
    import bayes_traj.bayes_traj_main as mod
    src = inspect.getsource(mod.main)
    assert '--resume_model' in src
    assert 'continue_structured_fit' in src
    assert 'objective_converged' in src
    assert 'final_max_delta_r' in src


def test_material_negative_elbo_step_uses_scale_aware_tolerance():
    from bayes_traj.bayes_traj_main import _material_negative_elbo_steps
    hist = [
        {'elbo': -10000.0000005, 'delta_elbo': -5e-7},  # numerical: tol 1e-6
        {'elbo': -10000.001, 'delta_elbo': -1e-3},       # material
    ]
    n, worst, max_tol = _material_negative_elbo_steps(hist, rel_tol=1e-10)
    assert n == 1
    assert np.isclose(worst, -1e-3)
    assert max_tol >= 1e-6


def _eligibility_dummy():
    import torch
    from types import SimpleNamespace
    mm = SimpleNamespace()
    mm.objective_converged_ = True
    mm.parameter_converged_ = False
    mm.fit_complete_ = True
    mm.w_mu_ = torch.zeros((1, 1, 2), dtype=torch.float64)
    mm.w_var_ = torch.ones((1, 1, 2), dtype=torch.float64)
    mm.lambda_a_ = torch.ones((1, 2), dtype=torch.float64)
    mm.lambda_b_ = torch.ones((1, 2), dtype=torch.float64)
    mm.v_a_ = torch.ones(2, dtype=torch.float64)
    mm.v_b_ = torch.ones(2, dtype=torch.float64)
    mm.ranef_cov_strategy_ = 'fixed'
    mm._get_group_responsibilities = lambda: torch.tensor(
        [[0.8, 0.2], [0.1, 0.9]], dtype=torch.float64)
    return mm


def test_structured_repeat_eligibility_uses_objective_not_parameter_convergence():
    from bayes_traj.bayes_traj_main import _structured_repeat_eligibility
    mm = _eligibility_dummy()
    eligible, reasons = _structured_repeat_eligibility(mm, -123.0, 0)
    assert eligible is True
    assert reasons == []


def test_structured_repeat_eligibility_rejects_material_pathology():
    from bayes_traj.bayes_traj_main import _structured_repeat_eligibility
    mm = _eligibility_dummy()
    eligible, reasons = _structured_repeat_eligibility(mm, -123.0, 1)
    assert eligible is False
    assert 'negative_elbo_step' in reasons

    mm.objective_converged_ = False
    eligible, reasons = _structured_repeat_eligibility(mm, -123.0, 0)
    assert eligible is False
    assert 'objective_not_converged' in reasons


def test_staged_repeat_must_release_D_before_selection():
    from bayes_traj.bayes_traj_main import _structured_repeat_eligibility
    mm = _eligibility_dummy()
    mm.ranef_cov_strategy_ = 'staged'
    mm.ranef_cov_release_iteration_ = None
    mm.ranef_cov_stage_ = 'fixed'
    eligible, reasons = _structured_repeat_eligibility(mm, -123.0, 0)
    assert eligible is False
    assert 'staged_D_not_released' in reasons
    assert 'staged_D_not_in_final_phase' in reasons

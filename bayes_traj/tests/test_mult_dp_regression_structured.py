#!/usr/bin/env python3
"""Focused tests for structured random effects in MultDPRegression."""

import inspect
import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from bayes_traj.mult_dp_regression import MultDPRegression


def build_manual_model(seed=3, g=30, nvis=5, k=2):
    rng = np.random.default_rng(seed)
    n = g * nvis
    ids = np.repeat(np.arange(g), nvis)
    time = np.tile(np.linspace(0.0, 1.5, nvis), g)
    x = np.column_stack([np.ones(n), time])

    cls = np.arange(g) % k
    beta = np.array([[0.3, -0.8], [-0.25, 0.55]])
    dtrue = np.array([[0.25, -0.03], [-0.03, 0.07]])
    b = rng.multivariate_normal(np.zeros(2), dtrue, size=g)
    y = np.empty(n)
    for i in range(g):
        rows = ids == i
        y[rows] = (
            x[rows] @ beta[:, cls[i]]
            + x[rows] @ b[i]
            + rng.normal(0.0, 0.18, rows.sum())
        )

    r_group = np.full((g, k), 0.15 / max(1, k - 1))
    r_group[np.arange(g), cls] = 0.85
    r_group /= r_group.sum(axis=1, keepdims=True)
    r = r_group[ids]

    mm = MultDPRegression.__new__(MultDPRegression)
    mm.X_ = torch.tensor(x, dtype=torch.float64)
    mm.Y_ = torch.tensor(y[:, None], dtype=torch.float64)
    mm.N_, mm.M_, mm.D_, mm.K_, mm.G_ = n, 2, 1, k, g
    mm.target_names_ = ["y"]
    mm.predictor_names_ = ["intercept", "time"]
    mm.target_type_ = {0: "gaussian"}
    mm.num_binary_targets_ = 0
    mm.ranef_indices_ = np.array([True, True])
    mm.shared_indices_ = np.array([], dtype=int)
    mm.traj_indices_ = np.array([0, 1], dtype=int)
    mm.num_shared_preds_ = 0
    mm.num_traj_preds_ = 2
    mm.w_mu_shared_ = torch.empty((0, 1), dtype=torch.float64)
    mm.w_var_shared_ = torch.empty((0, 1), dtype=torch.float64)
    mm.w_mu0_shared_ = torch.empty((0, 1), dtype=torch.float64)
    mm.w_var0_shared_ = torch.empty((0, 1), dtype=torch.float64)

    mm.w_mu_ = torch.tensor(beta[:, None, :], dtype=torch.float64)
    mm.w_var_ = torch.full((2, 1, k), 0.02, dtype=torch.float64)
    mm.w_mu0_ = torch.zeros((2, 1), dtype=torch.float64)
    mm.w_var0_ = torch.full((2, 1), 10.0, dtype=torch.float64)
    mm.w_mu0_override_ = None
    mm.w_var0_override_ = None
    mm.w_mu_fixed_ = None
    mm.fixed_ids_ = None

    mm.lambda_a_ = torch.full((1, k), 80.0, dtype=torch.float64)
    mm.lambda_b_ = torch.full((1, k), 3.0, dtype=torch.float64)
    mm.lambda_a0_mod_ = torch.tensor([2.0], dtype=torch.float64)
    mm.lambda_b0_mod_ = torch.tensor([0.2], dtype=torch.float64)
    mm.lambda_a0_ = torch.tensor([2.0], dtype=torch.float64)
    mm.lambda_b0_ = torch.tensor([0.2], dtype=torch.float64)

    mm.alpha_ = 1.0
    mm.v_a_ = torch.ones(k, dtype=torch.float64)
    mm.v_b_ = torch.ones(k, dtype=torch.float64)
    mm.R_ = torch.tensor(r, dtype=torch.float64)
    mm.sig_trajs_ = torch.ones(k, dtype=torch.bool)
    mm.prob_thresh_ = 1e-12

    mm.N_to_G_index_map_ = np.repeat(np.arange(g), nvis)
    first = np.zeros(n, dtype=bool)
    first[np.arange(0, n, nvis)] = True
    mm.group_first_index_ = first
    mm.df_ = pd.DataFrame({"id": ids, "intercept": 1.0, "time": time, "y": y})
    mm.df_helper_ = mm.df_[["id"]].copy()
    for kk in range(k):
        mm.df_helper_[f"like_accum_{kk}"] = np.nan
    mm.gb_ = mm.df_helper_.groupby("id")

    mm.Sig0_ = {"y": torch.tensor(dtrue, dtype=torch.float64)}
    mm.u_mu_ = torch.zeros((g, 1, k, 2), dtype=torch.float64)
    mm.u_Sig_ = torch.full((g, 1, k, 2, 2), 1e-20, dtype=torch.float64)

    mm.ranef_factorization_ = "structured"
    mm.ranef_cov_mode_ = "fixed"
    mm.ranef_cov_min_eig_ = 1e-8
    mm.structured_r_damping_ = 1.0
    mm.structured_tol_elbo_rel_ = None
    mm.structured_tol_r_ = 1e-5
    mm.structured_tol_w_ = 1e-5
    mm.structured_tol_lambda_ = 1e-5
    mm.structured_tol_ranef_cov_ = 1e-5
    mm.structured_min_iters_ = 5
    mm.ranef_cov_warmup_iters_ = 0
    mm.inference_history_ = []
    mm.lower_bounds_ = []
    mm.objective_converged_ = None
    mm.parameter_converged_ = False
    mm.converged_ = False
    mm.n_iter_ = 0
    mm.fit_segment_ = 0
    mm._initialize_structured_random_effect_state()
    return mm, cls, dtrue


class TestBackwardCompatibility(unittest.TestCase):
    def test_mean_field_remains_default(self):
        sig = inspect.signature(MultDPRegression.fit)
        self.assertEqual(sig.parameters["ranef_factorization"].default, "mean_field")
        self.assertEqual(sig.parameters["ranef_cov_mode"].default, "fixed")

    def test_historical_get_r_path_when_not_structured(self):
        mm, _, _ = build_manual_model()
        mm.ranef_factorization_ = "mean_field"
        # Historical get_R_matrix should execute without touching the structured
        # classifier. Existing random effects are zero here by construction.
        out = mm.get_R_matrix(df_helper=mm.df_helper_)
        self.assertEqual(tuple(out.shape), (mm.N_, mm.K_))
        np.testing.assert_allclose(out.sum(dim=1).numpy(), 1.0, atol=1e-12)


class TestStructuredCore(unittest.TestCase):
    def test_update_u_is_class_conditional_not_responsibility_weighted(self):
        mm, _, _ = build_manual_model()
        # Make one subject very uncertain; q(b|z=k) must still use full
        # class-conditional likelihood information.
        mm.R_[mm.N_to_G_index_map_ == 0] = torch.tensor([0.01, 0.99], dtype=torch.float64)
        mm.update_u_structured()

        rows = np.where(mm.N_to_G_index_map_ == 0)[0]
        z = mm.X_[rows][:, mm.ranef_indices_]
        y = mm.Y_[rows, 0]
        fixed = mm.X_[rows] @ mm.w_mu_[:, 0, 0]
        resid = y - fixed
        prec = mm.lambda_a_[0, 0] / mm.lambda_b_[0, 0]
        d = mm.Sig0_["y"]
        expected_cov = torch.linalg.inv(torch.linalg.inv(d) + prec * z.T @ z)
        expected_mean = expected_cov @ (prec * z.T @ resid)
        np.testing.assert_allclose(
            mm.u_mu_[0, 0, 0, mm.ranef_indices_].numpy(),
            expected_mean.numpy(), rtol=1e-11, atol=1e-11,
        )
        np.testing.assert_allclose(
            mm.u_Sig_[0, 0, 0][np.ix_(mm.ranef_indices_, mm.ranef_indices_)].numpy(),
            expected_cov.numpy(), rtol=1e-11, atol=1e-11,
        )

    def test_structured_new_data_infers_random_effects(self):
        mm, _, _ = build_manual_model()
        mm.update_v()
        mm.update_u_structured()
        r = mm.get_R_matrix_structured(df=mm.df_, gb_col="id", test_data=True)
        self.assertEqual(tuple(r.shape), (mm.N_, mm.K_))
        np.testing.assert_allclose(r.sum(dim=1).numpy(), 1.0, atol=1e-12)
        # All visits for an individual must have the same assignment vector.
        for i in range(mm.G_):
            rows = np.where(mm.N_to_G_index_map_ == i)[0]
            expected = np.repeat(r[rows[0]].numpy()[None, :], len(rows), axis=0)
            np.testing.assert_allclose(r[rows].numpy(), expected)

    def test_covariance_update_matches_weighted_second_moment(self):
        mm, _, _ = build_manual_model(g=8)
        mm.update_u_structured()
        mm.ranef_cov_mode_ = "estimate"

        active = np.where(mm.sig_trajs_)[0]
        rg = mm._get_group_responsibilities()
        numer = torch.zeros((2, 2), dtype=torch.float64)
        denom = torch.tensor(0.0, dtype=torch.float64)
        for k in active:
            w = rg[:, k]
            m = mm.u_mu_[:, 0, k, :]
            s = mm.u_Sig_[:, 0, k, :, :]
            numer += torch.sum(w[:, None, None] * (s + torch.einsum("gi,gj->gij", m, m)), dim=0)
            denom += torch.sum(w)
        expected = numer / denom
        expected = 0.5 * (expected + expected.T)

        mm.update_ranef_covariance_structured()
        np.testing.assert_allclose(
            mm.ranef_cov_["y"].numpy(), expected.numpy(), rtol=1e-10, atol=1e-10
        )

    def test_structured_elbo_is_finite_and_iteration_records_qc(self):
        mm, _, _ = build_manual_model(g=20)
        mm.structured_tol_elbo_rel_ = None
        mm.fit_coordinate_ascent_structured(2, verbose=False, weights_only=False)
        self.assertEqual(mm.n_iter_, 2)
        self.assertEqual(len(mm.inference_history_), 2)
        self.assertTrue(np.isfinite(mm.inference_history_[-1]["elbo"]))
        self.assertIn("max_delta_ranef_cov", mm.inference_history_[-1])

    def test_estimated_covariance_stays_positive_definite(self):
        mm, _, _ = build_manual_model(g=20)
        mm.ranef_cov_mode_ = "estimate"
        mm.structured_tol_elbo_rel_ = None
        mm.fit_coordinate_ascent_structured(2, verbose=False, weights_only=False)
        eig = torch.linalg.eigvalsh(mm.ranef_cov_["y"])
        self.assertTrue(torch.all(eig >= mm.ranef_cov_min_eig_ * 0.999))

    def test_estimated_covariance_elbo_is_monotone_on_synthetic_data(self):
        mm, _, _ = build_manual_model(g=30)
        mm.ranef_cov_mode_ = "estimate"
        mm.structured_tol_elbo_rel_ = None
        mm.fit_coordinate_ascent_structured(8, verbose=False, weights_only=False)
        elbo = np.array([row["elbo"] for row in mm.inference_history_])
        self.assertTrue(np.all(np.diff(elbo) >= -1e-8), elbo)


if __name__ == "__main__":
    unittest.main()


def test_structured_estimated_cov_verbose_does_not_format_generator(capsys):
    """Regression test for numpy.max(generator) in structured verbose QC."""
    mm, _, _ = build_manual_model(g=12)
    mm.ranef_cov_mode_ = "estimate"
    mm.structured_tol_elbo_rel_ = None
    mm.fit_coordinate_ascent_structured(
        iters=1, verbose=True, weights_only=False
    )
    captured = capsys.readouterr()
    assert "structured ELBO" in captured.out
    assert "dD" in captured.out
    assert len(mm.inference_history_) >= 1
    assert isinstance(
        mm.inference_history_[-1]["max_delta_ranef_cov"], float
    )



def test_structured_responsibilities_do_not_hard_prune_components():
    mm, _, _ = build_manual_model(g=8, k=2)
    mm.prob_thresh_ = 0.49
    # Even with an absurd historical threshold, structured responsibilities
    # remain soft and every row is normalized.
    mm.update_v()
    mm.update_u_structured()
    r = mm._update_z_structured_training()
    assert torch.all(r > 0)
    np.testing.assert_allclose(r.sum(dim=1).numpy(), 1.0, atol=1e-12)


def test_structured_fit_keeps_truncation_components_available():
    mm, _, _ = build_manual_model(g=12, k=2)
    mm.sig_trajs_[1] = False
    mm.fit_coordinate_ascent_structured(1, verbose=False, weights_only=False)
    assert torch.all(mm.sig_trajs_)


def test_structured_occupancy_diagnostics_are_probabilistic():
    mm, _, _ = build_manual_model(g=10, k=2)
    mm.update_v()
    d = mm.get_structured_occupancy_diagnostics(tail_components=1)
    assert d['truncation_k'] == 2
    assert 0.0 <= d['expected_occupied_k'] <= 2.0
    assert 1 <= d['map_occupied_k'] <= 2
    assert len(d['effective_membership']) == 2
    assert 0.0 <= d['residual_stick_mass_beyond_truncation'] <= 1.0


def test_covariance_warmup_defers_updates():
    mm, _, _ = build_manual_model(g=20)
    mm.ranef_cov_mode_ = 'estimate'
    mm.ranef_cov_warmup_iters_ = 2
    start = mm.ranef_cov_['y'].clone()
    mm.fit_coordinate_ascent_structured(2, verbose=False, weights_only=False)
    np.testing.assert_allclose(mm.ranef_cov_['y'].numpy(), start.numpy())
    assert not any(
        row['ranef_cov_update_attempted'] for row in mm.inference_history_)


def _scramble_group_labels(mm):
    """Make row-first group order differ from pandas GroupBy/internal G order."""
    g = mm.G_
    labels = np.array([f"g{g-i:03d}" for i in range(g)], dtype=object)
    row_labels = labels[mm.N_to_G_index_map_]
    mm.df_["id"] = row_labels
    mm.df_helper_["id"] = row_labels
    mm.gb_ = mm.df_helper_.groupby("id")
    mm._set_N_to_G_index_map()
    mm._set_group_first_index(mm.df_, mm.gb_)
    mm._initialize_structured_random_effect_state()
    return mm


def test_group_responsibilities_follow_internal_group_order_not_row_order():
    mm, _, _ = build_manual_model(g=8)
    mm = _scramble_group_labels(mm)

    row_first_group = np.repeat(np.arange(mm.G_), 5)
    p = 0.05 + 0.9 * (row_first_group / max(1, mm.G_ - 1))
    r = np.column_stack([p, 1.0 - p])
    mm.R_ = torch.tensor(r, dtype=torch.float64)

    group_r = mm._get_group_responsibilities()
    reconstructed = group_r[
        torch.as_tensor(mm.N_to_G_index_map_, dtype=torch.long)
    ]
    np.testing.assert_allclose(reconstructed.numpy(), mm.R_.numpy(), atol=0.0)


def test_structured_qz_update_increases_elbo_with_nonmatching_group_orders():
    mm, _, _ = build_manual_model(g=20)
    mm = _scramble_group_labels(mm)
    mm.update_v()
    mm.update_u_structured()

    before = mm.compute_structured_elbo()
    mm.R_ = mm._update_z_structured_training()
    after = mm.compute_structured_elbo()
    assert after >= before - 1e-9, (before, after)


def test_meanfield_update_u_uses_responsibility_for_correct_group():
    mm, _, _ = build_manual_model(g=10)
    mm = _scramble_group_labels(mm)
    mm.ranef_factorization_ = "mean_field"
    mm.G_r_r_ = mm.G_r_r_by_target_[0].clone()
    mm.invSig0_ = {"y": torch.inverse(mm.Sig0_["y"])}

    group_r = torch.zeros((mm.G_, mm.K_), dtype=torch.float64)
    group_r[:, 0] = torch.linspace(0.05, 0.95, mm.G_)
    group_r[:, 1] = 1.0 - group_r[:, 0]
    mm.R_ = group_r[
        torch.as_tensor(mm.N_to_G_index_map_, dtype=torch.long)
    ].clone()

    mm.update_u()

    g = mm.G_ // 2
    k = 0
    rows = np.where(mm.N_to_G_index_map_ == g)[0]
    z = mm.X_[rows][:, mm.ranef_indices_]
    y = mm.Y_[rows, 0]
    fixed = mm.X_[rows] @ mm.w_mu_[:, 0, k]
    resid = y - fixed
    r = group_r[g, k]
    prec = mm.lambda_a_[0, k] / mm.lambda_b_[0, k]
    d = mm.Sig0_["y"]
    expected_cov = torch.linalg.inv(
        torch.linalg.inv(d) + prec * r * (z.T @ z)
    )
    expected_mean = expected_cov @ (prec * r * z.T @ resid)

    np.testing.assert_allclose(
        mm.u_mu_[g, 0, k, mm.ranef_indices_].numpy(),
        expected_mean.numpy(), rtol=1e-10, atol=1e-10
    )



def test_structured_records_objective_and_parameter_convergence_separately():
    mm, _, _ = build_manual_model(g=20)
    mm.structured_tol_elbo_rel_ = 1.0
    mm.structured_tol_r_ = 0.0
    mm.structured_tol_w_ = 0.0
    mm.structured_tol_lambda_ = 0.0
    mm.structured_tol_ranef_cov_ = 0.0
    mm.structured_min_iters_ = 1
    mm.fit_coordinate_ascent_structured(1, verbose=False, weights_only=False)
    assert mm.objective_converged_ is True
    assert mm.parameter_converged_ is False
    assert mm.converged_ is False
    assert mm.inference_history_[-1]['objective_tolerance_met'] is True
    assert mm.inference_history_[-1]['parameter_tolerances_met'] is False


def test_continue_structured_fit_appends_history_and_iteration_numbers():
    mm, _, _ = build_manual_model(g=20)
    mm.fit_coordinate_ascent_structured(2, verbose=False, weights_only=False)
    first_elbo = mm.inference_history_[-1]['elbo']
    assert mm.n_iter_ == 2
    assert mm.fit_segment_ == 0

    mm.continue_structured_fit(
        iters=2, verbose=False, ranef_cov_mode='fixed',
        structured_tol_elbo_rel=1e-12,
        ranef_cov_warmup_iters=0)

    assert mm.n_iter_ == 4
    assert mm.fit_segment_ == 1
    assert len(mm.inference_history_) == 4
    assert [x['iteration'] for x in mm.inference_history_] == [1, 2, 3, 4]
    assert [x['fit_segment'] for x in mm.inference_history_] == [0, 0, 1, 1]
    assert mm.inference_history_[2]['elbo'] >= first_elbo - 1e-8


def test_continue_structured_fit_can_enable_covariance_estimation():
    mm, _, _ = build_manual_model(g=30)
    mm.fit_coordinate_ascent_structured(3, verbose=False, weights_only=False)
    start_cov = mm.ranef_cov_['y'].clone()
    mm.continue_structured_fit(
        iters=2, verbose=False, ranef_cov_mode='estimate',
        ranef_cov_warmup_iters=0)
    assert mm.ranef_cov_mode_ == 'estimate'
    assert any(
        row['ranef_cov_update_attempted']
        for row in mm.inference_history_ if row.get('fit_segment') == 1)
    eig = torch.linalg.eigvalsh(mm.ranef_cov_['y'])
    assert torch.all(eig >= mm.ranef_cov_min_eig_ * 0.999)


def test_external_inference_allows_absent_target_columns():
    mm, _, _ = build_manual_model(g=8)
    mm.update_v()
    df = mm.df_.drop(columns=['y'])
    out = mm.infer_new_data(df, gb_col='id', mode='strict')
    assert out['R'].shape == (mm.N_, mm.K_)
    np.testing.assert_allclose(
        out['R'].sum(dim=1).detach().cpu().numpy(), 1.0, atol=1e-12)


def test_external_inference_returns_posterior_marginal_random_effect_moments():
    mm, _, _ = build_manual_model(g=8)
    mm.update_v()
    out = mm.infer_new_data(mm.df_, gb_col='id', mode='strict')
    assert out['u_mu_marginal'].shape == (mm.G_, mm.D_, mm.M_)
    assert out['u_Sig_marginal'].shape == (mm.G_, mm.D_, mm.M_, mm.M_)

    g = 0
    d = 0
    r = out['R_group'][g]
    mu = out['u_mu'][g, d]
    sig = out['u_Sig'][g, d]
    expected_mu = torch.sum(r[:, None] * mu, dim=0)
    expected_second = torch.sum(
        r[:, None, None] * (
            sig + torch.einsum('ki,kj->kij', mu, mu)), dim=0)
    expected_sig = expected_second - torch.outer(expected_mu, expected_mu)
    np.testing.assert_allclose(
        out['u_mu_marginal'][g, d].numpy(), expected_mu.numpy(),
        rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(
        out['u_Sig_marginal'][g, d].numpy(), expected_sig.numpy(),
        rtol=1e-11, atol=1e-11)


def test_external_inference_reports_local_convergence():
    mm, _, _ = build_manual_model(g=8)
    strict = mm.infer_new_data(mm.df_, gb_col='id', mode='strict')
    assert strict['converged'] is True
    assert strict['iterations'] == 1
    assert strict['max_responsibility_change'] == 0.0

    adapted = mm.infer_new_data(
        mm.df_, gb_col='id', mode='adapt_prevalence', max_iters=100,
        tol=1e-8)
    assert isinstance(adapted['converged'], bool)
    assert adapted['iterations'] >= 1
    assert np.isfinite(adapted['max_responsibility_change'])
    if adapted['converged']:
        assert adapted['max_responsibility_change'] <= 1e-8

    with pytest.raises(ValueError, match='max_iters'):
        mm.infer_new_data(
            mm.df_, gb_col='id', mode='adapt_prevalence', max_iters=0)


def test_external_inference_does_not_mutate_trained_globals():
    mm, _, _ = build_manual_model(g=10)
    mm.update_v()
    before = {
        'w': mm.w_mu_.clone(),
        'lambda_a': mm.lambda_a_.clone(),
        'lambda_b': mm.lambda_b_.clone(),
        'v_a': mm.v_a_.clone(),
        'v_b': mm.v_b_.clone(),
        'D': mm.ranef_cov_['y'].clone(),
    }
    mm.infer_new_data(mm.df_, gb_col='id', mode='strict')
    mm.infer_new_data(
        mm.df_, gb_col='id', mode='adapt_prevalence', max_iters=10)
    np.testing.assert_allclose(mm.w_mu_.numpy(), before['w'].numpy())
    np.testing.assert_allclose(mm.lambda_a_.numpy(), before['lambda_a'].numpy())
    np.testing.assert_allclose(mm.lambda_b_.numpy(), before['lambda_b'].numpy())
    np.testing.assert_allclose(mm.v_a_.numpy(), before['v_a'].numpy())
    np.testing.assert_allclose(mm.v_b_.numpy(), before['v_b'].numpy())
    np.testing.assert_allclose(mm.ranef_cov_['y'].numpy(), before['D'].numpy())


def test_structured_reportable_ids_use_posterior_occupancy_not_sig_trajs():
    mm, _, _ = build_manual_model(g=20)
    mm.R_[:, 0] = 0.99
    mm.R_[:, 1] = 0.01
    mm.sig_trajs_[:] = True
    ids = mm.get_reportable_trajectory_ids(min_effective_membership=1.0)
    np.testing.assert_array_equal(ids, np.array([0]))


def test_staged_covariance_mode_releases_D_after_fixed_objective_convergence():
    mm, _, _ = build_manual_model(g=20)
    start = mm.ranef_cov_['y'].clone()
    mm.continue_structured_fit(
        iters=4, verbose=False, ranef_cov_mode='staged',
        structured_tol_elbo_rel=1.0, structured_min_iters=1,
        ranef_cov_warmup_iters=0)
    assert mm.ranef_cov_release_iteration_ is not None
    assert mm.ranef_cov_stage_ == 'estimate'
    assert mm.fit_complete_ is True
    diag = mm.staged_covariance_diagnostics_
    assert diag['release_iteration'] == mm.ranef_cov_release_iteration_
    assert np.isfinite(diag['fixed_phase_final_elbo'])
    assert np.isfinite(diag['final_elbo'])
    assert diag['final_stage'] == 'estimate'
    assert diag['fit_complete'] is True
    np.testing.assert_allclose(
        np.asarray(diag['initial_covariance']['y']), start.numpy())

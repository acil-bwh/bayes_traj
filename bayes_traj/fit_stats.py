"""Fit and classification diagnostics for :mod:`bayes_traj` models.

The historical package mixed row-level and subject/group-level quantities in a
few diagnostics.  Longitudinal trajectory membership is a group-level latent
variable, so assignment diagnostics in this module now operate on one posterior
responsibility vector per group.

For Gaussian structured models the module also provides a *plug-in marginal*
likelihood that analytically integrates the fitted Gaussian random effects for
each subject and trajectory while holding global parameters at their posterior
means / expected precisions.  This is intentionally distinct from a full
Bayesian marginal likelihood.  AIC/BIC/SABIC/ICL values based on this quantity
are useful model summaries, but should not be described as exact Bayes factors
or as directly identical to likelihood criteria from another software package.
"""

import numpy as np
import torch
from scipy.special import logsumexp


# ---------------------------------------------------------------------------
# Common helpers
# ---------------------------------------------------------------------------

def _to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def reportable_trajectory_ids(mm, min_effective_membership=1.0):
    """Return trajectory IDs appropriate for summaries/diagnostics.

    Structured DP fits keep every truncation component alive, so
    ``sig_trajs_`` is not an occupancy indicator there.  Prefer the model's
    posterior-occupancy helper when available; retain historical ``sig_trajs_``
    behavior for old mean-field models.
    """
    if hasattr(mm, 'get_reportable_trajectory_ids'):
        return np.asarray(
            mm.get_reportable_trajectory_ids(min_effective_membership),
            dtype=int)
    sig = _to_numpy(mm.sig_trajs_).astype(bool)
    return np.where(sig)[0].astype(int)


def group_responsibilities(mm):
    """Return a ``G x K`` responsibility matrix in the model's group order.

    Newer models expose an ordered helper.  Older/synthetic model objects used
    by the package tests may carry only ``group_first_index_``; keep a robust
    fallback for those objects while preferring the corrected group-order path.
    """
    if hasattr(mm, '_get_group_responsibilities'):
        try:
            return _to_numpy(mm._get_group_responsibilities()).astype(float)
        except (AttributeError, TypeError, ValueError):
            pass

    r = _to_numpy(mm.R_).astype(float)
    gb = getattr(mm, 'gb_', None)
    if gb is not None:
        rows = [np.asarray(vv)[0] for _, vv in gb.groups.items()]
        return r[np.asarray(rows, dtype=int), :]
    first = getattr(mm, 'group_first_index_', None)
    if first is not None:
        first = np.asarray(first)
        if first.dtype == bool and first.shape[0] == r.shape[0]:
            return r[first, :]
    return r


def _subset_group_responsibilities(mm, traj_ids=None, renormalize=True):
    rg = group_responsibilities(mm)
    if traj_ids is None:
        traj_ids = reportable_trajectory_ids(mm)
    traj_ids = np.asarray(traj_ids, dtype=int)
    if traj_ids.size == 0:
        raise ValueError('No reportable trajectories')
    p = rg[:, traj_ids].astype(float, copy=True)
    if renormalize:
        denom = np.sum(p, axis=1, keepdims=True)
        if np.any(denom <= 0) or np.any(~np.isfinite(denom)):
            raise ValueError('Invalid group responsibility normalization')
        p /= denom
    return p, traj_ids


# ---------------------------------------------------------------------------
# Classification diagnostics
# ---------------------------------------------------------------------------

def ave_pp(mm, traj_ids=None):
    """Average posterior probability among subjects MAP-assigned to each class.

    This is the Nagin average posterior probability (AvePP), computed at the
    subject/group level rather than the observation-row level.
    """
    rg = group_responsibilities(mm)
    if traj_ids is None:
        traj_ids = reportable_trajectory_ids(mm)
    traj_ids = np.asarray(traj_ids, dtype=int)
    map_ids = np.argmax(rg, axis=1)
    out = {}
    for t in traj_ids:
        ids = map_ids == t
        out[int(t)] = float(np.mean(rg[ids, t])) if np.any(ids) else np.nan
    return out


def odds_correct_classification(mm, traj_ids=None):
    """Odds of correct classification (OCC) for each reportable trajectory."""
    rg = group_responsibilities(mm)
    if traj_ids is None:
        traj_ids = reportable_trajectory_ids(mm)
    traj_ids = np.asarray(traj_ids, dtype=int)
    ave = ave_pp(mm, traj_ids=traj_ids)
    pis = np.mean(rg, axis=0)
    out = {}
    eps = np.finfo(float).eps
    for t in traj_ids:
        a = ave[int(t)]
        p = float(pis[t])
        if not np.isfinite(a):
            out[int(t)] = np.nan
            continue
        a = np.clip(a, eps, 1.0 - eps)
        p = np.clip(p, eps, 1.0 - eps)
        out[int(t)] = float((a / (1.0 - a)) / (p / (1.0 - p)))
    return out


def prob_prop(mm, traj_ids=None):
    """MAP class proportion versus posterior mean class probability."""
    rg = group_responsibilities(mm)
    if traj_ids is None:
        traj_ids = reportable_trajectory_ids(mm)
    traj_ids = np.asarray(traj_ids, dtype=int)
    map_ids = np.argmax(rg, axis=1)
    probs = np.mean(rg, axis=0)
    out = {}
    g = float(rg.shape[0])
    for t in traj_ids:
        out[int(t)] = (float(np.sum(map_ids == t) / g), float(probs[t]))
    return out


def assignment_diagnostics(mm, traj_ids=None, thresholds=(0.7, 0.8, 0.9)):
    """Return global posterior-assignment diagnostics.

    Probabilities are renormalized over the reportable trajectory set for the
    entropy/max-probability summaries.  Raw effective memberships remain based
    on the original full responsibility matrix.
    """
    p, traj_ids = _subset_group_responsibilities(
        mm, traj_ids=traj_ids, renormalize=True)
    rg_full = group_responsibilities(mm)
    raw_selected = rg_full[:, traj_ids]
    maxp = np.max(p, axis=1)
    eps = np.finfo(float).tiny
    raw_entropy = float(-np.sum(p * np.log(np.clip(p, eps, 1.0))))
    k = len(traj_ids)
    if k > 1:
        entropy = float(1.0 - raw_entropy / (p.shape[0] * np.log(k)))
    else:
        entropy = 1.0
    map_local = np.argmax(p, axis=1)
    map_counts = np.bincount(map_local, minlength=k)

    out = {
        'num_groups': int(p.shape[0]),
        'num_reportable_trajectories': int(k),
        'trajectory_ids': traj_ids.astype(int).tolist(),
        'entropy': entropy,
        'classification_entropy_raw': raw_entropy,
        'mean_max_posterior': float(np.mean(maxp)),
        'median_max_posterior': float(np.median(maxp)),
        'min_max_posterior': float(np.min(maxp)),
        'map_counts': {int(t): int(map_counts[j]) for j, t in enumerate(traj_ids)},
        'effective_membership': {
            int(t): float(np.sum(raw_selected[:, j]))
            for j, t in enumerate(traj_ids)
        },
    }
    for thr in thresholds:
        out[f'proportion_max_posterior_ge_{thr:g}'] = float(np.mean(maxp >= thr))
    return out


# ---------------------------------------------------------------------------
# Gaussian plug-in marginal likelihood and information criteria
# ---------------------------------------------------------------------------

def _expected_stick_weights(mm):
    va = _to_numpy(mm.v_a_).astype(float)
    vb = _to_numpy(mm.v_b_).astype(float)
    ev = va / (va + vb)
    e1m = vb / (va + vb)
    out = np.zeros_like(ev)
    remaining = 1.0
    for k in range(len(ev)):
        out[k] = remaining * ev[k]
        remaining *= e1m[k]
    return out, float(remaining)


def _group_rows(mm):
    """Observation-row indices for each group in the model's internal G order."""
    mapping = getattr(mm, 'N_to_G_index_map_', None)
    if mapping is not None and len(mapping) == int(mm.N_):
        mapping = np.asarray(mapping, dtype=int)
        g = int(getattr(mm, 'G_', np.max(mapping) + 1))
        return [np.where(mapping == ii)[0] for ii in range(g)]
    if getattr(mm, 'gb_', None) is None:
        return [np.asarray([i], dtype=int) for i in range(int(mm.N_))]
    try:
        return [np.asarray(vv, dtype=int)
                for _, vv in mm.gb_.groups.items()]
    except AttributeError:
        key = getattr(mm.gb_, 'keys', None)
        df = getattr(mm, 'df_', None)
        if isinstance(key, str) and df is not None and key in df.columns:
            gb = df[[key]].groupby(key)
            return [np.asarray(vv, dtype=int)
                    for _, vv in gb.groups.items()]
        raise


def _random_effect_covariance(mm, target_name):
    if getattr(mm, 'ranef_indices_', None) is None or \
       np.sum(np.asarray(mm.ranef_indices_, dtype=bool)) == 0:
        return None
    if hasattr(mm, '_get_structured_ranef_cov') and \
       getattr(mm, 'ranef_factorization_', 'mean_field') == 'structured':
        return _to_numpy(mm._get_structured_ranef_cov(target_name)).astype(float)
    sig0 = getattr(mm, 'Sig0_', None)
    if sig0 is None:
        return None
    return _to_numpy(sig0[target_name]).astype(float)


def gaussian_marginal_log_likelihood(mm, traj_ids=None, return_group_values=False):
    """Plug-in subject-level marginal log likelihood for Gaussian models.

    Random effects are analytically integrated using the model covariance ``D``.
    Fixed effects, residual precisions, and mixture weights are held at posterior
    means / expectations.  Posterior uncertainty in these global parameters is
    not integrated, hence the term *plug-in marginal likelihood*.
    """
    target_types = [mm.target_type_[d] for d in range(mm.D_)]
    if any(tt != 'gaussian' for tt in target_types):
        raise NotImplementedError(
            'Gaussian plug-in marginal likelihood currently requires all targets '
            'to be Gaussian')

    if traj_ids is None:
        traj_ids = reportable_trajectory_ids(mm)
    traj_ids = np.asarray(traj_ids, dtype=int)
    if traj_ids.size == 0:
        raise ValueError('No reportable trajectories')

    X = _to_numpy(mm.X_).astype(float)
    Y = _to_numpy(mm.Y_).astype(float)
    weights, residual_tail = _expected_stick_weights(mm)
    weights = np.asarray(weights[traj_ids], dtype=float)
    if np.sum(weights) <= 0:
        raise ValueError('Selected expected stick weights sum to zero')
    weights /= np.sum(weights)
    log_weights = np.log(np.clip(weights, np.finfo(float).tiny, 1.0))

    ranef_mask = None
    if getattr(mm, 'ranef_indices_', None) is not None:
        ranef_mask = np.asarray(mm.ranef_indices_, dtype=bool)
        if not np.any(ranef_mask):
            ranef_mask = None

    per_group = []
    for rows in _group_rows(mm):
        class_ll = np.asarray(log_weights, dtype=float).copy()
        for local_k, k in enumerate(traj_ids):
            llk = 0.0
            for d, target in enumerate(mm.target_names_):
                obs_local = ~np.isnan(Y[rows, d])
                if not np.any(obs_local):
                    continue
                row_obs = rows[obs_local]
                x = X[row_obs, :]
                y = Y[row_obs, d]

                xt = torch.as_tensor(x, dtype=torch.float64)
                shared = _to_numpy(mm._get_gaussian_shared_mean(xt, d)).reshape(-1)
                traj = _to_numpy(mm._get_gaussian_traj_mean(xt, d)[:, int(k)]).reshape(-1)
                mu = shared + traj

                precision = float(_to_numpy(mm.lambda_a_[d, k] / mm.lambda_b_[d, k]))
                if precision <= 0 or not np.isfinite(precision):
                    raise ValueError('Invalid residual precision')
                resid_var = 1.0 / precision
                v = resid_var * np.eye(len(row_obs), dtype=float)

                if ranef_mask is not None:
                    D = _random_effect_covariance(mm, target)
                    if D is not None:
                        z = x[:, ranef_mask]
                        v = v + z @ D @ z.T

                # Cholesky log density for numerical stability.
                chol = np.linalg.cholesky(v)
                resid = y - mu
                solved = np.linalg.solve(chol, resid)
                logdet = 2.0 * np.sum(np.log(np.diag(chol)))
                llk += -0.5 * (
                    len(y) * np.log(2.0 * np.pi) + logdet + np.dot(solved, solved))
            class_ll[local_k] += llk
        per_group.append(float(logsumexp(class_ll)))

    total = float(np.sum(per_group))
    if return_group_values:
        return total, np.asarray(per_group, dtype=float), residual_tail
    return total


def global_parameter_count(mm, traj_ids=None):
    """Count plug-in *model* parameters for Gaussian IC summaries.

    Variational posterior variances and subject-specific random effects are not
    counted as free model parameters.  Estimated population random-effect
    covariance elements are counted only when ``D`` was learned by the model.
    """
    target_types = [mm.target_type_[d] for d in range(mm.D_)]
    if any(tt != 'gaussian' for tt in target_types):
        raise NotImplementedError(
            'Global parameter count currently supports all-Gaussian models only')
    if traj_ids is None:
        traj_ids = reportable_trajectory_ids(mm)
    traj_ids = np.asarray(traj_ids, dtype=int)
    k = int(len(traj_ids))
    if k < 1:
        raise ValueError('No reportable trajectories')

    if hasattr(mm, 'traj_indices_'):
        traj_pred_ids = np.asarray(mm.traj_indices_, dtype=int)
    else:
        traj_pred_ids = np.arange(mm.M_, dtype=int)
    if hasattr(mm, 'shared_indices_'):
        shared_pred_ids = np.asarray(mm.shared_indices_, dtype=int)
    else:
        shared_pred_ids = np.asarray([], dtype=int)

    n_coef = k * len(traj_pred_ids) * mm.D_ + len(shared_pred_ids) * mm.D_

    # Do not count coefficients that were explicitly fixed by the user.
    fixed_mask = getattr(mm, 'fixed_ids_', None)
    if fixed_mask is not None:
        fm = _to_numpy(fixed_mask).astype(bool)
        fixed_count = 0
        for d in range(mm.D_):
            fixed_count += int(np.sum(fm[traj_pred_ids, d, :][:, traj_ids]))
        n_coef -= fixed_count

    n_resid = k * mm.D_
    n_weights = max(0, k - 1)
    n_D = 0
    ranef_mask = getattr(mm, 'ranef_indices_', None)
    if ranef_mask is not None:
        q = int(np.sum(np.asarray(ranef_mask, dtype=bool)))
        strategy = getattr(mm, 'ranef_cov_strategy_',
                           getattr(mm, 'ranef_cov_mode_', 'fixed'))
        if q > 0 and strategy in ('estimate', 'staged'):
            n_D = mm.D_ * q * (q + 1) // 2

    return int(n_coef + n_resid + n_weights + n_D)


def gaussian_information_criteria(mm, traj_ids=None):
    """Return Gaussian plug-in AIC/BIC/SABIC/ICL-style summaries.

    ``BIC`` and ``SABIC`` use the number of independent groups/subjects as the
    sample size.  The sample-size adjusted BIC uses ``log((n + 2) / 24)``, the
    convention used by the mpjlcmm comparison in this project.

    ``ICL1`` uses the MAP-posterior penalty ``-2 sum(log(max_k p_ik))`` and
    ``ICL2`` uses the classification entropy penalty ``-sum p_ik log p_ik``.
    """
    if traj_ids is None:
        traj_ids = reportable_trajectory_ids(mm)
    traj_ids = np.asarray(traj_ids, dtype=int)
    ll, group_ll, residual_tail = gaussian_marginal_log_likelihood(
        mm, traj_ids=traj_ids, return_group_values=True)
    p = global_parameter_count(mm, traj_ids=traj_ids)
    n = len(group_ll)
    aic = -2.0 * ll + 2.0 * p
    bic = -2.0 * ll + p * np.log(n)
    sabic = -2.0 * ll + p * np.log((n + 2.0) / 24.0)

    probs, _ = _subset_group_responsibilities(
        mm, traj_ids=traj_ids, renormalize=True)
    eps = np.finfo(float).tiny
    maxp = np.max(probs, axis=1)
    entropy_raw = float(-np.sum(probs * np.log(np.clip(probs, eps, 1.0))))
    icl1 = bic - 2.0 * float(np.sum(np.log(np.clip(maxp, eps, 1.0))))
    icl2 = bic + entropy_raw
    classification = assignment_diagnostics(mm, traj_ids=traj_ids)

    return {
        'log_likelihood': float(ll),
        'num_parameters': int(p),
        'num_groups': int(n),
        'aic': float(aic),
        'bic': float(bic),
        'sabic': float(sabic),
        'icl1': float(icl1),
        'icl2': float(icl2),
        'entropy': float(classification['entropy']),
        'classification_entropy_raw': entropy_raw,
        'residual_expected_stick_mass_beyond_truncation': float(residual_tail),
        'trajectory_ids': traj_ids.astype(int).tolist(),
    }


# ---------------------------------------------------------------------------
# Historical WAIC wrapper
# ---------------------------------------------------------------------------

def compute_waic2(mm):
    """Call the model's historical WAIC2 implementation.

    WAIC remains available for backward compatibility and diagnostics.  It is
    not used for selecting structured-DP restarts.
    """
    return mm.compute_waic2()

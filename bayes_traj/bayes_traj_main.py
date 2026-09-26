#!/usr/bin/env python

from argparse import ArgumentParser
import pandas as pd
import numpy as np
import pdb
from bayes_traj.mult_dp_regression import MultDPRegression
from bayes_traj.mult_pyro import MultPyro
from bayes_traj.prior_from_model import prior_from_model
from bayes_traj.utils import *
from bayes_traj.fit_stats import compute_waic2
import torch
import pyro
from bayes_traj.pyro_helper import *
from provenance_tools.provenance_tracker import write_provenance_data
import pickle, sys, warnings, os, json
from pathlib import Path
from contextlib import contextmanager

torch.set_default_dtype(torch.double) # TODO -- may not be desirable to set this globally


def _find_git_checkout_for_provenance():
    """Find the git checkout containing this bayes_traj source file."""
    here = Path(__file__).resolve()
    for parent in [here.parent] + list(here.parents):
        if (parent / '.git').exists():
            return parent
    return None


@contextmanager
def _provenance_working_directory():
    """Run provenance_tools from the source checkout, not output cwd."""
    old = Path.cwd()
    repo = _find_git_checkout_for_provenance()
    try:
        if repo is not None:
            os.chdir(repo)
        yield
    finally:
        os.chdir(old)


def _write_provenance(output_path, op):
    with _provenance_working_directory():
        write_provenance_data(output_path, generator_args=op)


def _tensor_all_finite(x):
    if x is None:
        return True
    if torch.is_tensor(x):
        return bool(torch.all(torch.isfinite(x)))
    try:
        return bool(np.all(np.isfinite(np.asarray(x, dtype=float))))
    except Exception:
        return False


def _material_negative_elbo_steps(history_rows, rel_tol=1e-10):
    """Count ELBO decreases larger than a scale-aware numerical tolerance."""
    count = 0
    worst = 0.0
    max_tol = 0.0
    for row in history_rows:
        delta = row.get('delta_elbo', np.nan)
        current = row.get('elbo', np.nan)
        if not np.isfinite(delta) or not np.isfinite(current):
            continue
        previous = float(current) - float(delta)
        tol = float(rel_tol) * max(1.0, abs(previous))
        max_tol = max(max_tol, tol)
        if float(delta) < -tol:
            count += 1
            worst = min(worst, float(delta))
    return count, worst, max_tol


def _structured_repeat_eligibility(mm, final_elbo, negative_steps):
    """Return (eligible, reasons) for structured-repeat model selection.

    Eligibility deliberately uses objective/QC validity rather than the much
    stricter parameter-change tolerances. Only eligible repeats compete on
    final ELBO.
    """
    reasons = []
    if not np.isfinite(final_elbo):
        reasons.append('nonfinite_final_elbo')
    if getattr(mm, 'objective_converged_', None) is not True:
        reasons.append('objective_not_converged')
    if not bool(getattr(mm, 'fit_complete_', False)):
        reasons.append('fit_not_complete')
    if int(negative_steps) > 0:
        reasons.append('negative_elbo_step')

    # Responsibilities must be finite, non-negative, and normalized.
    try:
        rg = mm._get_group_responsibilities().double()
        if torch.any(~torch.isfinite(rg)):
            reasons.append('nonfinite_responsibilities')
        if torch.any(rg < -1e-12):
            reasons.append('negative_responsibility')
        row_sum = torch.sum(rg, dim=1)
        if torch.max(torch.abs(row_sum - 1.0)).item() > 1e-8:
            reasons.append('responsibilities_not_normalized')
    except Exception:
        reasons.append('responsibility_qc_failed')

    for name in ('w_mu_', 'w_var_', 'lambda_a_', 'lambda_b_', 'v_a_', 'v_b_'):
        if not _tensor_all_finite(getattr(mm, name, None)):
            reasons.append(f'nonfinite_{name.rstrip("_")}')
    try:
        if torch.any(mm.w_var_ < 0):
            reasons.append('negative_w_variance')
        if torch.any(mm.lambda_a_ <= 0) or torch.any(mm.lambda_b_ <= 0):
            reasons.append('invalid_precision_parameters')
        if torch.any(mm.v_a_ <= 0) or torch.any(mm.v_b_ <= 0):
            reasons.append('invalid_stick_parameters')
    except Exception:
        reasons.append('global_parameter_qc_failed')

    strategy = getattr(mm, 'ranef_cov_strategy_',
                       getattr(mm, 'ranef_cov_mode_', 'fixed'))
    if strategy == 'staged':
        if getattr(mm, 'ranef_cov_release_iteration_', None) is None:
            reasons.append('staged_D_not_released')
        if getattr(mm, 'ranef_cov_stage_', None) != 'estimate':
            reasons.append('staged_D_not_in_final_phase')
    if strategy in ('estimate', 'staged'):
        try:
            for tt in mm.target_names_:
                cov = mm._get_structured_ranef_cov(tt)
                if torch.any(~torch.isfinite(cov)):
                    reasons.append(f'nonfinite_D_{tt}')
                    continue
                torch.linalg.cholesky(cov)
        except Exception:
            reasons.append('random_effect_covariance_not_positive_definite')

    # Keep order stable while removing duplicates.
    reasons = list(dict.fromkeys(reasons))
    return len(reasons) == 0, reasons


def _pad_axis(arr, new_k, axis, fill_value):
    if arr is None:
        return None
    is_torch = torch.is_tensor(arr)
    a = arr.detach().cpu().numpy() if is_torch else np.asarray(arr)
    old_k = a.shape[axis]
    if old_k >= new_k:
        return arr
    shape = list(a.shape)
    shape[axis] = new_k - old_k
    pad = np.full(shape, fill_value, dtype=float)
    out = np.concatenate([a.astype(float, copy=False), pad], axis=axis)
    return torch.tensor(out, dtype=torch.float64) if is_torch else out


def _expand_structured_prior_to_truncation(prior_data, old_k, new_k):
    """Pad trajectory-indexed initialization arrays for structured DP VI.

    Existing prior-informed components are retained. Added truncation slots get
    neutral stick parameters and otherwise uninformative/randomizable values.
    They are initialization slots, not pre-declared occupied trajectories.
    """
    if new_k <= old_k:
        return prior_data
    prior_data['v_a'] = _pad_axis(prior_data['v_a'], new_k, 0, 1.0)
    prior_data['v_b'] = _pad_axis(
        prior_data['v_b'], new_k, 0, float(prior_data['alpha']))
    prior_data['w_mu'] = _pad_axis(prior_data['w_mu'], new_k, 2, np.nan)
    prior_data['w_var'] = _pad_axis(prior_data['w_var'], new_k, 2, np.nan)
    prior_data['lambda_a'] = _pad_axis(
        prior_data['lambda_a'], new_k, 1, np.nan)
    prior_data['lambda_b'] = _pad_axis(
        prior_data['lambda_b'], new_k, 1, np.nan)
    prior_data['traj_probs'] = _pad_axis(
        prior_data['traj_probs'], new_k, 0, 0.0)
    prior_data['R'] = _pad_axis(prior_data['R'], new_k, 1, 0.0)
    return prior_data


def main():
    """
    """
    np.set_printoptions(precision = 1, suppress = True, threshold=1e6,
                        linewidth=300)

    desc = """Runs Bayesian trajectory analysis on the specified data file \
    with the specified predictors and target variables"""
    
    parser = ArgumentParser(description=desc)
    parser.add_argument('--in_csv', help='Input csv file containing data on \
        which to run Bayesian trajectory analysis', metavar='<string>',
        required=True)
    parser.add_argument('--targets', help='Comma-separated list of target \
        names. Must appear as column names of the input data file.',
        dest='targets', metavar='<string>', required=True)
    parser.add_argument('--groupby', help='Column name in input data file \
        indicating those data instances that must be in the same trajectory. \
        This is typically a subject identifier (e.g. in the case of a \
        longitudinal data set).', dest='groupby', metavar='<string>',
        default=None)
    parser.add_argument('--out_csv', help='If specified, an output csv file \
        will be generated that contains the contents of the input csv file, \
        but with additional columns indicating trajectory assignment \
        information for each data instance. There will be a column called traj \
        with an integer value indicating the most probable trajectory \
        assignment. There will also be columns prefixed with traj_ and then a \
        trajectory-identifying integer. The values of these columns indicate \
        the probability that the data instance belongs to each of the \
        corresponding trajectories.', dest='out_csv', metavar='<string>',
        type=str, default=None)
    parser.add_argument('--prior', help='Input pickle file containing prior \
        settings', metavar='<string>', required=True)
    parser.add_argument('--prec_prior_weight', help='Positive, floating point \
        value indicating how much weight to put on the prior over the residual \
        precisions. Values greater than 1 give more weight to the prior. \
        Values less than one give less weight to the prior.', metavar='<float>',
        type=float, default=1.0)    
    parser.add_argument('--alpha', help='If specified, over-rides the value in \
        the prior file', dest='alpha', metavar=float, default=None)
    parser.add_argument('--out_model', help='Pickle file name. If specified, \
        the model object will be written to this file.', dest='out_model',
        metavar='<string>', default=None, required=False)
    parser.add_argument('--iters', help='Maximum inference iterations. In '
        'structured staged covariance mode this is a per-phase budget: up to '
        'this many fixed-D iterations, then up to this many estimated-D '
        'iterations after release.', dest='iters', metavar='<int>', default=100)
    parser.add_argument('--repeats', help='Number of repeats to attempt. In '
        'structured mode, repeats of the same truncated-DP specification are '
        'ranked by final ELBO (higher is better). Historical mean-field mode '
        'retains WAIC2-based repeat selection.', type=int, metavar='<int>', default=1)
    parser.add_argument('-k', help='Truncation ceiling for the assignment '
        'matrix. In structured DP mode this is a computational ceiling, not '
        'an enforced number of occupied trajectories. If a prior contains '
        'fewer components, it is padded up to this ceiling.', metavar='<int>', default=30)
    parser.add_argument('--prob_thresh', help='Historical mean-field '
        'probability-pruning threshold. Structured DP mode does not hard-prune '
        'responsibilities and ignores this threshold during optimization.',
        metavar='<float>', type=float, default=0.001)
    parser.add_argument('--num_init_trajs', help='Initialization hint. In '
        'structured DP mode this affects only the starting responsibilities; '
        'all truncated components remain eligible to gain posterior mass.',
        metavar='<int>', type=int, default=None)
    parser.add_argument('--num_fin_trajs', help='Legacy mean-field save '
        'filter. Ignored in structured DP mode, where posterior occupancy is '
        'allowed to be data-driven.', metavar='<int>', type=int, default=None)
    parser.add_argument('--waic2_thresh', help='Model will only be written to \
        file provided that the WAIC2 value is below this threshold',
        dest='waic2_thresh', metavar='<float>', type=float,
        default=sys.float_info.max)
#    parser.add_argument('--bic_thresh', help='Model will only be written to \
#        file provided that BIC values are above this threshold',
#        dest='bic_thresh', metavar='<float>', type=float,
#        default=-sys.float_info.max)
#    parser.add_argument("--save_all", help="By default, only the model with the \
#        highest BIC scores is saved to file. However, if this flag is set a model \
#        file is saved for each repeat. The specified output file name is used \
#        with a 'repeat[n]' appended, where [n] indicates the repeat number.",
#        action="store_true")
    parser.add_argument("--verbose", help="Display per-trajectory counts \
        during optimization", action="store_true")
    parser.add_argument('--probs_weight', help='Value between 0 and 1 that \
        controls how much weight to assign to traj_probs, the marginal \
        probability of observing each trajectory. This value is only meaningful \
        if traj_probs has been set in the input prior file. Otherwise, it has no \
        effect. Higher values place more weight on the model-derived probabilities \
        and reflect a stronger belief in those assignment probabilities.',
        dest='probs_weight', metavar='<float>', type=float, default=None)
    parser.add_argument('--weights_only', help='Setting this flag will force \
        the fitting routine to only optimize the trajectory weights. The \
        assumption is that the specified prior file contains previously \
        modeled trajectory information, and that those trajectories should be \
        used for the current fit. This option can be useful if a model \
        learned from one cohort is applied to another cohort, where it is \
        possible that the relative proportions of different trajectory \
        subgroups differs. By using this flag, the proportions of previously \
        determined trajectory subgroups will be determined for the current \
        data set.', action='store_true')
    parser.add_argument('--fix', help='Fix the value of a predictor \
        coefficient for a specified trajectory. During inference, this value \
        for the specified trajectory will remained fixed at this value. \
        Specify as a comma-separated tuple: target_name,predictor_name,\
        trajectory,value. Multiple can be specified. For the purposes of \
        smapling, computing information criteria scores, etc, the \
        corresponding variance for this predictor will be set to the smallest \
        positive floating point value (float64).', type=str, default=None,
        action='append', nargs='+')
    parser.add_argument('--soft_fix', help='Similar to --fix. However, with \
        this flag the user specifies both a mean value and a corresponding \
        standard deviation indicating the confidence around the mean. \
        During inference, the mean and standard deviation replace the default \
        prior settings for the specified coefficient, but only for the \
        indicated trajectory. Specify as a comma-separated tuple: target_name,\
        predictor_name,trajectory,mean,stdev. Multiple can be specified.',
        type=str, default=None, action='append', nargs='+')        
    parser.add_argument('-s', help='Number of samples to use when computing \
        WAIC2', type=int, default=100)
    parser.add_argument('--seed', help='Seed to use for WAIC2 \
        sampling', type=int, default=None)
    parser.add_argument('--fit_seed', help='Optional base random seed for model '
        'initialization. Repeat r uses fit_seed+r. The seed is recorded in the '
        'repeat summary.', type=int, default=None)
    parser.add_argument('--ranef_factorization',
        choices=['mean_field', 'structured'], default='mean_field',
        help='Random-effect VI factorization. mean_field preserves the legacy '
             'implementation for backward compatibility; structured is the '
             'corrected/recommended q(z_i)q(b_i|z_i) implementation.')
    parser.add_argument('--ranef_cov_mode', choices=['fixed', 'estimate', 'staged'],
        default='fixed', help='Structured random-effect covariance strategy: fixed; '
        'estimate from the start; or staged (fit with D fixed until objective '
        'convergence, then release D and continue).')
    parser.add_argument('--ranef_cov_min_eig', type=float, default=1e-8,
        help='Minimum eigenvalue for estimated random-effect covariance.')
    parser.add_argument('--structured_tol_elbo_rel', type=float, default=1e-7,
        help='Relative ELBO convergence tolerance in structured mode. '
        'Objective convergence is the primary fit-completion criterion.')
    parser.add_argument('--structured_tol_r', type=float, default=1e-5,
        help='Structured-mode max responsibility-change tolerance.')
    parser.add_argument('--structured_tol_w', type=float, default=1e-5,
        help='Structured-mode max coefficient-mean change tolerance.')
    parser.add_argument('--structured_tol_lambda', type=float, default=1e-5,
        help='Structured-mode max expected-precision change tolerance.')
    parser.add_argument('--structured_tol_ranef_cov', type=float, default=1e-5,
        help='Structured-mode max random-effect covariance change tolerance.')
    parser.add_argument('--structured_min_iters', type=int, default=5,
        help='Minimum structured iterations before convergence can be declared.')
    parser.add_argument('--structured_r_damping', type=float, default=1.0,
        help='Structured responsibility damping in (0,1].')
    parser.add_argument('--ranef_cov_warmup_iters', type=int, default=10,
        help='Iterations to hold the supplied random-effect covariance fixed '
        'before estimated-covariance updates begin in structured mode.')
    parser.add_argument('--structured_compute_waic', action='store_true',
        help='Also compute WAIC2 for each structured repeat as a diagnostic. '
        'WAIC2 is not used to select the best structured repeat.')
    parser.add_argument('--repeat_summary', default=None,
        help='Optional CSV path for per-repeat fit/occupancy diagnostics. If '
        'omitted and --out_model is supplied, defaults to '
        '<out_model>.repeat_summary.csv in structured mode.')
    parser.add_argument('--resume_model', default=None,
        help='Continue a previously saved structured MultDPRegression pickle '
        'instead of reinitializing from the prior. --iters is interpreted as '
        'additional iterations (per phase if a new staged covariance strategy '
        'is requested). Only one repeat is allowed when resuming.')
    parser.add_argument('--resume_reset_history', action='store_true',
        help='When resuming a structured model, discard prior inference '
        'history and restart iteration numbering at zero. Model parameters '
        'are still continued from the saved state.')
#    parser.add_argument('--use_pyro', help='Use Pyro for inference',
#        action='store_true')
    
    op = parser.parse_args()
    iters = int(op.iters)
    repeats = int(op.repeats)
    targets = op.targets.split(',')
    in_csv = op.in_csv
    prior = op.prior
    out_model = op.out_model
    probs_weight = None #op.probs_weight

    assert op.prec_prior_weight > 0, "prec_prior_weight must be greater than 0"
    
    if probs_weight is not None:
        assert probs_weight >=0 and probs_weight <= 1, \
            "Invalide probs_weight value"
    
    #---------------------------------------------------------------------------
    # Get priors from file
    #---------------------------------------------------------------------------
    print("Reading prior...")
    with open(prior, 'rb') as f:
        prior_file_info = pickle.load(f)

        preds_traj = get_pred_names_from_prior_info(prior_file_info)

        shared_predictors = prior_file_info.get('shared_predictors', [])
        if shared_predictors is None:
            shared_predictors = []

        # preserve order and avoid duplicates
        shared_predictors = list(shared_predictors)
        assert len(set(shared_predictors)) == len(shared_predictors), \
            "Duplicate shared predictors in prior file"
        assert len(set(preds_traj).intersection(set(shared_predictors))) == 0, \
            "Predictors cannot be both trajectory-specific and shared"

        preds = preds_traj + shared_predictors

        D = len(targets)
        M = len(preds)
        
        if 'w_mu' in prior_file_info.keys():
            if prior_file_info['w_mu'] is not None:                
                K = prior_file_info['w_mu'][preds[0]][targets[0]].shape[0]
            else:
                K = int(op.k)
        else:
            K = int(op.k)
        
        prior_data = {}
        for i in ['v_a', 'v_b', 'w_mu', 'w_var', 'lambda_a', 'lambda_b',
                  'traj_probs', 'R', 'probs_weight', 'w_mu0', 'w_var0',
                  'lambda_a0', 'lambda_b0', 'alpha',  'Sig0', 'ranefs',
                  'ranef_indices', 'pred_to_ranef_index']:
            prior_data[i] = None
    
        prior_data['w_mu0'] = np.zeros([M, D])
        prior_data['w_var0'] = np.ones([M, D])
        prior_data['lambda_a0'] = np.ones([D])
        prior_data['lambda_b0'] = np.ones([D])
        prior_data['shared_predictors'] = shared_predictors
        prior_data['w_mu0_shared'] = np.zeros([len(shared_predictors), D])
        prior_data['w_var0_shared'] = np.ones([len(shared_predictors), D])
        prior_data['R'] = None

        if 'v_a' in prior_file_info.keys():
            prior_data['v_a'] = prior_file_info['v_a']
            if prior_file_info['v_a'] is not None:
                K = prior_file_info['v_a'].shape[0]
                print("Using K={} (from prior)".format(K))
        if 'v_b' in prior_file_info.keys():
            prior_data['v_b'] = prior_file_info['v_b']            

        if 'w_mu' in prior_file_info.keys():
            if prior_file_info['w_mu'] is not None:
                prior_data['w_mu'] = np.zeros([M, D, K])
        if 'w_var' in prior_file_info.keys():
            if prior_file_info['w_var'] is not None:
                prior_data['w_var'] = np.ones([M, D, K])
        if 'lambda_a' in prior_file_info.keys():
            if prior_file_info['lambda_a'] is not None:
                prior_data['lambda_a'] = np.ones([D, K])
        if 'lambda_b' in prior_file_info.keys():
            if prior_file_info['lambda_b'] is not None:
                prior_data['lambda_b'] = np.ones([D, K])
        if 'traj_probs' in prior_file_info.keys():
            prior_data['traj_probs'] = prior_file_info['traj_probs']
        if 'R' in prior_file_info.keys():
            prior_data['R'] = prior_file_info['R']
        if 'Sig0' in prior_file_info.keys():
            prior_data['Sig0'] = prior_file_info['Sig0']
        if 'ranef_indices' in prior_file_info.keys():
            ranef_indices_traj = prior_file_info['ranef_indices']

            if ranef_indices_traj is None:
                prior_data['ranef_indices'] = None
            else:
                ranef_indices_traj = \
                    np.asarray(ranef_indices_traj).astype(bool)

                assert ranef_indices_traj.shape[0] == len(preds_traj), \
                    "ranef_indices length does not match number of trajectory-specific predictors"

                ranef_indices_full = np.zeros(len(preds), dtype=bool)
                ranef_indices_full[:len(preds_traj)] = ranef_indices_traj

                prior_data['ranef_indices'] = ranef_indices_full        
            
        prior_data['alpha'] = prior_file_info['alpha']
        for (d, target) in enumerate(op.targets.split(',')):
            prior_data['lambda_a0'][d] = prior_file_info['lambda_a0'][target]
            prior_data['lambda_b0'][d] = prior_file_info['lambda_b0'][target]            

            if prior_data['lambda_a'] is not None:
                prior_data['lambda_a'][d, :] = \
                    prior_file_info['lambda_a'][target]
            if prior_data['lambda_b'] is not None:
                prior_data['lambda_b'][d, :] = \
                    prior_file_info['lambda_b'][target]

            for (m, pred) in enumerate(preds_traj):
                prior_data['w_mu0'][m, d] = \
                    prior_file_info['w_mu0'][target][pred]
                prior_data['w_var0'][m, d] = \
                    prior_file_info['w_var0'][target][pred]
                if prior_data['w_mu'] is not None:
                    prior_data['w_mu'][m, d, :] = \
                        prior_file_info['w_mu'][pred][target]
                if prior_data['w_var'] is not None:
                    prior_data['w_var'][m, d, :] = \
                        prior_file_info['w_var'][pred][target]

            for (j, pred) in enumerate(shared_predictors):
                # legacy block gets benign defaults for shared predictors
                m = len(preds_traj) + j
                prior_data['w_mu0'][m, d] = 0.0
                prior_data['w_var0'][m, d] = 1.0

                # explicit shared prior block
                if 'w_mu0_shared' in prior_file_info and \
                   target in prior_file_info['w_mu0_shared'] and \
                   pred in prior_file_info['w_mu0_shared'][target]:
                    prior_data['w_mu0_shared'][j, d] = \
                        prior_file_info['w_mu0_shared'][target][pred]
                else:
                    prior_data['w_mu0_shared'][j, d] = 0.0

                if 'w_var0_shared' in prior_file_info and \
                   target in prior_file_info['w_var0_shared'] and \
                   pred in prior_file_info['w_var0_shared'][target]:
                    prior_data['w_var0_shared'][j, d] = \
                        prior_file_info['w_var0_shared'][target][pred]
                else:
                    prior_data['w_var0_shared'][j, d] = 1.0            
                    
    if op.alpha is not None:
        prior_data['alpha'] = float(op.alpha)

    # Random effects plus the historical mean-field path are retained only for
    # backward compatibility.  The structured factorization is the corrected
    # implementation and should be used for new random-effect fits.
    if op.ranef_factorization == 'mean_field' and \
       prior_data.get('ranef_indices') is not None and \
       np.any(np.asarray(prior_data['ranef_indices']).astype(bool)):
        warnings.warn(
            'Random effects are configured with --ranef_factorization '
            'mean_field. This is the legacy random-effect implementation and '
            'is retained for backward compatibility only. Use '
            '--ranef_factorization structured for new fits.',
            FutureWarning)

    # Historical mode preserves the prior-defined K behavior. Structured DP
    # mode instead treats -k as a truncation ceiling. If the prior contains
    # fewer initialized components, pad the initialization arrays so posterior
    # occupancy can move above the prior's starting K.
    if op.ranef_factorization == 'structured' and op.resume_model is None:
        requested_k = int(op.k)
        prior_k = int(K)
        if requested_k < prior_k:
            warnings.warn(
                f"Structured truncation -k={requested_k} is below the prior's "
                f"K={prior_k}; using K={prior_k} instead.")
            requested_k = prior_k
        if requested_k > prior_k:
            print(
                f"Structured DP: expanding prior initialization from K={prior_k} "
                f"to truncation K={requested_k}.")
            prior_data = _expand_structured_prior_to_truncation(
                prior_data, prior_k, requested_k)
        K = requested_k

    print("Reading data...")
    df = pd.read_csv(in_csv)
    
    if np.sum(np.isnan(np.sum(df[preds].values, 1))) > 0:
        print("Warning: identified NaNs in predictor set. \
        Proceeding with non-NaN data")
        df = df.dropna(subset=preds).reset_index()

    #---------------------------------------------------------------------------
    # Get fixed values if any
    #---------------------------------------------------------------------------
    w_mu_fixed = None
    if op.fix is not None:
        w_mu_fixed = torch.nan*torch.ones([M, D, K])
        for tt in op.fix:
            assert len(tt[0].split(',')) == 4
            tmp_target = tt[0].split(',')[0]
            tmp_pred = tt[0].split(',')[1]
            tmp_traj = int(tt[0].split(',')[2])
            tmp_val = float(tt[0].split(',')[3])
            assert tmp_target in targets
            assert tmp_traj in range(0, K)

            which_target = \
                [i for i, s in enumerate(targets) if s == tmp_target][0]
            which_pred = \
                [i for i, s in enumerate(preds) if s == tmp_pred][0]
            w_mu_fixed[which_pred, which_target, tmp_traj] = tmp_val

    #---------------------------------------------------------------------------
    # Get soft-fix values if any
    #---------------------------------------------------------------------------
    w_mu0_override = None
    w_var0_override = None    
    if op.soft_fix is not None:
        w_mu0_override = torch.nan*torch.ones([M, D, K])
        w_var0_override = torch.nan*torch.ones([M, D, K])
        
        for tt in op.soft_fix:
            assert len(tt[0].split(',')) == 5
            tmp_target = tt[0].split(',')[0]
            tmp_pred = tt[0].split(',')[1]
            tmp_traj = int(tt[0].split(',')[2])
            tmp_mu = float(tt[0].split(',')[3])
            tmp_std = float(tt[0].split(',')[4])
            assert tmp_target in targets
            assert tmp_traj in range(0, K)
            assert tmp_std > 0

            which_target = \
                [i for i, s in enumerate(targets) if s == tmp_target][0]
            which_pred = \
                [i for i, s in enumerate(preds) if s == tmp_pred][0]
            w_mu0_override[which_pred, which_target, tmp_traj] = tmp_mu
            w_var0_override[which_pred, which_target, tmp_traj] = tmp_std**2            
        
    #---------------------------------------------------------------------------
    # Set up and run the traj alg
    #---------------------------------------------------------------------------
    repeat_rows = []
    best_mm = None
    best_waic2 = sys.float_info.max
    best_elbo = -sys.float_info.max
    best_repeat = None

    structured_mode = op.ranef_factorization == 'structured'
    resume_mode = op.resume_model is not None
    if resume_mode and not structured_mode:
        raise ValueError('--resume_model requires --ranef_factorization structured')
    if resume_mode and repeats != 1:
        raise ValueError('--resume_model currently requires --repeats 1')
    if resume_mode and op.alpha is not None:
        raise ValueError(
            '--alpha cannot be changed while resuming; start a new fit instead')
    if structured_mode and op.num_fin_trajs is not None:
        warnings.warn(
            '--num_fin_trajs is ignored in structured DP mode; posterior '
            'occupancy is data-driven.')
    if structured_mode and op.waic2_thresh < sys.float_info.max:
        warnings.warn(
            '--waic2_thresh is ignored for structured repeat selection; '
            'final ELBO is used instead.')

    repeat_summary_path = op.repeat_summary
    if structured_mode and repeat_summary_path is None:
        if op.out_model is not None:
            repeat_summary_path = str(op.out_model) + '.repeat_summary.csv'
        elif op.out_csv is not None:
            repeat_summary_path = str(op.out_csv) + '.repeat_summary.csv'

    resume_mm = None
    if resume_mode:
        print(f"Reading structured model to resume: {op.resume_model}")
        with open(op.resume_model, 'rb') as f:
            resume_obj = pickle.load(f)
        if isinstance(resume_obj, dict) and 'MultDPRegression' in resume_obj:
            resume_obj = resume_obj['MultDPRegression']
        if not isinstance(resume_obj, MultDPRegression):
            raise TypeError(
                '--resume_model did not contain a MultDPRegression object')
        resume_mm = MultDPRegression(resume_obj)
        if getattr(resume_mm, 'ranef_factorization_', None) != 'structured':
            raise ValueError('--resume_model is not a structured fit')
        if list(resume_mm.target_names_) != list(targets):
            raise ValueError('resume model target names do not match --targets')
        if list(resume_mm.predictor_names_) != list(preds):
            raise ValueError(
                'resume model predictors do not match the supplied prior')
        if resume_mm.K_ != int(getattr(resume_mm, 'R_').shape[1]):
            raise ValueError('resume model has inconsistent K/R dimensions')
        K = int(resume_mm.K_)
        print(
            f"Continuing structured model at Kmax={K} from iteration "
            f"{getattr(resume_mm, 'n_iter_', 0)}.")

    print("Fitting...")
    for r in np.arange(repeats):
        repeat_seed = None
        if op.fit_seed is not None:
            repeat_seed = int(op.fit_seed) + int(r)
            np.random.seed(repeat_seed)
            torch.manual_seed(repeat_seed)
        if r > 0:
            if structured_mode:
                print(
                    f"---------- Repeat {r}, Best ELBO: {best_elbo:.6f} ----------")
            else:
                print(
                    f"---------- Repeat {r}, Best WAIC2: {best_waic2} ----------")

        if resume_mode:
            mm = MultDPRegression(resume_mm)
            mm.continue_structured_fit(
                iters=iters, verbose=op.verbose,
                weights_only=op.weights_only,
                ranef_cov_mode=op.ranef_cov_mode,
                ranef_cov_min_eig=op.ranef_cov_min_eig,
                structured_tol_elbo_rel=op.structured_tol_elbo_rel,
                structured_tol_r=op.structured_tol_r,
                structured_tol_w=op.structured_tol_w,
                structured_tol_lambda=op.structured_tol_lambda,
                structured_tol_ranef_cov=op.structured_tol_ranef_cov,
                structured_min_iters=op.structured_min_iters,
                structured_r_damping=op.structured_r_damping,
                ranef_cov_warmup_iters=op.ranef_cov_warmup_iters,
                reset_history=op.resume_reset_history)
        elif True: #not op.use_pyro:
            mm = MultDPRegression(prior_data['w_mu0'],
                                  prior_data['w_var0'],
                                  prior_data['lambda_a0'],
                                  prior_data['lambda_b0'],
                                  op.prec_prior_weight,
                                  prior_data['alpha'], K=K,
                                  Sig0=prior_data['Sig0'],
                                  ranef_indices=prior_data['ranef_indices'],
                                  prob_thresh=op.prob_thresh)

            if len(shared_predictors) > 0:
                mm.w_mu0_shared_ = \
                    torch.from_numpy(prior_data['w_mu0_shared']).double()
                mm.w_var0_shared_ = \
                    torch.from_numpy(prior_data['w_var0_shared']).double()

            mm.fit(target_names=targets, predictor_names=preds, df=df,
                   groupby=op.groupby, iters=iters, verbose=op.verbose,
                   R=prior_data['R'],
                   traj_probs=prior_data['traj_probs'],
                   traj_probs_weight=op.probs_weight,
                   v_a=prior_data['v_a'],
                   v_b=prior_data['v_b'],
                   w_mu=prior_data['w_mu'],
                   w_var=prior_data['w_var'],
                   lambda_a=prior_data['lambda_a'],
                   lambda_b=prior_data['lambda_b'],
                   weights_only=op.weights_only,
                   num_init_trajs=op.num_init_trajs,
                   w_mu0_override=w_mu0_override,
                   w_var0_override=w_var0_override,
                   w_mu_fixed=w_mu_fixed,
                   shared_predictor_names=shared_predictors,
                   ranef_factorization=op.ranef_factorization,
                   ranef_cov_mode=op.ranef_cov_mode,
                   ranef_cov_min_eig=op.ranef_cov_min_eig,
                   structured_tol_elbo_rel=op.structured_tol_elbo_rel,
                   structured_tol_r=op.structured_tol_r,
                   structured_tol_w=op.structured_tol_w,
                   structured_tol_lambda=op.structured_tol_lambda,
                   structured_tol_ranef_cov=op.structured_tol_ranef_cov,
                   structured_min_iters=op.structured_min_iters,
                   structured_r_damping=op.structured_r_damping,
                   ranef_cov_warmup_iters=op.ranef_cov_warmup_iters)
        else:
            restructured_data = get_restructured_data(
                df, preds, targets, op.groupby)
            model = MultPyro(
                alpha0=torch.full((K,), 100.0, dtype=torch.double),
                w_mu0=torch.from_numpy(prior_data['w_mu0'].T).double(),
                w_var0=torch.from_numpy(prior_data['w_var0'].T).double(),
                lambda_a0=torch.from_numpy(prior_data['lambda_a0']).double(),
                lambda_b0=torch.from_numpy(prior_data['lambda_b0']).double(),
                **restructured_data)
            model.fit(num_steps=iters)
            if op.out_model is not None:
                torch.save(model, op.out_model)
                _write_provenance(op.out_model, op)
            continue

        if structured_mode:
            final_elbo = (
                mm.inference_history_[-1]['elbo']
                if len(mm.inference_history_) > 0
                else mm.compute_structured_elbo())
            occ = mm.get_structured_occupancy_diagnostics()
            mm.structured_occupancy_diagnostics_ = occ
            waic2 = np.nan
            if op.structured_compute_waic:
                waic2 = mm.compute_waic2(op.s, op.seed)
            hist = mm.inference_history_
            last = hist[-1] if len(hist) > 0 else {}
            segment = int(last.get('fit_segment', getattr(mm, 'fit_segment_', 0)))
            segment_rows = [
                hh for hh in hist if int(hh.get('fit_segment', 0)) == segment]
            deltas = np.asarray(
                [hh.get('delta_elbo', np.nan) for hh in segment_rows],
                dtype=float)
            finite_deltas = deltas[np.isfinite(deltas)]
            negative_steps, worst_negative_step, max_negative_tol = \
                _material_negative_elbo_steps(segment_rows)
            eligible, ineligible_reasons = _structured_repeat_eligibility(
                mm, final_elbo, negative_steps)
            row = {
                'repeat': int(r),
                'fit_seed': repeat_seed,
                'eligible': bool(eligible),
                'ineligibility_reasons': ';'.join(ineligible_reasons),
                'selection_metric': 'elbo',
                'final_elbo': float(final_elbo),
                'objective_converged': mm.objective_converged_,
                'parameter_converged': bool(mm.parameter_converged_),
                'converged': bool(mm.converged_),
                'fit_complete': bool(getattr(mm, 'fit_complete_', False)),
                'ranef_cov_strategy': getattr(
                    mm, 'ranef_cov_strategy_', mm.ranef_cov_mode_),
                'ranef_cov_stage': getattr(
                    mm, 'ranef_cov_stage_', mm.ranef_cov_mode_),
                'ranef_cov_release_iteration': getattr(
                    mm, 'ranef_cov_release_iteration_', None),
                'iterations': int(mm.n_iter_),
                'fit_segment': segment,
                'segment_iterations': int(len(segment_rows)),
                'final_delta_elbo': last.get('delta_elbo', np.nan),
                'final_relative_delta_elbo': last.get(
                    'relative_delta_elbo', np.nan),
                'final_max_delta_r': last.get('max_delta_r', np.nan),
                'final_max_delta_w_mean': last.get(
                    'max_delta_w_mean', np.nan),
                'final_max_delta_expected_precision': last.get(
                    'max_delta_expected_precision', np.nan),
                'final_max_delta_ranef_cov': last.get(
                    'max_delta_ranef_cov', np.nan),
                'final_map_agreement_previous': last.get(
                    'map_agreement_previous', np.nan),
                'negative_elbo_steps': negative_steps,
                'elbo_monotone_within_tolerance': bool(negative_steps == 0),
                'worst_material_negative_elbo_step': worst_negative_step,
                'maximum_negative_elbo_numerical_tolerance': max_negative_tol,
                'minimum_delta_elbo': (
                    float(np.min(finite_deltas))
                    if finite_deltas.size > 0 else np.nan),
                'expected_occupied_k': occ['expected_occupied_k'],
                'map_occupied_k': occ['map_occupied_k'],
                'n_eff_ge_1_k': occ['n_eff_ge_1_k'],
                'residual_stick_mass_beyond_truncation': (
                    occ['residual_stick_mass_beyond_truncation']),
                'tail_expected_membership': occ['tail_expected_membership'],
                'tail_expected_stick_weight': occ['tail_expected_stick_weight'],
                'effective_membership_json': json.dumps(
                    occ['effective_membership']),
                'probability_occupied_json': json.dumps(
                    occ['probability_occupied']),
                'expected_stick_weights_json': json.dumps(
                    occ['expected_stick_weights']),
                'waic2_diagnostic': float(waic2) if np.isfinite(waic2) else np.nan,
            }
            repeat_rows.append(row)
            print(
                f"Repeat {r}: ELBO={final_elbo:.6f}, "
                f"E[Kocc]={occ['expected_occupied_k']:.2f}, "
                f"MAPK={occ['map_occupied_k']}, "
                f"tail stick={occ['tail_expected_stick_weight']:.3e}, "
                f"objective_conv={mm.objective_converged_}, "
                f"parameter_conv={mm.parameter_converged_}, "
                f"eligible={eligible}")
            if not eligible:
                print('  ineligible: ' + ', '.join(ineligible_reasons))

            if eligible and final_elbo > best_elbo:
                best_elbo = float(final_elbo)
                best_mm = mm
                best_repeat = int(r)
        else:
            waic2 = mm.compute_waic2(op.s, op.seed)
            nfin = int(torch.sum(mm.sig_trajs_).item())
            repeat_rows.append({
                'repeat': int(r),
                'fit_seed': repeat_seed,
                'selection_metric': 'waic2',
                'waic2': float(waic2),
                'num_final_trajs': nfin,
            })
            if (waic2 < best_waic2) and (waic2 < op.waic2_thresh) and \
               (op.num_fin_trajs is None or op.num_fin_trajs == nfin):
                best_waic2 = float(waic2)
                best_mm = mm
                best_repeat = int(r)

        if repeat_summary_path is not None:
            pd.DataFrame(repeat_rows).to_csv(repeat_summary_path, index=False)

    if best_mm is None:
        for rr in repeat_rows:
            rr['selected'] = False
        if repeat_summary_path is not None:
            pd.DataFrame(repeat_rows).to_csv(repeat_summary_path, index=False)
            print(f"Saved repeat summary: {repeat_summary_path}")
            _write_provenance(repeat_summary_path, op)
        raise RuntimeError(
            'No eligible fit was produced across repeats; no model was selected. '
            'Inspect the complete repeat summary for QC failure reasons.')

    best_mm.repeat_selection_metric_ = 'eligible_final_elbo' if structured_mode else 'waic2'
    best_mm.best_repeat_ = best_repeat
    for rr in repeat_rows:
        rr['selected'] = bool(int(rr['repeat']) == int(best_repeat))
    best_mm.repeat_summary_ = repeat_rows

    if structured_mode:
        print(
            f"Selected eligible repeat {best_repeat} with highest ELBO {best_elbo:.6f}.")
        if not best_mm.parameter_converged_:
            warnings.warn(
                'Selected structured repeat is objective-converged and eligible '
                'but did not meet all parameter-change tolerances; these are '
                'reported as stricter diagnostics rather than selection gates.')
    else:
        print(
            f"Selected repeat {best_repeat} with lowest WAIC2 {best_waic2}.")

    if op.out_model is not None:
        print("Saving model...")
        with open(op.out_model, 'wb') as f:
            pickle.dump({'MultDPRegression': best_mm}, f)
        print("Saving model provenance info...")
        _write_provenance(op.out_model, op)

    if op.out_csv is not None:
        print("Saving data file with trajectory info...")
        best_mm.to_df().to_csv(op.out_csv, index=False)
        print("Saving data file provenance info...")
        _write_provenance(op.out_csv, op)

    if repeat_summary_path is not None:
        pd.DataFrame(repeat_rows).to_csv(repeat_summary_path, index=False)
        print(f"Saved repeat summary: {repeat_summary_path}")
        _write_provenance(repeat_summary_path, op)

    print("DONE.")

if __name__ == "__main__":
    main()
        

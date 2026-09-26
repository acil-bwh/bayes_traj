#!/usr/bin/env python

import torch
import numpy as np
import pdb
import pickle
import pandas as pd
from argparse import ArgumentParser
from provenance_tools.write_provenance_data import write_provenance_data
from bayes_traj.fit_stats import (
    ave_pp, odds_correct_classification, prob_prop, assignment_diagnostics,
    gaussian_information_criteria, reportable_trajectory_ids,
    group_responsibilities)

def compute_weighted_posterior_population_cov(mu_list, Sigma_list, weights):
    """
    Computes a weighted posterior population covariance matrix.

    Parameters:
    -----------
    mu_list : list or array of shape [N, d] 
        Posterior means

    Sigma_list : list or array of shape [N, d, d] 
        Posterior covariances

    weights: array of shape [N] 
        Non-negative, need not be normalized

    Returns:
    --------
    pop_cov: [d, d] 
        Posterior estimate of population covariance matrix
    """
    mu_array = np.stack(mu_list)         # shape: [N, d]
    Sigma_array = np.stack(Sigma_list)   # shape: [N, d, d]
    weights = np.array(weights).astype(np.float64)

    # Normalize weights
    weights /= np.sum(weights)           # now sum to 1
    N, d = mu_array.shape

    # Compute weighted mean of the means
    mu_bar = np.sum(weights[:, None] * mu_array, axis=0)  # shape: [d]

    # Compute weighted average of the Sigma_i matrices
    weighted_covs = np.sum(weights[:, None, None] * Sigma_array, axis=0)  

    # Compute weighted covariance of the mu_i means
    diffs = mu_array - mu_bar            # shape: [N, d]
    weighted_outer = np.einsum('ni,nj,n->ij', diffs, diffs, weights)  # shape: [d, d]

    # Total population covariance
    pop_cov = weighted_covs + weighted_outer
    
    return pop_cov


def get_ranef_cov_mat_output_str(mm, d, k, precision=3,
                                 sci_notation_threshold=1e-3):
    """Pretty-prints and returns a string of the selected random effects 
    covariance matrix from a model object `mm` for dimension `d` and component 
    `k`.

    Arguments:
    ----------
    mm : MultDPRegression instance
        Assumes attributes u_Sig_, ranef_indices_, predictor_names_

    d : int
        Target dimension index

    k : int
        Trajectory number

    precision : int
        Number of digits after decimal point

    sci_notation_threshold : float
        Below this absolute value, use scientific notation

    Returns:
    --------
    output_str : str
        Formatted string representation of the covariance matrix
    """
    probs = torch.as_tensor(
        group_responsibilities(mm)[:, k], dtype=torch.float64)

    cov_mat = compute_weighted_posterior_population_cov(\
        mm.u_mu_[:, d, k, mm.ranef_indices_],
        mm.u_Sig_[:, d, k, mm.ranef_indices_, :][:, :, mm.ranef_indices_],
        probs)

    # Predictor names
    preds_sel = np.array(mm.predictor_names_)[mm.ranef_indices_]
    num_preds_sel = preds_sel.shape[0]

    name_width = max(len(nn) for nn in preds_sel) + 2
    cell_width = max(precision, name_width) + 7  # space for sign, decimal, etc.

    # Prepare output lines
    lines = []

    # Header row
    header = " " * name_width + "".join(
        f"{nn:>{cell_width}}" for nn in preds_sel
    )
    lines.append(header)

    # Matrix rows
    for ii, nn in enumerate(preds_sel):
        row_vals = []
        for jj in range(num_preds_sel):
            val = cov_mat[ii, jj]
            if abs(val) < sci_notation_threshold and val != 0:
                formatted = f"{val:>{cell_width}.{precision}e}"
            else:
                formatted = f"{val:>{cell_width}.{precision}f}"
            row_vals.append(formatted)
        row_str = f"{nn:<{name_width}}" + "".join(row_vals)
        lines.append(row_str)

    output_str = "\n".join(lines)

    return output_str
    
def main():        
    desc = """"""
    
    parser = ArgumentParser(description=desc)
    parser.add_argument('--model', help='Bayesian trajectory model to summarize',
        type=str, required=True)
    parser.add_argument('--trajs', help='Comma-separated list of integers \
        indicating trajectories for which to print results. If none specified, \
        results for all trajectories will be printed', default=None)
    parser.add_argument('--min_traj_prob', help='The probability of a given \
        trajectory must be at least this value in order for results to be printed \
        for that trajectory. Value should be between 0 and 1 inclusive.', \
        type=float, default=0)
    parser.add_argument('--min_effective_membership', type=float, default=1.0,
        help='Structured-DP reporting threshold on posterior effective membership. '
        'This is descriptive only and does not alter the fitted model.')
    parser.add_argument('--hide_ic', help='Hide plug-in Gaussian information '
        'criteria. Assignment/occupancy diagnostics are still reported.',
        action="store_true")
    parser.add_argument('--compute_waic', action='store_true',
        help='Also compute the historical WAIC2 diagnostic. This is off by '
        'default because it is comparatively expensive and is not recommended '
        'for structured-DP model selection.')
    parser.add_argument('-s', help='Number of samples to use when computing \
        WAIC2', type=int, default=100)
    parser.add_argument('--seed', help='Seed to use for WAIC2 \
            sampling', type=int, default=None)
    
    op = parser.parse_args()

    sci_notation_threshold = 1e-2
    
    with open(op.model, 'rb') as f:
        mm = pd.read_pickle(f)['MultDPRegression']

    traj_probs = mm.get_traj_probs() if hasattr(mm, 'get_traj_probs') else \
        np.mean(group_responsibilities(mm), axis=0)

    reportable_ids = reportable_trajectory_ids(
        mm, min_effective_membership=op.min_effective_membership)
    if op.trajs is not None:
        traj_ids = np.array(op.trajs.split(','), dtype=int)
    else:
        traj_ids = reportable_ids
    all_traj_ids = reportable_ids

    # Information criteria based on the Gaussian plug-in subject-level marginal
    # likelihood. Historical WAIC is deliberately opt-in: it is retained as a
    # diagnostic for backward compatibility, not as the structured-DP model
    # selection criterion.
    ic = None
    waic2 = None
    fit_stat_notes = []
    if not op.hide_ic:
        try:
            ic = gaussian_information_criteria(mm, traj_ids=reportable_ids)
        except (NotImplementedError, ValueError, np.linalg.LinAlgError) as exc:
            fit_stat_notes.append(str(exc))
    if op.compute_waic:
        assert isinstance(op.s, int), 'Number of samples must be an integer.'
        assert op.s > 0, 'Number of samples must be greater than 0.'
        try:
            waic2 = mm.compute_waic2(op.s, op.seed)
        except Exception as exc:
            # Summary metrics should remain usable if the legacy WAIC path is not
            # applicable to a newer model configuration.
            fit_stat_notes.append(f'WAIC unavailable: {exc}')

    # Compute fit stats
    ave_pps = ave_pp(mm, traj_ids=reportable_ids)
    occs = odds_correct_classification(mm, traj_ids=reportable_ids)
    prop_probs = prob_prop(mm, traj_ids=reportable_ids)
    assign_diag = assignment_diagnostics(mm, traj_ids=reportable_ids)
    
    df_traj = mm.to_df()
    
    # Get dataframe column that was used to create groups, if groups exist
    if mm.gb_ is not None:
        groupby_col = mm.gb_.keys if isinstance(mm.gb_.keys, str) \
            else mm.gb_.keys().name        
        num_groups = (mm.gb_.obj).groupby(groupby_col).ngroups
    else:
        num_groups = df_traj.shape[0]
        
    max_tar_name_len = 0
    for tar in mm.target_names_:
        if len(tar) > max_tar_name_len:
            max_tar_name_len = len(tar)
    
    max_pred_name_len = 0
    for pred in mm.predictor_names_:
        if len(pred) > max_pred_name_len:
            max_pred_name_len = len(pred)
            
    first_col_width = max_tar_name_len + max_pred_name_len + 3
    row_width = first_col_width + 60

    print("Summary".center(row_width))
    print("="*row_width)
    print("{}{}".format("Reportable Trajs:".ljust(24),
                        "{}".format(len(all_traj_ids))))
    print("{}{}".format("Trajectories:".ljust(24),
        "{}".format(','.join(list(all_traj_ids.astype('str')))).ljust(40)))
    print("{}{}".format("Truncation K:".ljust(24), str(mm.K_)))
    print("{}{}".format("No. Observations:".ljust(24),
                        str(df_traj.shape[0]).ljust(15)))
    print("{}{}".format("No. Groups:".ljust(24), str(num_groups).ljust(15)))

    if getattr(mm, 'ranef_factorization_', 'mean_field') == 'structured':
        occ = mm.get_structured_occupancy_diagnostics()
        print("{}{:0.3f}".format("Expected occupied K:".ljust(24),
                                 occ['expected_occupied_k']))
        print("{}{}".format("MAP occupied K:".ljust(24),
                            occ['map_occupied_k']))
        print("{}{:.3e}".format("Residual stick mass:".ljust(24),
                                occ['residual_stick_mass_beyond_truncation']))
        print("{}{}".format("Covariance strategy:".ljust(24),
            getattr(mm, 'ranef_cov_strategy_', getattr(mm, 'ranef_cov_mode_', 'fixed'))))
        print("{}{}".format("Objective converged:".ljust(24),
                            getattr(mm, 'objective_converged_', None)))
        print("{}{}".format("Parameter converged:".ljust(24),
                            getattr(mm, 'parameter_converged_', None)))
        if getattr(mm, 'ranef_cov_strategy_', None) == 'staged':
            staged = getattr(mm, 'staged_covariance_diagnostics_', None) or {}
            print("{}{}".format("D release iteration:".ljust(24),
                                staged.get('release_iteration', None)))
            if staged.get('fixed_phase_final_elbo') is not None:
                print("{}{:.3f}".format("Fixed-D terminal ELBO:".ljust(24),
                    staged['fixed_phase_final_elbo']))

    print("{}{:.4f}".format("Entropy:".ljust(24), assign_diag['entropy']))
    print("{}{:.4f}".format("Mean max posterior:".ljust(24),
                            assign_diag['mean_max_posterior']))
    print("{}{:.4f}".format("Median max posterior:".ljust(24),
                            assign_diag['median_max_posterior']))
    for thr in (0.7, 0.8, 0.9):
        key = f'proportion_max_posterior_ge_{thr:g}'
        print("{}{:.3f}".format(f"P(max PP >= {thr:.1f}):".ljust(24),
                                assign_diag[key]))

    if ic is not None:
        print("\nGaussian plug-in marginal information criteria")
        print("(random effects integrated; global parameters held at fitted expectations)")
        print("{}{:,.3f}".format("Log likelihood:".ljust(24), ic['log_likelihood']))
        print("{}{}".format("Global parameters:".ljust(24), ic['num_parameters']))
        print("{}{:,.3f}".format("AIC:".ljust(24), ic['aic']))
        print("{}{:,.3f}".format("BIC:".ljust(24), ic['bic']))
        print("{}{:,.3f}".format("SABIC:".ljust(24), ic['sabic']))
        print("{}{:,.3f}".format("ICL1 (log-max):".ljust(24), ic['icl1']))
        print("{}{:,.3f}".format("ICL2 (entropy):".ljust(24), ic['icl2']))
    if waic2 is not None:
        print("{}{:,.3f}".format("Legacy WAIC2:".ljust(24), float(waic2)))
    for note in fit_stat_notes:
        print(f"Fit-stat note: {note}")

    #---------------------------------------------------------------------------
    # Shared continuous fixed effects
    #---------------------------------------------------------------------------
    if hasattr(mm, 'shared_indices_') and hasattr(mm, 'w_mu_shared_') and \
       (len(mm.shared_indices_) > 0):

        output_str = '\n'
        output_str += 'Shared continuous fixed effects\n'
        output_str += '--------------------------------\n'

        for (ii, target) in enumerate(mm.target_names_):
            if mm.target_type_[ii] != 'gaussian':
                continue

            output_str += '\n'
            output_str += f'Target: {target}\n'
            output_str += '\n'
            output_str += '{:<20} {:>12} {:>12}\n'.format(
                'Predictor', 'Mean', 'Std')

            output_str += '{:<20} {:>12} {:>12}\n'.format(
                '---------', '----', '---')

            for (j, pred_idx) in enumerate(mm.shared_indices_):
                pred = mm.predictor_names_[pred_idx]
                co = mm.w_mu_shared_[j, ii]
                std = np.sqrt(mm.w_var_shared_[j, ii])

                if torch.is_tensor(co):
                    co = co.item()
                if torch.is_tensor(std):
                    std = std.item()

                output_str += \
                    '{:<20} {:>12.4f} {:>12.4f}\n'.format(pred, co, std)
        print(output_str)                
            
    for traj in traj_ids:
        if traj_probs[traj] > op.min_traj_prob:
            print("")
            print("Summary for Trajectory {}".format(traj).center(row_width))
            print("="*row_width)
        
            num_obs_in_traj = sum(df_traj.traj.values == traj)
            if mm.gb_ is not None:
                num_groups_in_traj = df_traj[df_traj.traj.values == traj].\
                    groupby(groupby_col).ngroups
            else:
                num_groups_in_traj = num_obs_in_traj
                
            perc = 100*num_groups_in_traj/num_groups

            print("{}{}".format("No. Observations:".ljust(35), "{}".\
                                format(num_obs_in_traj).rjust(15)))
            print("{}{}".format("No. Groups:".ljust(35), "{}".\
                                format(num_groups_in_traj).rjust(15)))
            print("{}{}".format("% of Sample:".ljust(35), "{:.1f}".\
                                format(perc).rjust(15)))
            print("{}{}".format("Odds Correct Classification:".ljust(35), "{:.1f}".\
                                format(occs[traj]).rjust(15))) 
            print("{}{}".format("Ave. Post. Prob. of Assignment:".ljust(35), \
                                "{:.2f}".format(ave_pps[traj]).rjust(15)))     
            if traj in prop_probs:
                prop, prob = prop_probs[traj]
                print("{}{}".format("MAP Proportion:".ljust(35),
                                    "{:.3f}".format(prop).rjust(15)))
                print("{}{}".format("Posterior Mean Probability:".ljust(35),
                                    "{:.3f}".format(prob).rjust(15)))
        
            print("")
            print("{}{}{}{}".format(" "*first_col_width, "Residual STD".center(20),
                                    "Precision Mean".center(20),
                                    "Precision Var".center(20)))
            print("-"*row_width)
            for (ii, tar) in enumerate(mm.target_names_):
                prec_mean = mm.lambda_a_[ii, traj]/mm.lambda_b_[ii, traj]
                prec_var = mm.lambda_a_[ii, traj]/(mm.lambda_b_[ii, traj]**2)
                resid_std = np.sqrt(1/prec_mean)
                if mm.target_type_[ii] == 'binary':
                    space = " "*(first_col_width - len(tar))
                    print("{}{}{}{}{}".format(tar, space,
                                        "NA".center(20),
                                        "NA".center(20),
                                        "NA".center(20)))
                else:
                    space = " "*(first_col_width - len(tar))
                    print("{}{}{}{}{}".format(tar, space,
                                        "{:.2f}".format(resid_std).center(20),
                                        "{:.2f}".format(prec_mean).center(20),
                                        "{:.4f}".format(prec_var).center(20)))
                    
            print("")
            print("{}{}{}{}".format(" "*first_col_width, "coef".center(20),
                                    "STD".center(20),
                                    "[95% Cred. Int.]".center(20)))
            print("-"*row_width)
            for (ii, tar) in enumerate(mm.target_names_):
                traj_indices = \
                    mm.traj_indices_ if hasattr(mm, 'traj_indices_') \
                    else np.arange(len(mm.predictor_names_))
                for jj in traj_indices:
                    pred = mm.predictor_names_[jj]
                    co = mm.w_mu_[jj, ii, traj]
                    std = np.sqrt(mm.w_var_[jj, ii, traj])
                
                    if abs(co) < sci_notation_threshold:
                        co_str = f'{co:.2e}'.center(20)
                    else:
                        co_str = f'{co:.3f}'.center(20)

                    if abs(std) < sci_notation_threshold:
                        std_str = f'{std:.2e}'.center(20)
                    else:
                        std_str = f'{std:.3f}'.center(20)                        
                        
                    low95 = co - 2*std
                    high95 = co + 2*std

                    if abs(low95) < sci_notation_threshold or \
                       abs(high95) < sci_notation_threshold:
                        interval = "{}{}".format("{:.2e}".format(low95).ljust(10),
                                                 "{:.2e}".format(high95).rjust(10))
                    else:
                        interval = "   {}{}   ".format("{:.3f}".format(low95).ljust(7),
                                                       "{:.3f}".format(high95).rjust(7))
                        
                    space = " "*(first_col_width - len(tar) - len(pred) - 3)
                    print("{} ({}){}{}{}{}".format(pred, tar, space, co_str,
                                                   std_str, interval))
                print("")

                if hasattr(mm, 'ranef_indices_'):
                    if mm.ranef_indices_ is not None:
                        if np.sum(mm.ranef_indices_) > 0:                        
                            ranef_cov_str = \
                                get_ranef_cov_mat_output_str(mm, ii, traj, 3,
                                        sci_notation_threshold)
                            print(f'Random effect posterior covariance matrix ({tar}):')
                            print(ranef_cov_str)
                            print("")                
            print("")
            


            
            
if __name__ == "__main__":
    main()
            

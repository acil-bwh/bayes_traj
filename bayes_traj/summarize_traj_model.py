#!/usr/bin/env python

import torch
import numpy as np
import pdb
import pickle
import pandas as pd
from argparse import ArgumentParser
from provenance_tools.write_provenance_data import write_provenance_data
from bayes_traj.fit_stats import ave_pp, odds_correct_classification

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
    probs = mm.R_[mm.group_first_index_, k]

    cov_mat = compute_weighted_posterior_population_cov(\
        mm.u_mu_[:, d, k, mm.ranef_indices_],
        mm.u_Sig_[:, d, k, mm.ranef_indices_, :][:, :, mm.ranef_indices_],
        mm.R_[mm.group_first_index_, k])

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
    parser.add_argument('--hide_ic', help='Use this flag to hide compuation \
        and display of information criterai (BIC and WAIC2), which can take \
        several moments to compute.', action="store_true")
    parser.add_argument('-s', help='Number of samples to use when computing \
        WAIC2', type=int, default=100)
    parser.add_argument('--seed', help='Seed to use for WAIC2 \
            sampling', type=int, default=None)
    
    op = parser.parse_args()

    sci_notation_threshold = 1e-2
    
    with open(op.model, 'rb') as f:
        mm = pd.read_pickle(f)['MultDPRegression']

    if torch.is_tensor(mm.R_):
        traj_probs = np.sum(mm.R_.numpy(), 0)/np.sum(mm.R_.numpy())
    else:
        traj_probs = np.sum(mm.R_, 0)/np.sum(mm.R_)
        
    if op.trajs is not None:
        traj_ids = np.array(op.trajs.split(','), dtype=int)
    else:
        traj_ids = np.where(mm.sig_trajs_)[0]
    
    all_traj_ids = np.where(mm.sig_trajs_)[0]

    if op.hide_ic:
        bic = None
        waic2 = None
    else:
        bic = mm.bic()
        assert isinstance(op.s, int), 'Number of samples must be an integer.'
        assert op.s > 0, 'Number of samples must be greater than 0.'
        waic2 = mm.compute_waic2(op.s, op.seed)
    
    # Compute fit stats
    ave_pps = ave_pp(mm)
    occs = odds_correct_classification(mm)
    
    df_traj = mm.to_df()
    
    # Get dataframe column that was used to create groups, if groups exist
    if mm.gb_ is not None:
        groupby_col = mm.gb_.keys if isinstance(mm.gb_.keys, str) \
            else self.gb_.keys().name        
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

    if torch.is_tensor(mm.sig_trajs_):
        sig_trajs = mm.sig_trajs_.numpy()
    else:
        sig_trajs = mm.sig_trajs_
    
    print("Summary".center(row_width))
    print("="*row_width)
    print("{}{}".format("Num. Trajs:".ljust(20),
                        "{}".format(np.sum(sig_trajs))))
    print("{}{}".format("Trajectories:".ljust(20),
        "{}".format(','.join(list(all_traj_ids.astype('str')))).ljust(40)))
    
    print("{}{}".format("No. Observations:".ljust(20), "{}".\
                        format(df_traj.shape[0]).ljust(15)))
    print("{}{}".format("No. Groups:".ljust(20), "{}".\
                        format(num_groups).ljust(15)))

    if waic2 is not None:
        print("{}{}".format("WAIC2:".ljust(20), "{}".\
                            format(int(waic2)).ljust(10))) 
    if bic is not None:
        if len(bic) == 2:
            print("{}{}".format("BIC1:".ljust(20), "{}".\
                                format(int(bic[0])).ljust(10)))
            print("{}{}".format("BIC2:".ljust(20), "{}".\
                                format(int(bic[1])).ljust(10))) 
        else:
            print("{}{}".format("BIC:".ljust(20), "{}".\
                                format(int(bic)).ljust(10)))         
        
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
                for (jj, pred) in enumerate(mm.predictor_names_):
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
            

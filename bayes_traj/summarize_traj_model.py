#!/usr/bin/env python

import torch
import numpy as np
import pdb
import pickle
from argparse import ArgumentParser
from provenance_tools.write_provenance_data import write_provenance_data
from bayes_traj.fit_stats import ave_pp, odds_correct_classification

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
    
    op = parser.parse_args()
    
    with open(op.model, 'rb') as f:
        mm = pickle.load(f)['MultDPRegression']

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
        waic2 = mm.compute_waic2()
    
    # Compute fit stats
    ave_pps = ave_pp(mm)
    occs = odds_correct_classification(mm)
    
    df_traj = mm.to_df()
    
    # Get dataframe column that was used to create groups, if groups exist
    if mm.gb_ is not None:
        num_groups = mm.gb_.ngroups
        groupby_col = mm.gb_.count().index.name               
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
                    co_str = "{:.3f}".format(co).center(20)
                    std_str = "{:.3f}".format(std).center(20)
                    low95 = co - 2*std
                    high95 = co + 2*std
                    interval = "   {}{}   ".format("{:.3f}".format(low95).ljust(7),
                                                   "{:.3f}".format(high95).rjust(7))
                    space = " "*(first_col_width - len(tar) - len(pred) - 3)
                    print("{} ({}){}{}{}{}".format(pred, tar, space, co_str,
                                                   std_str, interval))            
            print("")

if __name__ == "__main__":
    main()
            

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
from provenance_tools.write_provenance_data import write_provenance_data
import pickle, sys, warnings

torch.set_default_dtype(torch.double) # TODO -- may not be desirable to set this globally

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
    parser.add_argument('--iters', help='Number of inference iterations',
        dest='iters', metavar='<int>', default=100)
    parser.add_argument('--repeats', help='Number of repeats to attempt. If a \
        value greater than 1 is specified, the WAIC2 fit criterion will be \
        computed at the end of each repeat. If, for a given repeat, the WAIC2 \
        score is lower than the lowest score seen at that point, the model \
        will be saved to file.', type=int, metavar='<int>', default=1)
    parser.add_argument('-k', help='Number of columns in the truncated \
        assignment matrix', metavar='<int>', default=30)
    parser.add_argument('--prob_thresh', help='If during data fitting the \
        probability of a data instance belonging to a given trajectory drops \
        below this threshold, then the probabality of that data instance \
        belonging to the trajectory will be set to 0', metavar='<float>',
        type=float, default=0.001)
    parser.add_argument('--num_init_trajs', help='If specified, the \
        initialization procedure will attempt to ensure that the number of \
        initial trajectories in the fitting routine equals the specified \
        number.', metavar='<int>', type=int, default=None)        
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

        preds = get_pred_names_from_prior_info(prior_file_info)
        
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
            prior_data['ranef_indices'] = prior_file_info['ranef_indices']
        
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
            
            for (m, pred) in enumerate(preds):
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
                
    if op.alpha is not None:
        prior_data['alpha'] = float(op.alpha)

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
    waics_tracker = []
    bics_tracker = []
    num_tracker = []
    best_mm = None
    best_waic2 = sys.float_info.max
    bic_thresh = -sys.float_info.max
    best_bics = (bic_thresh, bic_thresh)

    print("Fitting...")
    for r in np.arange(repeats):        
        if r > 0:
            print(f"---------- Repeat {r}, Best WAIC2: {best_waic2} ----------")

        if True: #not op.use_pyro:
            mm = MultDPRegression(prior_data['w_mu0'],
                                  prior_data['w_var0'],
                                  prior_data['lambda_a0'],
                                  prior_data['lambda_b0'],
                                  op.prec_prior_weight,
                                  prior_data['alpha'], K=K,
                                  Sig0=prior_data['Sig0'],
                                  ranef_indices=prior_data['ranef_indices'],
                                  prob_thresh=op.prob_thresh)

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
                   w_mu_fixed=w_mu_fixed)
        else:
            restructured_data = get_restructured_data(df, preds, targets, op.groupby)
            model = MultPyro(
                alpha0=torch.full((K,), 100.0, dtype=torch.double),
                w_mu0=torch.from_numpy(prior_data['w_mu0'].T).double(),
                w_var0=torch.from_numpy(prior_data['w_var0'].T).double(),
                lambda_a0=torch.from_numpy(prior_data['lambda_a0']).double(),
                lambda_b0=torch.from_numpy(prior_data['lambda_b0']).double(),
                **restructured_data
            )

            model.fit(num_steps=iters)

            if op.out_model is not None:
                torch.save(model, op.out_model)

        waic2 = mm.compute_waic2(op.s, op.seed)
                
        if False: #op.use_pyro:
            pass
        elif r == 0:
            if (op.out_model is not None) and (waic2 < op.waic2_thresh):
                print("Saving model...")
                pickle.dump({'MultDPRegression': mm}, open(op.out_model, 'wb'))

                print("Saving model provenance info...")
                provenance_desc = """ """
                write_provenance_data(op.out_model, generator_args=op,
                                      desc=provenance_desc,
                                      module_name='bayes_traj')

            if op.out_csv is not None:
                print("Saving data file with trajectory info...")
                mm.to_df().to_csv(op.out_csv, index=False)

                print("Saving data file provenance info...")
                provenance_desc = """ """
                write_provenance_data(op.out_csv, generator_args=op,
                                      desc=provenance_desc,
                                      module_name='bayes_traj')
                
            if repeats > 1:
                best_waic2 = waic2
        else:            
            print(f"Current WAIC2: {waic2}")
            if (waic2 < best_waic2) and (waic2 < op.waic2_thresh):
                best_waic2 = waic2
        
                if op.out_model is not None:
                    print("Saving model...")
                    pickle.dump({'MultDPRegression': mm},
                                open(op.out_model, 'wb'))
    
                    print("Saving model provenance info...")
                    provenance_desc = """ """
                    write_provenance_data(op.out_model, generator_args=op,
                                          desc=provenance_desc,
                                          module_name='bayes_traj')

                if op.out_csv is not None:
                    print("Saving data file with trajectory info...")
                    mm.to_df().to_csv(op.out_csv, index=False)
    
                    print("Saving data file provenance info...")
                    provenance_desc = """ """
                    write_provenance_data(op.out_csv, generator_args=op,
                                          desc=provenance_desc,
                                          module_name='bayes_traj')                    
                    
    print("DONE.")

if __name__ == "__main__":
    main()
        

#!/usr/bin/env python

from argparse import ArgumentParser
import pickle
import pandas as pd
import statsmodels.api as sm
import numpy as np
import pdb
from provenance_tools.write_provenance_data import write_provenance_data

def prior_info_from_df(df, target_name, preds, num_trajs, prior_info):
    """
    """
    res_tmp = sm.OLS(df[target_name], df[preds], missing='drop').fit()

    sel_ids = np.zeros(res_tmp.resid.shape[0], dtype=bool)
    sel_ids[0:int(res_tmp.resid.shape[0]/2.)] = True
    
    prec_vec = []
    for i in range(100):
        sel_ids = np.random.permutation(sel_ids)
        prec_vec.append(1./np.var(res_tmp.resid.values[sel_ids]))

    gamma_mean = np.mean(prec_vec)
    gamma_var = np.var(prec_vec)

    prior_info['lambda_b0'][target_name] = gamma_mean/gamma_var
    prior_info['lambda_a0'][target_name] = num_trajs*gamma_mean**2/gamma_var

    samples = np.random.multivariate_normal(res_tmp.params.values,
                                            np.diag(res_tmp.HC0_se.values**2),
                                            10000)

    for (i, m) in enumerate(preds):
        prior_info['w_mu0'][target_name][m] = np.mean(samples, 0)[i]
        prior_info['w_var0'][target_name][m] = np.var(samples, 0)[i]
                
    prior_info['alpha'] = num_trajs/np.log10(df.shape[0])
    
def prior_info_from_df_traj(df_traj, target_name, preds, prior_info,
                            traj_ids=None):
    """ Takes in a dataframe in which each data instance has a trajectory 
    assignment, and from this dataframe and specified target name and 
    predictors, estimates information for the prior. Estimates are made by 
    performing OLS regression within each trajectory subgroup and tallying
    regression coefficients and residuals.

    Parameters
    ----------
    df_traj : pandas dataframe
        Must have as columns 'target_name' and 'preds'. Must also have a 'traj' 
        column as well as columns called 'traj_x', where 'x' is an integer
        indicating a trajectory number. These columns contain the probability 
        that the data instance belongs to that trajectory.

    target_name : str
        Name of the target variable

    preds : list of strings
        The predictor names

    prior_info : dict
        Prior data structure that will be updated by this function

    traj_ids : array of ints, optional
        Subset of trajectories to use for informing prior setting. It may be 
        desirable to use a subset in case some trajectories have been deemed to
        be spurios or unstable
    """
    if traj_ids is None:
        traj_ids = np.array(list(set(df_traj.traj.values)))

    traj_col_names = []
    for i in traj_ids:
        traj_col_names.append('traj_{}'.format(i))
            
    traj_probs = np.sum(df_traj[traj_col_names].values, 0)/\
        np.sum(df_traj[traj_col_names].values)
    
    num_traj_samples = np.random.multinomial(10000, traj_probs)    

    samples = np.zeros([10000, len(preds)])
    prev = 0
    resids = np.zeros(df_traj.shape[0])
    for (i, t) in enumerate(traj_ids):
        ids = (df_traj.traj.values == t) & \
            ~np.isnan(df_traj[target_name].values)
        res_tmp = sm.OLS(df_traj[ids][target_name], df_traj[ids][preds],
                         missing='drop').fit()
        
        samples[prev:np.cumsum(num_traj_samples)[i], :] = \
            np.random.multivariate_normal(res_tmp.params.values,
                                          np.diag(res_tmp.HC0_se.values**2),
                                          num_traj_samples[i])
        prev = np.cumsum(num_traj_samples)[i]
        resids[ids] = res_tmp.resid.values

    for (i, m) in enumerate(preds):
        prior_info['w_mu0'][target_name][m] = np.mean(samples, 0)[i]
        prior_info['w_var0'][target_name][m] = np.var(samples, 0)[i]

    sel_ids = np.zeros(df_traj.shape[0], dtype=bool)
    sel_ids[0:int(df_traj.shape[0]/2.)] = True

    prec_vec = []
    for i in range(100):
        sel_ids = np.random.permutation(sel_ids)
        prec_vec.append(1./np.var(resids[sel_ids]))

    gamma_mean = np.mean(prec_vec)
    gamma_var = np.var(prec_vec)

    prior_info['lambda_b0'][target_name] = gamma_mean/gamma_var
    prior_info['lambda_a0'][target_name] = gamma_mean**2/gamma_var

    prior_info['alpha'] = traj_ids.shape[0]/np.log10(df_traj.shape[0])
    
def prior_info_from_model(target_name, mm, prior_info, traj_ids=None):
    """This function estimates prior values from draws of the specified model
    posterior. Estimates are made for a specific target variable. The assumption
    is the model's predictors and the predictor set for which estimates are
    desired are the same.

    Parameters
    ----------
    target_name : str
        Name of the target variable for which predictor prior values will be 
        estimated.

    mm : MultDPRegression instance
        Trajectory model containing a fit to data. The trajectories in this 
        model will be used to estimate the prior values.

    prior_info : dict
        Prior data structure that will be updated by this function

    traj_ids : array of ints, optional
        Subset of trajectories to use for informing prior setting. It may be 
        desirable to use a subset in case some trajectories have been deemed to
        be spurios or unstable
    """
    assert set(prior_info['w_mu0'][target_name].keys()) == \
        set(mm.predictor_names_), "Predictor name mismatch"

    if traj_ids is None:
        traj_ids = np.where(mm.sig_trajs_)[0]
        
    traj_probs = np.sum(mm.R_, 0)/np.sum(mm.R_)
    num_traj_samples = np.random.multinomial(10000, traj_probs)

    model_target_index = \
        np.where(np.array(mm.target_names_, dtype=str) == target_name)[0][0]
    
    for m in range(mm.M_):
        samples = []
        for t in traj_ids:
            samples.append(mm.w_mu_[m, model_target_index, t] + \
                           np.sqrt(mm.w_var_[m, model_target_index, t])*\
                           np.random.randn(num_traj_samples[t]))

        prior_info['w_mu0'][target_name][mm.predictor_names_[m]] = \
            np.mean(np.hstack(samples))
        prior_info['w_var0'][target_name][mm.predictor_names_[m]] = \
            np.var(np.hstack(samples))  

    # For precision parameters, we'll use a similar sample-based procedure as
    # was done for the coefficients
    samples = []
    for t in traj_ids:
        scale_tmp = 1./mm.lambda_b_[model_target_index, t]
        shape_tmp = mm.lambda_a_[model_target_index, t]
        samples.append(np.random.gamma(shape_tmp, scale_tmp,
                                       num_traj_samples[t]))

    prior_info['lambda_a0'][target_name] = \
        np.mean(np.hstack(samples))**2/np.var(np.hstack(samples))
    prior_info['lambda_b0'][target_name] = \
	np.mean(np.hstack(samples))/np.var(np.hstack(samples))

    if mm.gb_ is not None:
        prior_info['alpha'] = traj_ids.shape[0]/np.log10(mm.gb_.ngroups)
    else:
        prior_info['alpha'] = traj_ids.shape[0]/np.log10(mm.N_)
    

def main():        
    desc = """Generates a pickled file containing Bayesian trajectory prior 
    information"""
    
    parser = ArgumentParser(description=desc)
    parser.add_argument('--preds', help='Comma-separated list of predictor names',
        dest='preds', type=str, default=None)
    parser.add_argument('--targets', help='Comma-separated list of target names',
        dest='targets', type=str, default=None)
    parser.add_argument('--out_file', help='Output (pickle) file that will \
        contain the prior', dest='out_file', type=str, default=None)
    parser.add_argument('-k', help='Number of columns in the truncated assignment \
        matrix', metavar='<int>', default=20)
    parser.add_argument('--tar_resid', help='Use this flag to specify the residual \
        precision mean and variance for the corresponding target value. Specify as \
        a comma-separated tuple: target_name,mean,var. Note that precision is the \
        inverse of the variance.', type=str, default=None, action='append',
        nargs='+')
    parser.add_argument('--coef', help='Coefficient prior for a specified \
        target and predictor. Specify as a comma-separated tuple: \
        target_name,predictor_name,mean,std', type=str,
        default=None, action='append', nargs='+')
    parser.add_argument('--coef_std', help='Coefficient prior standard deviation \
        for a specified target and predictor. Specify as a comma-separated tuple: \
        target_name,predictor_name,std', type=str, default=None,
        action='append', nargs='+')
    parser.add_argument('--in_data', help='If a data file is specified, it will be \
        read in and used to set reasonable prior values using OLS regression. It \
        is assumed that the file contains data columns with names corresponding \
        to the predictor and target names specified on the command line.',
        type=str, default=None)
    parser.add_argument('--num_trajs', help='Rough estimate of the number of \
        trajectories expected in the data set.', type=int, default=2)
    parser.add_argument('--model', help='Pickled bayes_traj model that \
        has been fit to data and from which information will be extracted to \
        produce an updated prior file', type=str, default=None)
    parser.add_argument('--model_trajs', help='Comma-separated list of integers \
        indicating which trajectories to use from the specified model. If a model \
        is not specified, the values specified with this flag will be ignored. If \
        a model is specified, and specific trajectories are not specified with \
        this flag, then all trajectories will be used to inform the prior', \
        default=None)
    parser.add_argument('--groupby', help='Column name in input data file \
        indicating those data instances that must be in the same trajectory. This \
        is typically a subject identifier (e.g. in the case of a longitudinal data \
        set).', dest='groupby', metavar='<string>', default=None)
    
    op = parser.parse_args()
    
    preds = op.preds.split(',')
    targets = op.targets.split(',')
    
    model_trajs = None
    if op.model_trajs is not None:
        model_trajs = np.array(op.model_trajs.split(','), dtype=int)
    
    D = len(targets)
    M = len(preds)
    K = float(op.k)
    
    #-------------------------------------------------------------------------------
    # Initialize prior info
    #-------------------------------------------------------------------------------
    prior_info = {}
    
    prior_info['w_mu0'] = {}
    prior_info['w_var0'] = {}
    prior_info['lambda_a0'] = {}
    prior_info['lambda_b0'] = {}
    
    for tt in targets:
        prior_info['w_mu0'][tt] = {}
        prior_info['w_var0'][tt] = {}
        prior_info['lambda_a0'][tt] = 1
        prior_info['lambda_b0'][tt] = 1        
        
        for pp in preds:
            prior_info['w_mu0'][tt][pp] = 0
            prior_info['w_var0'][tt][pp] = 5
    
    # Guesstimate of how big a data sample. Will be used to generate an estimate of
    # alpha. Will be overwritten if a data file has been specified.
    N = 10000
    prior_info['alpha'] = op.num_trajs/np.log10(N)
    
    #-------------------------------------------------------------------------------
    # Read in and process data and models as availabe
    #-------------------------------------------------------------------------------
    df_traj_model = None
    df_traj_data = None
    df_data = None
    mm = None
    
    if op.model is not None:
        with open(op.model, 'rb') as f:
            mm = pickle.load(f)['MultDPRegression']
    
        df_traj_model = mm.to_df()
        
    if op.in_data is not None:
        df_data = pd.read_csv(op.in_data)
        if mm is not None:
            compute_df_traj_data = False
            if set(mm.predictor_names_) <= set(df_data.columns):
                for tt in mm.target_names_:
                    if tt not in df_data.columns:
                        df_data[tt] = np.nan*np.ones(df_data.shape)
                    else:
                        compute_df_traj_data = True
    
                    if compute_df_traj_data:
                        df_traj_data = \
                            mm.augment_df_with_traj_info(mm.target_names_,
                                                         mm.predictor_names_,
                                                         df_data, op.groupby)
    
    for tt in targets:
        if df_traj_data is not None:
            prior_info_from_df_traj(df_traj_data, tt, preds, prior_info,
                                    model_trajs)
        elif mm is not None:
            if tt in mm.target_names_ and set(mm.predictor_names_) == set(preds):
                prior_info_from_model(tt, mm, prior_info)
            elif tt in df_traj_model.columns and set(preds) <= \
                 set(df_traj_model.columns):
                prior_info_from_df_traj(df_traj_model, tt, preds, prior_info,
                                        model_trajs)
        elif df_data is not None:
            prior_info_from_df(df_data, tt, preds, op.num_trajs, prior_info)
            
    #-------------------------------------------------------------------------------
    # Override prior settings with user-specified preferences
    #-------------------------------------------------------------------------------
    if op.tar_resid is not None:
        for i in range(len(op.tar_resid)):
            tt = op.tar_resid[i][0].split(',')[0]
            mean_tmp = float(op.tar_resid[i][0].split(',')[1])
            var_tmp = float(op.tar_resid[i][0].split(',')[2])        
            assert tt in targets, "{} not among specified targets".format(tt)
                    
            prior_info['lambda_b0'][tt] = mean_tmp/var_tmp
            prior_info['lambda_a0'][tt] = (mean_tmp**2)/var_tmp
    
    if op.coef is not None:
        for i in range(len(op.coef)):
            tt = op.coef[i][0].split(',')[0]
            pp = op.coef[i][0].split(',')[1]
            m = float(op.coef[i][0].split(',')[2])
            s = float(op.coef[i][0].split(',')[3])
    
            assert tt in targets, "{} not among specified targets".format(tt)
            assert pp in preds, "{} not among specified predictors".format(pp)        
    
            prior_info['w_mu0'][tt][pp] = m
            prior_info['w_var0'][tt][pp] = s**2
    
    if op.coef_std is not None:
        for i in range(len(op.coef_std)):
            tt = op.coef_std[i][0].split(',')[0]
            pp = op.coef_std[i][0].split(',')[1]
            s = float(op.coef_std[i][0].split(',')[2])
    
            assert tt in targets, "{} not among specified targets".format(tt)
            assert pp in preds, "{} not among specified predictors".format(pp)        
    
            prior_info['w_var0'][tt][pp] = s**2
    
    #-------------------------------------------------------------------------------
    # Summarize prior info and save to file
    #-------------------------------------------------------------------------------        
    print('---------- Prior Info ----------')
    print('alpha: {}'.format(prior_info['alpha']))        
    for tt in targets:
        print(" ")
        prec_mean = prior_info['lambda_a0'][tt]/prior_info['lambda_b0'][tt]
        prec_var = prior_info['lambda_a0'][tt]/(prior_info['lambda_b0'][tt]**2)
        print("{} residual (precision mean, precision variance): ({}, {})".\
              format(tt, prec_mean, prec_var))
        for pp in preds:
            tmp_mean = prior_info['w_mu0'][tt][pp]
            tmp_std = np.sqrt(prior_info['w_var0'][tt][pp])
            print("{} {} (mean, std): ({}, {})".format(tt, pp, tmp_mean, tmp_std))
            
    if op.out_file is not None:                    
        pickle.dump(prior_info, open(op.out_file, 'wb'))
        desc = """ """
        write_provenance_data(op.out_file, generator_args=op, desc=desc,
                              module_name='bayes_traj')
        
if __name__ == "__main__":
    main()
    

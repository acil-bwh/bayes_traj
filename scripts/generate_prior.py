from argparse import ArgumentParser
import pickle
import pandas as pd
import statsmodels.api as sm
import numpy as np
import pdb
from provenance_tools.provenance_tracker import write_provenance_data

desc = """ """

parser = ArgumentParser(description=desc)
parser.add_argument('--preds', help='Comma-separated list of predictor names',
    dest='preds', type=str, default=None)
parser.add_argument('--targets', help='Comma-separated list of target names',
    dest='targets', type=str, default=None)
parser.add_argument('--out_file', help='Output (pickle) file that will \
    contain the prior', dest='out_file', type=str, default=None)
parser.add_argument('-k', help='Number of columns in the truncated assignment \
    matrix', metavar='<int>', default=20)
parser.add_argument('--resid_std', help='Residual standard deviation prior \
    value for specified target. Specify as a comman-separated tuple:  \
    target_name,value', type=str, dest='resid_std', default=None,
    action='append', nargs='+')
parser.add_argument('--coef', help='Coefficient prior for a specified \
    target and predictor. Specify as a comman-separated tuple: \
    target_name,predictor_name,mean,std', type=str,
    default=None, action='append', nargs='+')
parser.add_argument('--in_data', help='If a data file is specified, it will be \
    read in and used to set reasonable prior values using OLS regression. It \
    is assumed that the file contains data columns with names corresponding \
    to the predictor and target names specified on the command line.',
    type=str, default=None)
parser.add_argument('--num_trajs', help='Rough estimate of the number of \
    trajectories expected in the data set.', type=int, default=2)

op = parser.parse_args()

preds = op.preds.split(',')
targets = op.targets.split(',')

D = len(targets)
M = len(preds)
K = float(op.k)

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

if op.in_data is not None:
    df = pd.read_csv(op.in_data)
    N = df.shape[0]
    for tt in targets:
        res_tmp = sm.OLS(df[tt], df[preds], missing='drop').fit()

        gamma_mean = 1./(np.nanvar(res_tmp.resid)/op.num_trajs)
        gamma_var = 1e-5 # Might want to expose this to user
        prior_info['lambda_b0'][tt] = gamma_mean/gamma_var
        prior_info['lambda_a0'][tt] = gamma_mean**2/gamma_var        
        for pp in preds:
            prior_info['w_mu0'][tt][pp] = res_tmp.params[pp]

            # The following defaults result in a reasonable spread of
            # trajectories in synthetic experiments
            tmp = pp.split('^')
            if len(tmp) > 1:
                prior_info['w_var0'][tt][pp] = 10**(-int(tmp[-1])*5)
            else:
                prior_info['w_var0'][tt][pp] = 1e-5
                
# Generate a rough estimate of alpha
prior_info['alpha'] = op.num_trajs/np.log10(N)
            
if op.resid_std is not None:
    for i in range(len(op.resid_std)):
        tt = op.resid_std[i][0].split(',')[0]
        resid_std_tmp = float(op.resid_std[i][0].split(',')[1])
        assert tt in targets, "{} not among specified targets".format(tt)
                
        gamma_mean = 1./(resid_std_tmp**2)
        gamma_var = 1e-5 # Might want to expose this to user
        prior_info['lambda_b0'][tt] = gamma_mean/gamma_var
        prior_info['lambda_a0'][tt] = gamma_mean**2/gamma_var

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
            
if op.out_file is not None:                    
    pickle.dump(prior_info, open(op.out_file, 'wb'))
    desc = """ """
    write_provenance_data(op.out_file, generator_args=op, desc=desc)

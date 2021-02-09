from argparse import ArgumentParser
import pickle
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
parser.add_argument('--resid_var', help='Residual variance prior value for \
    specified target. Specify as a comman-separated tuple: target_name,value',
                    type=str, dest='resid_var', default=None, action='append', nargs='+')
parser.add_argument('--coef', help='Coefficient prior for a specified \
    target and predictor. Specify as a comman-separated tuple: \
    target_name,predictor_name,mean,std', type=str,
    default=None, action='append', nargs='+')

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
prior_info['alpha'] = 1

for tt in targets:
    prior_info['w_mu0'][tt] = {}
    prior_info['w_var0'][tt] = {}
    prior_info['lambda_a0'][tt] = 1
    prior_info['lambda_b0'][tt] = 1        
    
    for pp in preds:
        prior_info['w_mu0'][tt][pp] = 0
        prior_info['w_var0'][tt][pp] = 5

if op.resid_var is not None:
    for i in range(len(op.resid_var)):
        tt = op.resid_var[i][0].split(',')[0]
        resid_var_tmp = float(op.resid_var[i][0].split(',')[1])
        assert tt in targets, "{} not among specified targets".format(tt)
                
        gamma_mean = 1./resid_var_tmp
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
    write_provenance_data(op.prior, generator_args=op, desc=desc)

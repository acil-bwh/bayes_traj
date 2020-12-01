from argparse import ArgumentParser
import pandas as pd
import numpy as np
import networkx as nx
from bayes_traj.mult_dp_regression import MultDPRegression
from bayes_traj.prior_from_model import prior_from_model
from bayes_traj.utils import sample_cos
from bayes_traj.get_longitudinal_constraints_graph \
  import get_longitudinal_constraints_graph
from provenance_tools.provenance_tracker import write_provenance_data
import pdb, pickle, sys, warnings

np.set_printoptions(precision = 1, suppress = True, threshold=1e6,
                    linewidth=300)

desc = """Runs MultDPRegression on the specified data file with the specified
predictors and target variables"""

parser = ArgumentParser(description=desc)
parser.add_argument('--in_csv', help='Input csv file containing data on which \
  to run MultDPRegression', dest='in_csv', metavar='<string>', default=None)
parser.add_argument('--prior_p', help='Input pickle file containing prior \
  settings', dest='prior_p', metavar='<string>', default=None)
parser.add_argument('--alpha', help='If specified, over-rides the value in the \
  prior file', dest='alpha', metavar=float, default=None)
parser.add_argument('--preds', help='Comma-separated list of predictor names',
    dest='preds', metavar='<string>', default=None)
parser.add_argument('--targets', help='Comma-separated list of target names',
    dest='targets', metavar='<string>', default=None)
parser.add_argument('--out_file', help='Pickle file name to which to dump the \
    result', dest='out_file', metavar='<string>', default=None)
parser.add_argument('--iters', help='Number of iterations per repeat attempt',
    dest='iters', metavar='<int>', default=100)
parser.add_argument('--repeats', help='Number of repeats to attempt',
    dest='repeats', metavar='<int>', default=100)
parser.add_argument('-k', help='Number of columns in the truncated assignment \
    matrix', metavar='<int>', default=20)
parser.add_argument('--waic2_thresh', help='Model will only be written to \
    file provided that the WAIC2 value is below this threshold',
    dest='waic2_thresh', metavar='<float>', type=float,
    default=sys.float_info.max)
parser.add_argument('--bic_thresh', help='Model will only be written to \
    file provided that BIC values are above this threshold',
    dest='bic_thresh', metavar='<float>', type=float,
    default=-sys.float_info.max)
parser.add_argument("--save_all", help="By default, only the model with the \
    lowest WAIC score is saved to file. However, if this flag is set a model \
    file is saved for each repeat. The specified output file name is used \
    with a 'repeat[n]' appended, where [n] indicates the repeat number.",
    action="store_true")
parser.add_argument("--verbose", help="Display per-trajectory counts during \
    optimization", action="store_true")
parser.add_argument("--constraints", help="File name of pickled networkx \
    pairwise constraints to impose during model fitting", dest='constraints',
    default=None)
parser.add_argument('--probs_weight', help='Value between 0 and 1 that \
    controls how much weight to assign to traj_probs, the marginal \
    probability of observing each trajectory. This value is only meaningful \
    if traj_probs has been set in the input prior file. Otherwise, it has no \
    effect. Higher values place more weight on the model-derived probabilities \
    and reflect a stronger belief in those assignment probabilities.',
    dest='probs_weight', metavar='<float>', type=float, default=None)

op = parser.parse_args()
iters = int(op.iters)
repeats = int(op.repeats)
preds =  op.preds.split(',')
targets = op.targets.split(',')
in_csv = op.in_csv
prior_p = op.prior_p
out_file = op.out_file
probs_weight = op.probs_weight

if probs_weight is not None:
    assert probs_weight >=0 and probs_weight <= 1, \
        "Invalide probs_weight value"

df = pd.read_csv(in_csv)
if 'sid' not in df.columns:
    df['sid'] = [n.split('_')[0] for n in df.data_names.values]

ids = ~np.isnan(np.sum(df[preds].values, 1))
if np.sum(~ids) > 0:
    print("Warning: identified NaNs in predictor set for {} individuals. \
    Proceeding with non-NaN data".format(np.sum(~ids)))
X = df[preds].values[ids]
Y = df[targets].values[ids]

data_names = df.data_names.values[ids]
longitudinal_constraints = \
    get_longitudinal_constraints_graph(df.sid.values[ids])
if op.constraints is not None:
    input_constraints = pickle.load(open(op.constraints, 'r'))['Graph']
    constraints_graph = nx.compose(input_constraints, longitudinal_constraints)
else:
    constraints_graph = longitudinal_constraints

D = len(targets)
M = len(preds)
K = int(op.k)
                
prior_info = {}
for i in ['v_a', 'v_b', 'w_mu', 'w_var', 'lambda_a', 'lambda_b', 'traj_probs',
          'probs_weight', 'w_mu0', 'w_var0', 'lambda_a0', 'lambda_b0',
          'alpha']:
    prior_info[i] = None

#------------------------------------------------------------------------------
# Get priors from file
#------------------------------------------------------------------------------
if prior_p is not None:
    with open(prior_p, 'rb') as f:
        file_data = pickle.load(f)

        for key in file_data.keys():
            prior_info[key] = file_data[key]

#------------------------------------------------------------------------------
# Randomly sample trajectory coefficients if they are not already set in the
# prior
#------------------------------------------------------------------------------            
for k in range(K):
    for d in range(D):
        if np.isnan(np.sum(prior_info['w_mu'][:, d, k])):
            prior_info['w_mu'][:, d, k] = \
                sample_cos(file_data['w_mu0'],
                           file_data['w_var0'])[:, d, 0]
                            
if op.alpha is not None:
    prior_info['alpha'] = float(op.alpha)
    
#------------------------------------------------------------------------------
# Set up and run the traj alg
#------------------------------------------------------------------------------
waics_tracker = []
bics_tracker = []
num_tracker = []
best_mm = None
best_waic2 = op.waic2_thresh
best_bics = (op.bic_thresh, op.bic_thresh)
for r in np.arange(repeats):    
    print("---------- Repeat {}, Best BICs: {}, {} ----------".\
      format(r, best_bics[0], best_bics[1]))
    mm = MultDPRegression(prior_info['w_mu0'], prior_info['w_var0'],
                          prior_info['lambda_a0'], prior_info['lambda_b0'],
                          prior_info['alpha'], K=K)
    mm.fit(X, Y, iters=iters, verbose=op.verbose,
           constraints=constraints_graph, data_names=data_names,
           target_names=targets, predictor_names=preds,
           traj_probs=prior_info['traj_probs'], traj_probs_weight=probs_weight,
           v_a=prior_info['v_a'], v_b=prior_info['v_b'], w_mu=prior_info['w_mu'],
           w_var=prior_info['w_var'], lambda_a=prior_info['lambda_a'],
           lambda_b=prior_info['lambda_b'])

    if op.save_all:
        out_file_tmp = out_file.split('.')[0] + '_repeat{}.p'.format(r)
        pickle.dump({'MultDPRegression': mm}, open(out_file_tmp, 'wb'))

        provenance_desc = """ """
        write_provenance_data(out_file_tmp, generator_args=op,
                              desc=provenance_desc)
    else:
        bics = mm.bic()
        
        bics_tracker.append(bics)
        waics_tracker.append(mm.compute_waic2())
        num_tracker.append(np.sum(mm.sig_trajs_))
        
        if bics[0] > best_bics[0] and bics[1] > best_bics[1]:
            best_bics = bics

            pickle.dump({'MultDPRegression': mm}, open(out_file, 'wb'))

            provenance_desc = """ """
            write_provenance_data(out_file, generator_args=op,
                                  desc=provenance_desc)
            
        #waic2 = mm.compute_waic2()
        #waics_tracker.append(waic2)
        #if waic2 < best_waic2:
        #    best_waic2 = waic2
        #    best_mm = mm
        #    pickle.dump({'MultDPRegression': mm}, open(out_file, 'wb'))
        # 
        #    provenance_desc = """ """
        #    write_provenance_data(out_file, generator_args=op,
        #                          desc=provenance_desc)


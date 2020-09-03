from argparse import ArgumentParser
import pandas as pd
import numpy as np
import networkx as nx
from bayes_traj.mult_dp_regression import MultDPRegression
from bayes_traj.get_longitudinal_constraints_graph \
  import get_longitudinal_constraints_graph
from provenance_tools.provenance_tracker import write_provenance_data
import pdb, pickle, sys

np.set_printoptions(precision = 3, suppress = True, threshold=1e6,
                    linewidth=200)

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
parser.add_argument("--save_all", help="By default, only the model with the \
    lowest WAIC score is saved to file. However, if this flag is set a model \
    file is saved for each repeat. The specified output file name is used \
    with a 'repeat[n]' appended, where [n] indicates the repeat number.",
    action="store_true")
parser.add_argument("--constraints", help="File name of pickled networkx \
    pairwise constraints to impose during model fitting", dest='constraints',
    default=None)

op = parser.parse_args()
iters = int(op.iters)
repeats = int(op.repeats)
preds =  op.preds.split(',')
targets = op.targets.split(',')
in_csv = op.in_csv
prior_p = op.prior_p
out_file = op.out_file

df = pd.read_csv(in_csv)

ids = ~np.isnan(np.sum(df[preds].values, 1))
print("Warning: identified NaNs in predictor set for {} individuals. \
Proceeding with non-NaN data".format(np.sum(~ids)))
X = df[preds].values[ids]
Y = df[targets].values[ids]

data_names = df.data_names.values[ids]
longitudinal_constraints = get_longitudinal_constraints_graph(df.sid.values[ids])
if op.constraints is not None:
    input_constraints = pickle.load(open(op.constraints, 'r'))['Graph']
    constraints_graph = nx.compose(input_constraints, longitudinal_constraints)
else:
    constraints_graph = longitudinal_constraints
    
D = len(targets)
M = len(preds)
K = int(op.k)

#------------------------------------------------------------------------------
# Get priors from file
#------------------------------------------------------------------------------
with open(prior_p, 'r') as f:
    priors = pickle.load(f)
    w_mu0 = priors['w_mu0']
    w_var0 = priors['w_var0']
    lambda_a0 = priors['lambda_a0']
    lambda_b0 = priors['lambda_b0']
    alpha = priors['alpha']

if op.alpha is not None:
    alpha = float(op.alpha)
    
#------------------------------------------------------------------------------
# Set up and run the traj alg
#------------------------------------------------------------------------------
waics_tracker = []
bics_tracker = []
best_mm = None
best_waic2 = op.waic2_thresh
best_bics = (-sys.float_info.max, -sys.float_info.max)
for r in np.arange(repeats):
    print "---------- Repeat {}, Best BICs: {}, {} ----------".\
      format(r, best_bics[0], best_bics[1])
    mm = MultDPRegression(w_mu0, w_var0, lambda_a0, lambda_b0, alpha, K=K)
    mm.fit(X, Y, iters=iters, verbose=True,
           constraints=constraints_graph, data_names=data_names,
           target_names=targets, predictor_names=preds)

    if op.save_all:
        out_file_tmp = out_file.split('.')[0] + '_repeat{}.p'.format(r)
        pickle.dump({'MultDPRegression': mm}, open(out_file_tmp, 'wb'))

        provenance_desc = """ """
        write_provenance_data(out_file_tmp, generator_args=op,
                              desc=provenance_desc)
    else:
        bics = mm.bic()
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


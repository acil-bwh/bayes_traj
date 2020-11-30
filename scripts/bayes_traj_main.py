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
parser.add_argument("--in_model", help="File name of pickled MultDPRegression \
    instance that has been fit to data. This model need not have been fit on \
    the supplied data (indeed, the intent is to use this model to inform the \
    current data fitting). It is assumed that the predictor names stored in \
    this model and supplied here on the command line are identical. \
    If a prior is specified with the prior_p flag, it will be used to set \
    prior params. If a model is additionally specified, relevent info the 
    model will override corresponding values already set by the prior. \
    Additionally, the w_mu_, w_var_, lambda_a_, lambda_b_, v_a_, and v_b_ \
    parameters will be initialized with the specified model: trajectories with \
    non-zero probability will be initialized with the stored parameter \
    settings; trajectories with zero probability will be initialized with the \
    prior values. Additionally, the input data will be used to compute \
    probability of membership for each trajectory in the model file. This \
    probability matrix will be blended with a randomly generated probability \
    matrix with the value specified using the probs_weight flag.",
    dest='in_model', default=None)
parser.add_argument('--probs_weight', help='Value between 0 and 1 that \
    controls how much weight to assign to the per-individual trajectory \
    assignment probabilities derived from the input model (specified with \
    the --in_model flag), as opposed to random initialization. Higher values \
    place more weight on the model-derived probabilities and reflect a \
    stronger belief in those assignment probabilities.', dest='probs_weight', \
    metavar='<float>', type=float, default=None)

op = parser.parse_args()
iters = int(op.iters)
repeats = int(op.repeats)
preds =  op.preds.split(',')
targets = op.targets.split(',')
in_csv = op.in_csv
prior_p = op.prior_p
out_file = op.out_file
in_model = op.in_model
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
                
v_a = None
v_b = None
w_mu = None
w_var = None
lambda_a = None
lambda_b = None
R_blend = None
traj_probs = None

#------------------------------------------------------------------------------
# Get priors from file
#------------------------------------------------------------------------------
if prior_p is not None:
    with open(prior_p, 'rb') as f:
        priors = pickle.load(f)
        w_mu0 = priors['w_mu0']
        w_var0 = priors['w_var0']
        lambda_a0 = priors['lambda_a0']
        lambda_b0 = priors['lambda_b0']
        alpha = priors['alpha']

        lambda_a = np.zeros([D, K])
        for k in range(0, K):
            lambda_a[:, k] = lambda_a0

        lambda_b = np.zeros([D, K])
        for k in range(0, K):
            lambda_b[:, k] = lambda_b0

        w_mu = np.zeros([M, D, K])
        for k in range(0, K):
            w_mu[:, :, k] = sample_cos(w_mu0, w_var0)[:, :, 0]
                
        w_var = np.zeros([M, D, K])
        for k in range(0, K):
            w_var[:, :, k] = w_var0
        
#------------------------------------------------------------------------------
# Read input model if specified
#------------------------------------------------------------------------------
if in_model is not None:
    if lambda_a is None:
        lambda_a = np.zeros([D, K])

    if lambda_b is None:
        lambda_b = np.zeros([D, K])

    if w_mu is None:
        w_mu = np.zeros([M, D, K])

    if w_var is None:
        w_var = np.zeros([M, D, K])
    
    with open(in_model, 'rb') as f:
        mm_fit = pickle.load(f)['MultDPRegression']

        # First check that stored predictor names match with those supplied at
        # command line
        for p in preds:
            assert p in mm_fit.predictor_names_, \
                "Predictor name mismatch with supplied model"
        for p in mm_fit.predictor_names_:
            assert p in preds, \
                "Predictor name mismatch with supplied model"            

        assert K == mm_fit.K_, "K mismatch with supplied model"

        # Set the prior using fit model trajectories ordered from most probable
        # to least probable
        ordered_indices = np.flip(np.argsort(np.sum(mm_fit.R_, 0)))

        # The prior will be generated from the posterior in the model
        prior = prior_from_model(mm_fit)
        alpha = prior['alpha']
        traj_probs = prior['traj_probs'][ordered_indices]

        v_a = np.array(mm_fit.v_a_[ordered_indices])
        v_b = np.array(mm_fit.v_b_[ordered_indices])
        
        for inc, target in enumerate(targets):
            if target not in mm_fit.target_names_:
                continue
            
            target_index = np.where(np.array(mm_fit.target_names_) == \
                                    target)[0][0]
            
            w_mu0[:, inc] = prior['w_mu0'][:, target_index]
            w_var0[:, inc] = prior['w_var0'][:, target_index]
            lambda_a0[inc] = prior['lambda_a0'][target_index]
            lambda_b0[inc] = prior['lambda_b0'][target_index]
    
            lambda_a[inc, :] = \
                np.array(mm_fit.lambda_a_[:, ordered_indices])[target_index, :]
            lambda_b[inc, :] = \
                np.array(mm_fit.lambda_b_[:, ordered_indices])[target_index, :]
            
            w_mu[:, inc, :] = np.array(mm_fit.w_mu_[:, :, ordered_indices])\
                [:, target_index, :]
            w_var[:, inc, :] = np.array(mm_fit.w_var_[:, :, ordered_indices])\
                [:, target_index, :]
    
        # The trajectories with zero probability are meaningless, so just set
        # these to the prior values. Note that we're assuming a re-ordering of
        # trajectories that puts all the non-zero trajectories first
        for k in range(np.sum(mm_fit.sig_trajs_), K):
            lambda_a[:, k] = np.array(lambda_a0)
            lambda_b[:, k] = np.array(lambda_b0)   
            w_mu[:, :, k] = sample_cos(w_mu0, w_var0)[:, :, 0]
            w_var[:, :, k] = np.array(w_var0)
                    
if op.alpha is not None:
    alpha = float(op.alpha)
    
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

    # traj_probs will only be non-None if we have read in a previously fit.
    # model. If that's the case, we want to preserve the coefficients describing
    # the trajectories that were discovered in that model (set above), but we'll
    # want to randomly sample from the prior for the other trajectory bins.
    if traj_probs is not None:
        for k in range(0, K):
            if k in np.where(traj_probs == 0)[0]:
                w_mu[:, :, k] = sample_cos(w_mu0, w_var0)[:, :, 0]
    
    print("---------- Repeat {}, Best BICs: {}, {} ----------".\
      format(r, best_bics[0], best_bics[1]))
    mm = MultDPRegression(w_mu0, w_var0, lambda_a0, lambda_b0, alpha, K=K)
    mm.fit(X, Y, iters=iters, verbose=op.verbose,
           constraints=constraints_graph, data_names=data_names,
           target_names=targets, predictor_names=preds,
           traj_probs=traj_probs, traj_probs_weight=probs_weight,
           v_a=v_a, v_b=v_b, w_mu=w_mu, w_var=w_var,
           lambda_a=lambda_a, lambda_b=lambda_b)

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


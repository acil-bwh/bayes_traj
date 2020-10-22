from argparse import ArgumentParser
import pandas as pd
import numpy as np
import networkx as nx
from bayes_traj.mult_dp_regression import MultDPRegression
from bayes_traj.prior_from_model import prior_from_model
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
parser.add_argument("--verbose", help="Display per-trajectory counts during \
    optimization", action="store_true")
parser.add_argument("--constraints", help="File name of pickled networkx \
    pairwise constraints to impose during model fitting", dest='constraints',
    default=None)
parser.add_argument("--in_model", help="File name of pickled MultDPRegression \
    instance that has been fit to data. This model need not have been fit on \
    the supplied data (indeed, the intent is to use this model to inform the \
    current data fitting). It is assumed that the predictor and target names \
    stored in this model and supplied here on the command line are identical. \
    If a model is specified, a prior (using the prior_p flag) need not be \
    specified. However, if a prior is specified with the prior_p flag, it \
    will take precedence. Otherwise, a prior will be generated from the \
    specified model. Additionally, the w_mu_, w_var_, lambda_a_, lambda_b_, \
    v_a_, and v_b_ parameters will be initialized with the specified model: \
    trajectories with non-zero probability will be initialized with the \
    stored parameter settings; trajectories with zero probability will be \
    initialized with the prior values. Additionally, the input data will be \
    used to compute probability of membership for each trajectory in the \
    model file. This probability matrix will be blended with a randomly \
    generated probability matrix with the value specified using the z_blend \
    flag.", dest='in_model', default=None)
parser.add_argument('--z_blend', help='Value between 0 and 1 that controls \
    how much weight to assign to the per-individual trajectory assignment \
    probabilities derived from the input model (specified with the --in_model \
    flag), as opposed to random initialization. Higher values place more \
    weight on the model-derived probabilities and reflect a stronger belief \
    in those assignment probabilities.', dest='z_blend', metavar='<float>',
    type=float, default=0.5)

op = parser.parse_args()
iters = int(op.iters)
repeats = int(op.repeats)
preds =  op.preds.split(',')
targets = op.targets.split(',')
in_csv = op.in_csv
prior_p = op.prior_p
out_file = op.out_file
in_model = op.in_model
z_blend = op.z_blend

if z_blend is not None:
    assert z_blend >=0 and z_blend <= 1, "Invalide z_blend value"

df = pd.read_csv(in_csv)

ids = ~np.isnan(np.sum(df[preds].values, 1))
if np.sum(~ids) > 0:
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
                
v_a = None
v_b = None
w_mu = None
w_var = None
lambda_a = None
lambda_b = None
R_blend = None

#------------------------------------------------------------------------------
# Read input model if specified
#------------------------------------------------------------------------------
if in_model is not None:
    with open(in_model, 'rb') as f:
        mm_fit = pickle.load(f)
        
        # First check that stored predictor names and target names match with
        # those supplied at command line
        for t in targets:
            assert t in mm_fit.target_names_, \
                "Target name mismatch with supplied model"
        for t in mm_fit.target_names_:
            assert t in targets, \
                "Target name mismatch with supplied model"
        for p in predictors:
            assert p in mm_fit.predictor_names_, \
                "Predictor name mismatch with supplied model"
        for p in mm_fit.predictor_names_:
            assert p in predictors, \
                "Predictor name mismatch with supplied model"            

        assert K == mm_fit.K_, "K mismatch with supplied model"
            
        # The prior will be generated from the posterior in the model
        prior = prior_from_model(mm_fit)
        w_mu0 = prior['w_mu0']
        w_var0 = prior['w_var0']
        lambda_a0 = prior['lambda_a0']
        lambda_b0 = prior['lambda_b0']
        alpha = prior['alpha']

        # Use the fit model to predict trajectory membership probabilities for
        # the input data. These predictions will be blended with a randomly
        # generated initialization matrix according to user-specified blending
        # term.
        X_tmp = df[mm_fit.predictor_names_].values[ids]
        Y_tmp = df[mm_fit.target_names_].values[ids]
        R_pred = mm_fit.predict_proba(X_tmp, Y_tmp, constraints_graph)

        lambda_a = np.zeros([D, K])
        lambda_b = np.zeros([D, K])
        w_mu = np.zeros([M, D, K])
        v_a = np.ones(K)
        v_b = alpha_*np.ones(K)
        for k in range(0, K):
            if k in np.where(mm_fit.sig_trajs_)[0]:
                lambda_a[:, k] = mm_fit.lambda_a_[:, k]
                lambda_b[:, k] = mm_fit.lambda_b_[:, k]
                w_mu[:, :, k] = mm_fit.w_mu_[:, :, k]
                w_var[:, :, k] = mm_fit.w_var_[:, :, k]
                v_a[k] = mm_fit.v_a_[k]
                v_b[k] = mm_fit.v_b_[k]
            else:
                lambda_a[:, k] = lambda_a0
                lambda_b[:, k] = lambda_b0   
                w_mu[:, :, k] = w_mu0
                w_var[:, :, k] = w_var0

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
best_bics = (-sys.float_info.max, -sys.float_info.max)
for r in np.arange(repeats):    
    print("---------- Repeat {}, Best BICs: {}, {} ----------".\
      format(r, best_bics[0], best_bics[1]))
    mm = MultDPRegression(w_mu0, w_var0, lambda_a0, lambda_b0, alpha, K=K)
    mm.fit(X, Y, iters=iters, verbose=op.verbose,
           constraints=constraints_graph, data_names=data_names,
           target_names=targets, predictor_names=preds, R=R_pred,
           R_blend=z_blend, v_a=v_a, v_b=v_b, w_mu=w_mu, w_var=w_var,
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


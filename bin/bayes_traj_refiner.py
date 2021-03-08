from argparse import ArgumentParser
import numpy as np
from bayes_traj.mult_dp_regression import MultDPRegression
from bayes_traj.get_longitudinal_constraints_graph \
  import get_longitudinal_constraints_graph
#from bayes_traj.get_constraints_graph import get_constraints_graph
from provenance_tools.provenance_tracker import write_provenance_data
import pdb, pickle, sys, git, os

np.set_printoptions(precision = 1, suppress = True, threshold=1e6,
                    linewidth=300)

desc = """Reads an instance of MultDPRegression and performs additional \
iterations in order to refine the model"""

parser = ArgumentParser(description=desc)
parser.add_argument('--in_p', help='Input pickle file containing instance of \
  MultDPRegression to refine', dest='in_p', metavar='<string>', default=None)
parser.add_argument('--out_file', help='Pickle file name to which to dump the \
  refined instance of MultDPRegression', dest='out_file', metavar='<string>',
  default=None)
parser.add_argument('--iters', help='Number of iterations to refine',
    dest='iters', metavar='<int>', default=100)

op = parser.parse_args()
iters = int(op.iters)
in_p = op.in_p
out_file = op.out_file

mm = pickle.load(open(in_p, 'rb'))['MultDPRegression']

#------------------------------------------------------------------------------
# Set up and run the traj alg
#------------------------------------------------------------------------------
mm.fit(mm.X_, mm.Y_, iters=iters, R=mm.R_, v_a=mm.v_a_, v_b=mm.v_b_,
  w_mu=mm.w_mu_, w_var=mm.w_var_, lambda_a=mm.lambda_a_, lambda_b=mm.lambda_b_,
  constraints=mm.constraints_, data_names=mm.data_names_,
  target_names=mm.target_names_, predictor_names=mm.predictor_names_,
  verbose=True)

with open(out_file, 'wb') as f:
    pickle.dump({'MultDPRegression': mm}, f)

write_provenance_data(out_file, generator_args=op)

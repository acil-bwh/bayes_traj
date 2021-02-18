import numpy as np
import pandas as pd
import pdb
import pickle
from bayes_traj.utils import *
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from provenance_tools.provenance_tracker import write_provenance_data

desc = """Produces a scatter plot of the data contained in the input data file
as well as plots of random draws from the prior. This is useful to inspect 
whether the prior appropriately captures prior belief."""

parser = ArgumentParser(description=desc)
parser.add_argument('--data_file', help='Input data file', type=str,
    default=None)
parser.add_argument('--prior', help='Input prior file', type=str, default=None)
parser.add_argument('--num_draws', help='Number of random draws to take from \
    prior', type=int, default=10)
parser.add_argument('--y_axis', help='Name of the target variable that will \
    be plotted on the y-axis', type=str, default=None)
parser.add_argument('--x_axis', help='Name of the predictor variable that will \
    be plotted on the x-axis', type=str, default=None)

op = parser.parse_args()

df = pd.read_csv(op.data_file)

if op.prior is not None:
    with open(op.prior, 'rb') as f:
        prior_file_info = pickle.load(f)

        targets = get_target_names_from_prior_info(prior_file_info)
        preds = get_pred_names_from_prior_info(prior_file_info)
        
        D = len(targets)
        M = len(preds)
        
        prior_data = {}
        prior_data['w_mu0'] = np.zeros([M, D])
        prior_data['w_var0'] = np.ones([M, D])
        prior_data['lambda_a0'] = np.ones([D])
        prior_data['lambda_b0'] = np.ones([D])
        
        for (d, target) in enumerate(targets):
            prior_data['lambda_a0'][d] = prior_file_info['lambda_a0'][target]
            prior_data['lambda_b0'][d] = prior_file_info['lambda_b0'][target]            
            
            for (m, pred) in enumerate(preds):
                prior_data['w_mu0'][m, d] = prior_file_info['w_mu0'][target][pred]
                prior_data['w_var0'][m, d] = prior_file_info['w_var0'][target][pred]
                
plt.scatter(df[op.x_axis].values, df[op.y_axis].values)

num_dom_locs = 100
x_dom = np.linspace(np.min(df[op.x_axis].values), np.max(df[op.x_axis].values),
                    num_dom_locs)

for nn in range(op.num_draws):
    target_index = np.where(np.array(targets) == op.y_axis)[0][0]    
    scale = 1./prior_data['lambda_b0'][target_index]
    shape = prior_data['lambda_a0'][target_index]
    std = np.sqrt(1./np.random.gamma(shape, scale, size=1))
    
    co = sample_cos(prior_data['w_mu0'],
                    prior_data['w_var0'])[:, target_index, 0]

    X_tmp = np.ones([num_dom_locs, M])
    for (inc, pp) in enumerate(preds):
        if pp == 'intercept':
            pass
        else:
            tmp = pp.split('^')
            
            if len(tmp) > 1:
                if op.x_axis in tmp:                
                    X_tmp[:, inc] = x_dom**(int(tmp[-1]))
                else:                
                    X_tmp[:, inc] = np.mean(df[tmp[0]].values)**(int(tmp[-1]))
            elif pp == op.x_axis:
                X_tmp[:, inc] = x_dom
            else:
                X_tmp[:, inc] = np.mean(df[tmp[0]].values)
                
    y_tmp = np.dot(co, X_tmp.T)

    plt.plot(x_dom, y_tmp)
    plt.fill_between(x_dom, y_tmp-2*std, y_tmp+2*std, alpha=0.3)

plt.xlabel(op.x_axis)
plt.ylabel(op.y_axis)
plt.show()






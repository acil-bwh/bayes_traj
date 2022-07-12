#!/usr/bin/env python

from argparse import ArgumentParser
import numpy as np
from scipy.special import gamma
from scipy.special import loggamma
import pickle, pdb
from provenance_tools.write_provenance_data import write_provenance_data
import matplotlib.pyplot as plt

def main():
    desc = """This utility generates plots of gamma distributions and can be 
    useful for inspecting prior and posterior distributions over residual
    precisions."""
    
    parser = ArgumentParser(description=desc)
    parser.add_argument('--model', help='Model from which to extract precision \
        prior and posterior info', type=str, default=None)
    parser.add_argument('--target', help='Model target variable for which to \
        plot prior and posteriors. Only relevant if model is also specified.',
        type=str, default=None)
    parser.add_argument('--min_traj_prob', help='Only trajectories that make \
        up at least this fraction of the data sample will be considered',
        type=float, default=0.05)
    parser.add_argument('--info', help='Comma-separated list of gamma \
        distribution info. Specify as mean,variance[,label], where label is an \
        optional text string that will be used to labeld the plot. This flag \
        can be called multiple times', type=str, default=None, action='append',
        nargs='+')
    parser.add_argument('--min', help='Min value on x-axis to plot', type=float,
        default=0.001)
    parser.add_argument('--max', help='Max value on x-axis to plot', type=float,
        default=20)
    parser.add_argument('--title', help='Title to add to plot', type=str,
        default=None)
    
    op = parser.parse_args()
    
    dist_info_list = []
    
    if op.model is not None:
        mm = pickle.load(open(op.model, 'rb'))['MultDPRegression']
        if op.target is None:
            target = mm.target_names_[0]
        else:
            target = op.target
    
        target_index = np.where(np.array(mm.target_names_) == target)[0][0]

        #prec_prior_scale = mm.prec_prior_weight_*\
        #    (mm.gb_.ngroups if mm.gb_ is not None else mm.N_)
        
        #lambda_a0 = mm.lambda_a0_[target_index]/prec_prior_scale
        #lambda_b0 = mm.lambda_b0_[target_index]/prec_prior_scale

        lambda_a0 = mm.lambda_a0_[target_index]
        lambda_b0 = mm.lambda_b0_[target_index]
        
        m = lambda_a0/lambda_b0
        v = lambda_a0/(lambda_b0**2)

        dist_info = [m, v, 'Prior']
        dist_info_list.append(dist_info)
            
        df_traj = mm.to_df()
        
        # Get dataframe column that was used to create groups, if groups exist
        if mm.gb_ is not None:
            num_groups = mm.gb_.ngroups
            groupby_col = mm.gb_.count().index.name               
        else:
            num_groups = df_traj.shape[0]    
    
        for traj in np.where(mm.sig_trajs_)[0]:
            num_obs_in_traj = sum(df_traj.traj.values == traj)
            if mm.gb_ is not None:
                num_groups_in_traj = df_traj[df_traj.traj.values == traj].\
                    groupby(groupby_col).ngroups
            else:
                num_groups_in_traj = num_obs_in_traj
            
            frac = num_groups_in_traj/num_groups
            if frac > op.min_traj_prob:
                m = mm.lambda_a_[target_index, traj]/\
                    mm.lambda_b_[target_index, traj]
                v = mm.lambda_a_[target_index, traj]/\
                    (mm.lambda_b_[target_index, traj]**2)
                dist_info = [m, v, 'Traj {}'.format(traj)]
                dist_info_list.append(dist_info)

    if op.info is not None:
        for ii in op.info:
            info = ii[0].split(',')
            m = float(info[0])
            v = float(info[1])
            if len(info) == 3:
                label = info[2]
            else:
                label = "Mean: {}, Var: {}".format(m, v)
    
            dist_info_list.append([m, v, label])
    
    for info in dist_info_list:
        m = info[0]
        v = info[1]
        label = info[2]

        lambda_a = m*m/v
        lambda_b = m/v
    
        dom = np.linspace(op.min, op.max, 5000)
        
        dens = np.exp(-loggamma(lambda_a) + lambda_a*np.log(lambda_b) + \
                      (lambda_a-1)*np.log(dom) - lambda_b*dom)

        plt.plot(dom, dens, label=label)
        plt.fill_between(dom, dens, np.zeros(5000), alpha=0.3)    
    
    if op.title is not None:
        plt.title(op.title)
    plt.legend()
    plt.show()    

if __name__ == "__main__":
    main()      

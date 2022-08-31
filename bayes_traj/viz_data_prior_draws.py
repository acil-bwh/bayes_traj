#!/usr/bin/env python

import numpy as np
import pandas as pd
import pdb
import pickle
from bayes_traj.utils import *
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from provenance_tools.write_provenance_data import write_provenance_data

def main():
    desc = """Produces a scatter plot of the data contained in the input data 
    file as well as plots of random draws from the prior. This is useful to 
    inspect whether the prior appropriately captures prior belief."""
    
    parser = ArgumentParser(description=desc)
    parser.add_argument('--data_file', help='Input data file', type=str,
        default=None)
    parser.add_argument('--prior', help='Input prior file', type=str,
        default=None)
    parser.add_argument('--num_draws', help='Number of random draws to take \
        from prior', type=int, default=10)
    parser.add_argument('--y_axis', help='Name of the target variable that \
        will be plotted on the y-axis', type=str, default=None)
    parser.add_argument('--y_label', help='Label to display on y-axis. If none \
        given, the variable name specified with the y_axis flag will be used.',
        type=str, default=None)
    parser.add_argument('--x_axis', help='Name of the predictor variable that \
        will be plotted on the x-axis', type=str, default=None)
    parser.add_argument('--x_label', help='Label to display on x-axis. If none \
        given, the variable name specified with the x_axis flag will be used.',
        type=str, default=None)
    parser.add_argument('--ylim', help='Comma-separated tuple to set the \
        limits of display for the y-axis', type=str, default=None)    
    parser.add_argument('--hide_resid', help='If set, shaded regions \
        corresponding to residual spread will not be displayed. This can be \
        useful to reduce visual clutter. Only relevant for continuous target \
        variables.', action='store_true')    
    parser.add_argument('--fig_file', help='File name where figure will be \
        saved', type=str, default=None)
    
    op = parser.parse_args()
    
    df = pd.read_csv(op.data_file)

    nonnan_ids = ~np.isnan(df[op.y_axis].values)
    target_type = 'gaussian'
    if set(df[op.y_axis].values[nonnan_ids]).issubset({1.0, 0.0}):
        target_type = 'binary'        
    
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
                prior_data['lambda_a0'][d] = \
                    prior_file_info['lambda_a0'][target]
                prior_data['lambda_b0'][d] = \
                    prior_file_info['lambda_b0'][target]            
                
                for (m, pred) in enumerate(preds):
                    prior_data['w_mu0'][m, d] = \
                        prior_file_info['w_mu0'][target][pred]
                    prior_data['w_var0'][m, d] = \
                        prior_file_info['w_var0'][target][pred]
    
    fig, ax = plt.subplots(figsize=(8, 8))                
    ax.scatter(df[op.x_axis].values, df[op.y_axis].values, facecolor='none',
                edgecolor='k', alpha=0.2)
    
    num_dom_locs = 100
    x_dom = np.linspace(np.nanmin(df[op.x_axis].values), \
                        np.nanmax(df[op.x_axis].values), num_dom_locs)
    
    for nn in range(op.num_draws):
        target_index = np.where(np.array(targets) == op.y_axis)[0][0]
        if target_type == 'gaussian':
            scale = 1./prior_data['lambda_b0'][target_index]
            shape = prior_data['lambda_a0'][target_index]
            std = np.sqrt(1./np.random.gamma(shape, scale, size=1))
        
        co = sample_cos(prior_data['w_mu0'],
                        prior_data['w_var0'])[:, target_index, 0]
    
        X_tmp = np.ones([num_dom_locs, M])
        for (inc, pp) in enumerate(preds):
            tmp_pow = pp.split('^')
            tmp_int = pp.split('*')
            
            if len(tmp_pow) > 1:
                if op.x_axis in tmp_pow:                
                    X_tmp[:, inc] = x_dom**(int(tmp_pow[-1]))
                else:                
                    X_tmp[:, inc] = np.mean(df[tmp_pow[0]].values)**\
                        (int(tmp_pow[-1]))
            elif len(tmp_int) > 1:
                if op.x_axis in tmp_int:                
                    X_tmp[:, inc] = \
                        x_dom**np.mean(df[tmp_int[np.where(np.array(tmp_int) \
                                    != op.x_axis)[0][0]]].values)
                else:
                    X_tmp[:, inc] = np.mean(df[tmp_int[0]])*\
                        np.mean(df[tmp_int[1]])
            elif pp == op.x_axis:
                X_tmp[:, inc] = x_dom
            else:
                X_tmp[:, inc] = np.mean(df[tmp_pow[0]].values)

        if target_type == 'gaussian':
            y_tmp = np.dot(co, X_tmp.T)
        elif target_type == 'binary':
            y_tmp = np.exp(np.dot(co, X_tmp.T))/\
                (1 + np.exp(np.dot(co, X_tmp.T)))
        
        ax.plot(x_dom, y_tmp)
        if target_type == 'gaussian' and not op.hide_resid:
            ax.fill_between(x_dom, y_tmp-2*std, y_tmp+2*std, alpha=0.3)

    x_label = op.x_label if op.x_label is not None else op.x_axis
    y_label = op.y_label if op.y_label is not None else op.y_axis
    ax.set_xlabel(x_label, fontsize=16)
    ax.set_ylabel(y_label, fontsize=16)
    if op.ylim is not None:
        ax.set_ylim(float(op.ylim.strip('--').split(',')[0]),
                    float(op.ylim.strip('--').split(',')[1]))
    
    if op.fig_file is not None:
        print("Saving figure...")
        plt.savefig(op.fig_file)
        print("Writing provenance info...")
        write_provenance_data(op.fig_file, generator_args=op, desc=""" """,
                              module_name='bayes_traj')
        print("DONE.")
    else:
        plt.show()

if __name__ == "__main__":
    main()    

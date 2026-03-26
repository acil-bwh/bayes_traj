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
    parser.add_argument('--set_vals', help='Comma-separated predictor=value '
        'pairs to pin predictors to fixed values when generating prior draws. '
        'Example: "cohort=1,sex=0". Predictors not specified here and not used '
        'for the x-axis default to their sample means.', type=str, default=None)
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

    set_vals = {}
    if op.set_vals is not None:
        for item in op.set_vals.split(','):
            item = item.strip()
            if item == '':
                continue
            assert '=' in item, \
                "--set_vals entries must have the form predictor=value"
            pred, val = item.split('=')
            pred = pred.strip()
            val = float(val.strip())
            set_vals[pred] = val
    
    nonnan_ids = ~np.isnan(df[op.y_axis].values)
    target_type = 'gaussian'
    if set(df[op.y_axis].values[nonnan_ids]).issubset({1.0, 0.0}):
        target_type = 'binary'        
    
    if op.prior is not None:
        with open(op.prior, 'rb') as f:
            prior_file_info = pickle.load(f)

            targets = get_target_names_from_prior_info(prior_file_info)
            preds_traj = get_pred_names_from_prior_info(prior_file_info)
            shared_predictors = prior_file_info.get('shared_predictors', [])
            if shared_predictors is None:
                shared_predictors = []

            preds = preds_traj + shared_predictors

            D = len(targets)
            M = len(preds)

            prior_data = {}
            prior_data['w_mu0'] = np.zeros([M, D])
            prior_data['w_var0'] = np.ones([M, D])
            prior_data['w_mu0_shared'] = np.zeros([len(shared_predictors), D])
            prior_data['w_var0_shared'] = np.ones([len(shared_predictors), D])
            prior_data['lambda_a0'] = np.ones([D])
            prior_data['lambda_b0'] = np.ones([D])

            for (d, target) in enumerate(targets):
                prior_data['lambda_a0'][d] = \
                    prior_file_info['lambda_a0'][target]
                prior_data['lambda_b0'][d] = \
                    prior_file_info['lambda_b0'][target]

                for (m, pred) in enumerate(preds_traj):
                    prior_data['w_mu0'][m, d] = \
                        prior_file_info['w_mu0'][target][pred]
                    prior_data['w_var0'][m, d] = \
                        prior_file_info['w_var0'][target][pred]

                for (j, pred) in enumerate(shared_predictors):
                    m = len(preds_traj) + j

                    # benign defaults in the full coefficient block
                    prior_data['w_mu0'][m, d] = 0.0
                    prior_data['w_var0'][m, d] = 1.0

                    if 'w_mu0_shared' in prior_file_info and \
                       target in prior_file_info['w_mu0_shared'] and \
                       pred in prior_file_info['w_mu0_shared'][target]:
                        prior_data['w_mu0_shared'][j, d] = \
                            prior_file_info['w_mu0_shared'][target][pred]
                    else:
                        prior_data['w_mu0_shared'][j, d] = 0.0

                    if 'w_var0_shared' in prior_file_info and \
                       target in prior_file_info['w_var0_shared'] and \
                       pred in prior_file_info['w_var0_shared'][target]:
                        prior_data['w_var0_shared'][j, d] = \
                            prior_file_info['w_var0_shared'][target][pred]
                    else:
                        prior_data['w_var0_shared'][j, d] = 1.0        
    
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

        co_shared = None
        if len(shared_predictors) > 0:
            co_shared = np.random.normal(
                loc=prior_data['w_mu0_shared'][:, target_index],
                scale=np.sqrt(prior_data['w_var0_shared'][:, target_index])
            )
        
        X_tmp = np.ones([num_dom_locs, M])
        for (inc, pp) in enumerate(preds):
            tmp_pow = pp.split('^')
            tmp_int = pp.split('*')

            # direct override via --set_vals
            if pp in set_vals:
                X_tmp[:, inc] = set_vals[pp]
                continue

            # intercept handling
            if pp.lower() in ['intercept', 'const']:
                X_tmp[:, inc] = 1.0
                continue

            if len(tmp_pow) > 1:
                base_pred = tmp_pow[0]
                power = int(tmp_pow[-1])

                if op.x_axis == base_pred:
                    X_tmp[:, inc] = x_dom ** power
                elif base_pred in set_vals:
                    X_tmp[:, inc] = set_vals[base_pred] ** power
                else:
                    X_tmp[:, inc] = np.mean(df[base_pred].values) ** power

            elif len(tmp_int) > 1:
                vals = []
                for pred_part in tmp_int:
                    if pred_part == op.x_axis:
                        vals.append(x_dom)
                    elif pred_part in set_vals:
                        vals.append(np.ones(num_dom_locs) * set_vals[pred_part])
                    elif pred_part.lower() in ['intercept', 'const']:
                        vals.append(np.ones(num_dom_locs))
                    else:
                        vals.append(np.ones(num_dom_locs) *
                                    np.mean(df[pred_part].values))

                prod = np.ones(num_dom_locs)
                for vv in vals:
                    prod = prod * vv
                X_tmp[:, inc] = prod

            elif pp == op.x_axis:
                X_tmp[:, inc] = x_dom

            elif pp in set_vals:
                X_tmp[:, inc] = set_vals[pp]

            else:
                X_tmp[:, inc] = np.mean(df[pp].values)

        linpred = np.dot(co, X_tmp.T)

        if len(shared_predictors) > 0:
            X_tmp_shared = X_tmp[:, len(preds_traj):]
            linpred = linpred + np.dot(co_shared, X_tmp_shared.T)

        if target_type == 'gaussian':
            y_tmp = linpred
        elif target_type == 'binary':
            y_tmp = np.exp(linpred) / (1 + np.exp(linpred))

            
        ax.plot(x_dom, y_tmp)
        if target_type == 'gaussian' and not op.hide_resid:
            ax.fill_between(x_dom, y_tmp-2*std, y_tmp+2*std, alpha=0.3)

    x_label = op.x_label if op.x_label is not None else op.x_axis
    y_label = op.y_label if op.y_label is not None else op.y_axis
    ax.set_xlabel(x_label, fontsize=16)
    ax.set_ylabel(y_label, fontsize=16)
    if op.ylim is not None:
        ax.set_ylim(np.min(np.array(op.ylim.split(','), dtype=float)),
                    np.max(np.array(op.ylim.split(','), dtype=float)))
    
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

#!/usr/bin/env python

import numpy as np
import pdb
import pickle
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from provenance_tools.write_provenance_data import write_provenance_data

def main():
    desc = """"""
    
    parser = ArgumentParser(description=desc)
    parser.add_argument('--model', help='Model containing trajectories to visualize',
        type=str, required=True)
    parser.add_argument('--y_axis', help='Name of the target variable that will \
        be plotted on the y-axis', type=str, required=True)
    parser.add_argument('--y_label', help='Label to display on y-axis. If none \
        given, the variable name specified with the y_axis flag will be used.',
        type=str, default=None)
    parser.add_argument('--x_axis', help='Name of the predictor variable that will \
        be plotted on the x-axis', type=str, required=True)
    parser.add_argument('--x_label', help='Label to display on x-axis. If none \
        given, the variable name specified with the x_axis flag will be used.',
        type=str, default=None)
    parser.add_argument('--trajs', help='Comma-separated list of trajectories to \
        plot. If none specified, all trajectories will be plotted.', type=str,
        default=None)
    parser.add_argument('--min_traj_prob', help='The probability of a given \
        trajectory must be at least this value in order to be rendered. Value \
        should be between 0 and 1 inclusive.', type=float, default=0)
    parser.add_argument('--max_traj_prob', help='The probability of a given \
        trajectory can not be larger than this value in order to be rendered. \
        Value should be between 0 and 1 inclusive.', type=float, default=1.01)
    parser.add_argument('--fig_file', help='If specified, will save the figure to \
        file.', type=str, default=None)
    parser.add_argument('--traj_map', help='The default trajectory numbering \
        scheme is somewhat arbitrary. Use this flag to provide a mapping \
        between the defualt trajectory numbers and a desired numbering scheme. \
        Provide as a comma-separated list of hyphenated mappings. \
        E.g.: 3-1,18-2,7-3 would indicate a mapping from 3 to 1, from 18 to 2, \
        and from 7 to 3. Only the default trajectories in the mapping will be \
        plotted. If this flag is specified, it will override --trajs', type=str,
        default=None)
    parser.add_argument('--xlim', help='Comma-separated tuple to set the \
        limits of display for the x-axis', type=str, default=None)
    parser.add_argument('--ylim', help='Comma-separated tuple to set the \
        limits of display for the y-axis', type=str, default=None)    
    parser.add_argument('--hs', help='This flag will hide the data scatter \
        plot', action="store_true")
    parser.add_argument('--htd', help='This flag will hide trajectory legend \
        details (can reduce clutter)', action="store_true")
    parser.add_argument('--traj_markers', help='Comma-separated list of \
        markers to use for each trajectory. The number of markers should match \
        the number of trajectories to renders. See matplotlib documentation \
        for marker options', default=None)
    parser.add_argument('--traj_colors', help='Comma-separated list of \
        colors to use for each trajectory. The number of colors should match \
        the number of trajectories to renders. See matplotlib documentation \
        for color options', default=None)
    parser.add_argument('--fill_alpha', help='Value between 0 and 1 that \
        controls opacity of each trajectorys fill region (which indicates \
        +\- 2 residual standard deviations about the mean)', default=0.3,
        type=float)        
    
    op = parser.parse_args()

    traj_map = None
    if op.traj_map is not None:
        traj_map = {}
        for ii in op.traj_map.split(','):
            traj_map[int(ii.split('-')[0])] = int(ii.split('-')[1])
    
    with open(op.model, 'rb') as f:
        mm = pickle.load(f)['MultDPRegression']
        assert op.x_axis in mm.predictor_names_, \
            'x-axis variable not among model predictor variables'
        assert op.y_axis in mm.target_names_, \
            'y-axis variable not among model target variables'
        
        show = op.fig_file is None

        traj_markers = None
        if op.traj_markers is not None:            
            traj_markers = op.traj_markers.split(',')

        traj_colors = None
        if op.traj_colors is not None:            
            traj_colors = op.traj_colors.split(',')            
        
        if op.trajs is not None:
            ax = mm.plot(op.x_axis, op.y_axis, op.x_label, op.y_label,
                         np.array(op.trajs.split(','), dtype=int),
                         show=show, min_traj_prob=op.min_traj_prob,
                         max_traj_prob=op.max_traj_prob, traj_map=traj_map,
                         hide_scatter=op.hs, hide_traj_details=op.htd,
                         traj_markers=traj_markers, traj_colors=traj_colors,
                         fill_alpha=op.fill_alpha)
        else:            
            ax = mm.plot(op.x_axis, op.y_axis, op.x_label, op.y_label,
                         show=show, min_traj_prob=op.min_traj_prob,
                         max_traj_prob=op.max_traj_prob, traj_map=traj_map,
                         hide_scatter=op.hs, hide_traj_details=op.htd,
                         traj_markers=traj_markers, traj_colors=traj_colors,
                         fill_alpha=op.fill_alpha)
            
        if op.ylim is not None:
            plt.ylim(float(op.ylim.strip('--').split(',')[0]),
                     float(op.ylim.strip('--').split(',')[1]))
        if op.xlim is not None:
            plt.xlim(float(op.xlim.strip('--').split(',')[0]),
                     float(op.xlim.strip('--').split(',')[1]))            
            
        if op.fig_file is not None:
            print("Saving figure...")
            plt.savefig(op.fig_file)
            print("Writing provenance info...")
            write_provenance_data(op.fig_file, generator_args=op, desc=""" """,
                                  module_name='bayes_traj')
            print("DONE.")

if __name__ == "__main__":
    main()                  

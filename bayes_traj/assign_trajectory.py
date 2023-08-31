#!/usr/bin/env python

from argparse import ArgumentParser
import pandas as pd
import numpy as np
from bayes_traj.mult_dp_regression import MultDPRegression
from provenance_tools.write_provenance_data import write_provenance_data
import pdb, pickle

def main():
    """
    """
    desc = """Assigns individuals to trajectory subgroups using their data 
    contained the input csv file and a trajectory model. The individuals can be 
    different from those used to train the model. However, it is assumed that 
    the predictor names and target names  match."""

    args = ArgumentParser(desc)
    args.add_argument('--in_csv', help='Input csv data file. Individuals in \
        this file will be assigned to the best trajectory', required=True,
        type=str)
    args.add_argument('--groupby', help='Subject identifier column name in the \
        input data file to use for grouping. ', required=False, type=str,
        default=None)    
    args.add_argument('--model', help='Pickled trajectory model to use for \
        assigning data instances to trajectories', type=str, required=True)
    args.add_argument('--out_csv', help='Output csv file with data instances \
        assigned to trajectories. The output csv file will be identical to the \
        input csv file, but it will additionally have a traj column indicating \
        the trajectory number with the highest assigmnet probability. It will \
        also contain columns with the traj_ prefix, followed by a numer. These \
        columns contain the probability of assignment to the corresponding \
        trajectory.', type=str, default=None)
    args.add_argument('--traj_map', help='The default trajectory numbering \
        scheme is somewhat arbitrary. Use this flag to provide a mapping \
        between the defualt trajectory numbers and a desired numbering scheme. \
        Provide as a comma-separated list of hyphenated mappings. \
        E.g.: 3-1,18-2,7-3 would indicate a mapping from 3 to 1, from 18 to 2, \
        and from 7 to 3. Original trajectory values not used in the mapping \
        will be reassigned to NaNs ', type=str, default=None)        

    op = args.parse_args()
    
    print("Reading data...")
    df = pd.read_csv(op.in_csv)

    print("Reading model...")
    mm = pickle.load(open(op.model, 'rb'))['MultDPRegression']

    traj_map = {}
    if op.traj_map is not None:
        for ii in np.where(mm.sig_trajs_)[0]:
            traj_map[ii] = np.nan
        for ii in op.traj_map.split(','):
            traj_map[int(ii.split('-')[0])] = int(ii.split('-')[1])
    else:
        for ii in np.where(mm.sig_trajs_)[0]:
            traj_map[ii] = ii
    
    print("Assigning...")    
    df_out = mm.augment_df_with_traj_info(df, op.groupby)
    df_out.replace({'traj': traj_map}, inplace=True)
    
    if op.out_csv is not None:
        print("Saving data with trajectory info...")
        df_out.to_csv(op.out_csv, index=False)
        
        print("Saving data file provenance info...")
        provenance_desc = """ """
        write_provenance_data(op.out_csv, generator_args=op,
                              desc=provenance_desc,
                              module_name='bayes_traj')  

    print("DONE.")
        
if __name__ == "__main__":
    main()
    

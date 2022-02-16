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
    the predictor names and target names, as well as the groupby name, match."""

    args = ArgumentParser(desc)
    args.add_argument('--in_csv', help='Input csv data file. Individuals in \
        this file will be assigned to the best trajectory', required=True,
        type=str)
    args.add_argument('--gb', help='Subject identifier column name in the \
        input data file to use for grouping. If none specified, an attempt \
        will be made to get this from the input model. However, there may be a \
        mismatch between the subject identifier stored in the model and the \
        appropriate column in the input data. If this is the case, this flag \
        should be used.', required=False, type=str)    
    args.add_argument('--model', help='Pickled trajectory model to use for \
        assigning data instances to trajectories', type=str, required=True)
    args.add_argument('--out_csv', help='Output csv file with data instances \
        assigned to trajectories. The output csv file will be identical to the \
        input csv file, but it will additionally have a traj column indicating \
        the trajectory number with the highest assigmnet probability. It will \
        also contain columns with the traj_ prefix, followed by a numer. These \
        columns contain the probability of assignment to the corresponding \
        trajectory.', type=str, default=None)

    op = args.parse_args()

    print("Reading data...")
    df = pd.read_csv(op.in_csv)

    print("Reading model...")
    mm = pickle.load(open(op.model, 'rb'))['MultDPRegression']

    print("Assigning...")
    groupby_col = None
    if op.gb is not None:
        groupby_col = op.gb
    elif mm.gb_ is not None:
        groupby_col = mm.gb_.count().index.name
        
    df_out = mm.augment_df_with_traj_info(mm.target_names_,
        mm.predictor_names_, df, groupby_col)

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
    

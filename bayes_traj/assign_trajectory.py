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
    probs = mm.predict_proba_(df[mm.predictor_names_].values,
                              df[mm.target_names_].values)

    df['traj'] = [np.where(probs[ii,:] == np.max(probs[ii, :]))[0][0] \
                  for ii in range(df.shape[0])]

    for tt in np.where(mm.sig_trajs_)[0]:
        df['traj_{}'.format(tt)] = probs[:, tt]
    
    if op.out_csv is not None:
        print("Saving data with trajectory info...")
        df.to_csv(op.out_csv, index=False)
        
        print("Saving data file provenance info...")
        provenance_desc = """ """
        write_provenance_data(op.out_csv, generator_args=op,
                              desc=provenance_desc,
                              module_name='bayes_traj')  

    print("DONE.")
        
if __name__ == "__main__":
    main()
    

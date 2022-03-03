#!/usr/bin/env python

from argparse import ArgumentParser
from bayes_traj.mult_dp_regression import MultDPRegression
from provenance_tools.write_provenance_data import write_provenance_data
import pdb, pickle, sys, warnings

def main():
    """
    """

    desc = """Updates older models to ensure compatibility with latest 
    implementation"""
    
    parser = ArgumentParser(description=desc)
    parser.add_argument('--in_model', help='Filename of input model to update',
        metavar='<string>', required=True)
    parser.add_argument('--out_model', help='Filename of updated output model',
        metavar='<string>')    
    
    op = parser.parse_args()
    
    print("Reading model...")
    with open(op.in_model, 'rb') as f:
        mm_in = pickle.load(f)['MultDPRegression']

        print("Updating...")
        mm_out = MultDPRegression(mm_in)
        
        if op.out_model is not None:
            print("Saving updated model...")
            pickle.dump({'MultDPRegression': mm_out}, open(op.out_model, 'wb'))
    
            print("Saving model provenance info...")
            provenance_desc = """ """
            write_provenance_data(op.out_model, generator_args=op,
                                  desc=provenance_desc,
                                  module_name='bayes_traj')

    print("DONE.")
    
if __name__ == "__main__":
    main()
        

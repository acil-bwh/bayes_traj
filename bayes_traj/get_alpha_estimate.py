#!/usr/bin/env python

from argparse import ArgumentParser
import numpy as np

def get_alpha_estimate(n, k):
    """Provides an estimate of the alpha parameter used in the Dirichlet 
    Process stick-breaking procedure given the number of individuals in the data
    sample and the expected number of trajectories

    Parameters
    ----------
    n : int
        Number of individuals in the data sample

    k : int
        Number of expected trajectories

    Returns
    -------
    alpha : float
        Estimate of alpha
    """

    alpha = None
    for aa in np.linspace(0.001, 10, 1000):
        if aa*np.log(1 + n/aa) > k:
            alpha = aa
            break
        
    return alpha
            
def main():        
    desc = """Provides an estimate of the alpha parameter used in the Dirichlet 
    Process stick-breaking procedure given the number of individuals in the data 
    sample and the expected number of trajectories."""
    
    parser = ArgumentParser(description=desc)
    parser.add_argument('-n', help='Number of subjects in data sample',
        type=int, required=True, default=None)
    parser.add_argument('-k', help='Number of expected trajectories',
        type=int, required=True, default=None)
    
    op = parser.parse_args()

    alpha = get_alpha_estimate(op.n, op.k)
    output = f'Alpha estimate: {alpha:.3f} '
    output += f'(for {op.n} individuals '
    output += f'and {op.k} exptected trajectories)'
    print(output)
    
if __name__ == "__main__":
    main()
    

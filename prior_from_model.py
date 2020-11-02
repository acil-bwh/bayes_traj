import pickle
import numpy as np
from bayes_traj.mult_dp_regression import MultDPRegression
from argparse import ArgumentParser

def prior_from_model(mm):
    """Computes and returns a MultDPRegression prior given an input model by 
    considering samples from non-zero trajectory posteriors.

    Parameters
    ----------
    mm : MultDPRegression instance
        Model from which the prior will be estimated.

    Returns
    -------
    prior : dict
        Prior with keys w_mu0, w_var0, lambda_a0, lambda_b0, traj_probs, alpha.
    """
    traj_ids = np.where(mm.sig_trajs_)[0]

    M = mm.M_
    D = mm.D_

    w_mu0_post = np.zeros([M, D])
    w_var0_post = np.ones([M, D])

    # Compute the weights of each trajectory by marginalizing over individuals
    traj_probs = np.sum(mm.R_, 0)/np.sum(mm.R_)        
    
    # Each trajectory regression coefficient is a draw from the corresponding
    # prior. That prior is characterized by a mean (held in mm.w_mu0_) and
    # a variance (held in mm.w_var0_). We can examine the actual regression
    # coefficients found after the fitting routine to update our belief about
    # what the prior should be. In order to do this, we'll draw samples from
    # w_mu_ (the per-trajectory posterior mean) scaled by the variance in
    # the posterior estimate w_var_ and marginalized over the probability
    # of each trajectory. The mean and variance of the resulting sample
    # provides an update for prior over coefficients.
    num_traj_samples = np.random.multinomial(10000, traj_probs)

    for m in range(M):
        for d in range(D):
            samples = []
            for t in traj_ids:
                samples.append(mm.w_mu_[m, d, t] + \
                               np.sqrt(mm.w_var_[m, d, t])*\
                               np.random.randn(num_traj_samples[t]))

            w_mu0_post[m, d] = np.mean(np.hstack(samples))
            w_var0_post[m, d] = np.var(np.hstack(samples))

    # For precision parameters, we'll use a similar sample-based procedure as
    # was done for the coefficients
    lambda_a0_post = np.ones(D)
    lambda_b0_post = np.ones(D)
    for d in range(D):
        samples = []
        for t in traj_ids:
            scale_tmp = 1./mm.lambda_b_[d, t]
            shape_tmp = mm.lambda_a_[d, t]
            samples.append(np.random.gamma(shape_tmp, scale_tmp,
                                           num_traj_samples[t]))
    
        lambda_a0_post[d] = np.mean(np.hstack(samples))**2/\
            np.var(np.hstack(samples))
        lambda_b0_post[d] = np.mean(np.hstack(samples))/\
            np.var(np.hstack(samples))
        
    prior = {'w_mu0': w_mu0_post, 'w_var0': w_var0_post,
             'lambda_a0': lambda_a0_post, 'lambda_b0': lambda_b0_post,
             'traj_probs': traj_probs, 'alpha': mm.alpha_}

    return prior

if __name__ == "__main__":
    desc = """This script generates a prior based on a fit input model"""
    args = ArgumentParser()
    args.add_argument('--model', help='Pickled MultDPRegression object that \
      has been fit to data and from which we will extract information to \
      produce an updated prior file', dest='model', default=None)
    args.add_argument('--prior', help='Output pickle file containing updated \
      prior settings', dest='prior', default=None)

    op.parse_args()
    
    mm = pickle.load(open(op.model, 'rb'))['MultDPRegression']
    prior = prior_from_model(mm)
    
    pickle.dump(prior, open(op.prior, 'wb'))
    
    desc = """  """
    write_provenance_data(op.prior, generator_args=op, desc=desc)    

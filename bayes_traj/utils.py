import pickle
import numpy as np
from argparse import ArgumentParser

def get_pred_names_from_prior_info(prior_info):
    """Gets the list of predictor names used to construct a prior info 
    dictionary.

    Parameters
    ----------
    prior_info : dict
        Dictionary containing prior information. The dictionary structure is 
        equivalent to that produced by the generate_prior.py utility.

    Returns
    -------
    preds : list of strings
        List of predictor names
    
    """
    return list(list(prior_info['w_mu0'].values())[0].keys())


def get_target_names_from_prior_info(prior_info):
    """Gets the list of target names used to construct a prior info dictionary.

    Parameters
    ----------
    prior_info : dict
        Dictionary containing prior information. The dictionary structure is 
        equivalent to that produced by the generate_prior.py utility.

    Returns
    -------
    target : list of strings
        List of target names
    
    """
    return list(prior_info['w_mu0'].keys())


def sample_precs(lambda_a0, lambda_b0, num_samples):
    """Samples trajectory precision values given parameters describing the 
    distribution over precisions.

    Parameters
    ----------
    lambda_a0 : array, shape ( D )
        The first parameter of the gamma distribution over trajectory 
        precisions for each of the D target dimensions.

    lambda_b0 : array, shape ( D )
        The second parameter of the gamma distribution over trajectory 
        precisions for each of the D target dimensions.

    num_samples : int
        Number of samples to draw

    Returns
    -------
    prec : array, shape ( D, num_samples )
        The randomly generated trajectory precisions
    """
    assert lambda_a0.ndim == lambda_b0.ndim == 1,  \
        "Unexpected number of dimensions"
    
    D = lambda_a0.shape[0]

    prec = np.zeros([D, num_samples])
    for dd in range(D):
        scale_tmp = 1./lambda_b0[dd]
        shape_tmp = lambda_a0[dd]
        prec[dd, :] = np.random.gamma(shape_tmp, scale_tmp, num_samples)

    return prec


def sample_cos(w_mu0, w_var0, num_samples=1):
    """Samples trajectory coefficients given parameters describing the 
    distribution over coefficients

    Parameters
    ----------
    w_mu0 : array, shape ( M, D )
        The mean of the multivariate normal distribution over trajectoreis. M
        is the number of predictors, and D is the number of dimensions.

    w_var0 : array, shape ( M, D )
        The variances of the Normal distributions over the trajectory 
        coefficients.

    num_samples : int , optional
        Number of samples to draw

    Returns
    -------
    w : array, shape ( M, D, num_samples )
        The randomly generated trajectory coefficients    
    """
    assert w_mu0.ndim == w_var0.ndim == 2, \
        "Unexpected number of dimensions"
    
    M = w_mu0.shape[0]
    D = w_mu0.shape[1]

    assert w_var0.shape[0] == M and w_var0.shape[1] == D, \
        "Unexpected shape"

    w = np.zeros([M, D, num_samples])
    for mm in range(M):
        for dd in range(D):
            w[mm, dd, :] = w_mu0[mm, dd] + \
                np.sqrt(w_var0[mm, dd])*np.random.randn(num_samples)

    return w


def sample_traj(w_mu0, var_covar0, lambda_a0, lambda_b0, num_samples):
    """Samples a trajectory given an input description of the distribution over
    trajectories. A sampled trajectory is represented in terms of predictor
    coefficients (w_mu) and precision values for each dimension.

    Parameters
    ----------
    w_mu0 : array, shape ( M, D )
        The mean of the multivariate normal distribution over trajectoreis. M
        is the number of predictors, and D is the number of dimensions.

    var_covar : array, shape ( M, D ) or ( MxD, MxD )
        The variance (covariance) of the multivariate normal distribution over
        trajectories. If the shape is equivalent to that of w_mu0, the elements
        will be taken as diagonal elements of the multivariate's covariance
        matrix. Otherwise, the matrix is expected to be a full MxD by MxD 
        covariance matrix.

    lambda_a0 : array, shape ( D )
        The first parameter of the gamma distribution over trajectory 
        precisions for each of the D target dimensions.

    lambda_b0 : array, shape ( D )
        The second parameter of the gamma distribution over trajectory 
        precisions for each of the D target dimensions.

    num_samples : int
        Number of samples to draw

    Returns
    -------
    w : array, shape ( M, D, num_samples )
        The randomly generated trajectory coefficients    

    precs : array, shape ( D, num_samples )
        The randomly generated trajectory precisions
    """

    w = sample_cos(w_mu0, var_covar0, num_samples)
    precs = sample_precs(lambda_a0, lambda_b0, num_samples)

    return w, precs

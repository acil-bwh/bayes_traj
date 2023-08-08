import torch
from bayes_traj.mult_dp_regression import MultDPRegression
from bayes_traj.psis import psisloo
from numpy.random import multivariate_normal, randn, gamma
import numpy as np
import pdb

def ave_pp(mm):
    """Computes the average posterior probability of assignment. Ideally, 
    the posterior probability (pp) of assignment is 1 for each individual, so 
    the average pp should also be 1 for each trajectory. As a rule of thumb,
    the average pp should be at least .7 for all groups (see ref, section 
    5.5.1).

    Parameters
    ----------
    mm : MultDPRegression
        Post-fit trajectory model

    Returns
    -------
    ave_pps : dict
        A dictionary where the keys are integers indicating trajectories in the
        trajectory model, and the corresponding values are the average posterior
        probabilities for those trajectories.

    References
    ----------
    Nagin DS, NAGIN D. Group-based modeling of development. Harvard University 
    Press; 2005.
    """
    df_traj = mm.to_df()
    ave_pps = {}
    if torch.is_tensor(mm.sig_trajs_):
        sig_trajs = mm.sig_trajs_.numpy()
    else:
        sig_trajs = mm.sig_trajs_

    if torch.is_tensor(mm.R_):
        R = mm.R_.numpy()
    else:
        R = mm.R_
        
    for t in np.where(sig_trajs)[0]:
        ids = df_traj['traj'] == t
        ave_pps[t] = np.mean(R[ids, t])

    return ave_pps

def odds_correct_classification(mm):
    """Measures the odds of correct classification (OCC) for each trajectory. As 
    the average posterior probability of assignment approaches its ideal value 
    of 1, OCC for a given trajectory increases, indicating better assignment
    accuracy. As a rule of thumb, OCC values should be > 5 for all trajectories.
    See section 5.5.2 in the reference.
    
    Parameters
    ----------
    mm : MultDPRegression
        Post-fit trajectory model

    Returns
    -------
    occs : dict
        A dictionary where the keys are integers indicating trajectories in the 
        trajectory model, and the corresponding values are the odds of correct
        classification.

    References
    ----------
    Nagin DS, NAGIN D. Group-based modeling of development. Harvard University 
    Press; 2005.
    """
    ave_pps = ave_pp(mm)

    if torch.is_tensor(mm.R_):
        pis = np.sum(mm.R_.numpy(), 0)/np.sum(mm.R_.numpy())
    else:
        pis = np.sum(mm.R_, 0)/np.sum(mm.R_)        
    
    df_traj = mm.to_df()
    occs = {}
    for t in np.where(mm.sig_trajs_)[0]:
        ids = df_traj['traj'] == t

        # Odds of correct classification into trajectory t:
        occ_num = ave_pps[t]/(1-ave_pps[t])

        # Odds of correct classification based on random assignment:
        occ_denom = pis[t]/(1-pis[t])
        
        occs[t] = occ_num/occ_denom
        
    return occs


def prob_prop(mm):
    """Estimated group probabilities (prob) versus the proportion (prop) of the 
    sample assigned to the group (using the maximum posterior assignment rule).
    If individuals are assigned to their respective trajectories with perfect
    certainty, prob and prop would be identical. As assignment error increases,
    the correspondence may deteriorate. There is no rule for determining what 
    level of disagreement is too much, however. See 5.5.3 of reference for 
    further information.
    
    Parameters
    ----------
    mm : MultDPRegression
        Post-fit trajectory model

    Returns
    -------
    prop_probs : dict
        A dictionary where the keys are integers indicating trajectories in the 
        trajectory model, and the corresponding values are tuples, where the 
        first element is the proportion of individuals assigned to that 
        trajectory, and the second element is the probability of occurrence of
        that trajectory.        

    References
    ----------
    Nagin DS, NAGIN D. Group-based modeling of development. Harvard University 
    Press; 2005.
    """
    if torch.is_tensor(mm.R_):
        probs = np.sum(mm.R_.numpy(), 0)/np.sum(mm.R_.numpy())
    else:
        probs = np.sum(mm.R_, 0)/np.sum(mm.R_)
    
    df_traj = mm.to_df()
        
    prop_probs = {}
    for t in np.where(mm.sig_trajs_)[0]:
        ids = df_traj['traj'] == t
        prop = np.sum(ids)/float(mm.N_)

        prop_probs[t] = (prop, probs[t])

    return prop_probs

def get_group_likelihood_samples(mm, num_samples=100):
    """This is a helper function for compute_waic2. It is somewhat memory 
    intensive to gather a large number of samples from the posterior 
    likelihood, so the strategy is to do it in chunks. This function will be
    iteratively called to produce samples.

    Parameters
    ---------
    mm : MultDPRegression
        Post-fit trajectory model

    num_samples : int, optional
        The number of samples to draw

    Returns
    -------
    group_likelihood : array, shape ( Ng, S )
        The likelihood. Each row corresponds to an individual (note there may be 
        multiple longitudinal observations per individual, as well as multiple
        dimensions. The likelihood for an individual is computed as the product
        over all time points and target dimensions); each column corresponds to
        a sample. Very small values are set to a small constant (1e-300). 
    """
    # Impute any missing target values by sampling from the posterior
    # distribution
    if torch.is_tensor(mm.Y_):
        Y = mm.Y_.numpy()
    else:
        Y = mm.Y_
        
    tmp_indices = np.where(np.isnan(np.sum(Y, 1)))[0]

    for tt in tmp_indices:
        tmp_sample = mm.sample(index=tt)

        if torch.is_tensor(mm.Y_):
            Y[tt, np.isnan(mm.Y_[tt, :].numpy())] = \
                tmp_sample[0, np.isnan(mm.Y_[tt, :].numpy())]
        else:
            Y[tt, np.isnan(mm.Y_[tt, :])] = \
                tmp_sample[0, np.isnan(mm.Y_[tt, :])]        

    # Sampling from the multinomial distribution is done is such a way that the
    # last element of the probability vector accounts for the excess prob.
    # needed to bring the sum of the vec to 1. This can be a problem if the R_
    # matrix in the model doesn't *quite* sum to one. In that case, it's
    # possible to get a sample of the Kth trajectory, even if the Kth element
    # of the R_ matrix is 0. 
    if torch.is_tensor(mm.R_):
        mm.R_ = mm.R_ / torch.sum(mm.R_, dim=1, keepdim=True)
    else:
        mm.R_ = mm.R_/np.sum(mm.R_, 1)[:, np.newaxis]

    # Sample from the posterior multinomial distribution -- do it per-group
    tmp = np.array([np.random.multinomial(1, \
        mm.R_[mm.gb_.indices[ss][0], mm.sig_trajs_], num_samples) \
                    for ss in mm.gb_.indices.keys()])

    if torch.is_tensor(mm.sig_trajs_):
        num_trajs = np.sum(mm.sig_trajs_.numpy())
    else:
        num_trajs = np.sum(mm.sig_trajs_)
        
    # Now draw samples from the posterior. Also create 'traj_mat_group, which boils
    # 'tmp' down to the trajectory number itself (as opposed to a one-vector draw
    # from the multinomial distribution)
    sample_sig_mat = np.zeros([mm.D_, num_trajs, num_samples])
    sample_co_mat = np.zeros([mm.M_, mm.D_, num_trajs, num_samples])
    
    sample_mu_mat = np.zeros([mm.N_, mm.D_, num_trajs, num_samples])
    traj_mat_group = np.zeros([mm.gb_.ngroups, num_samples])

    if torch.is_tensor(mm.sig_trajs_):
        sig_trajs = mm.sig_trajs_.numpy()
    else:
        sig_trajs = mm.sig_trajs_
        
    for ii, kk in enumerate(np.where(sig_trajs)[0]):
        traj_mat_group += ii*tmp[:, :, ii]
    
        for dd in range(mm.D_):
            if torch.is_tensor(mm.lambda_b_):
                scale = 1./mm.lambda_b_[dd, kk].numpy()
                shape = mm.lambda_a_[dd, kk].numpy()
            else:
                scale = 1./mm.lambda_b_[dd, kk]
                shape = mm.lambda_a_[dd, kk]
                
            sample_sig_mat[dd, ii, :] = \
                np.sqrt(1./gamma(shape, scale, size=num_samples))

            if torch.is_tensor(mm.w_mu_):                
                sample_co_mat[:, dd, ii, :] = \
                    np.random.multivariate_normal(mm.w_mu_[:, dd, kk].numpy(),
                                        np.diag(mm.w_var_[:, dd, kk].numpy()),
                                                num_samples).T                
                sample_mu_mat[:, dd, ii, :] = \
                    np.dot(mm.X_.numpy(), sample_co_mat[:, dd, ii, :])
            else:
                sample_co_mat[:, dd, ii, :] = \
                    np.random.multivariate_normal(mm.w_mu_[:, dd, kk],
                                        np.diag(mm.w_var_[:, dd, kk]),
                                                num_samples).T                
                sample_mu_mat[:, dd, ii, :] = \
                    np.dot(mm.X_, sample_co_mat[:, dd, ii, :])

    del tmp
    del sample_co_mat
    
    # Now create 'traj_mat_n'. Each column corresponds to a sample, and each entry
    # refers to a trajectory
    traj_mat_n = np.zeros([mm.N_, num_samples])

    for ii, ss in enumerate(mm.gb_.indices.keys()):
        traj_mat_n[mm.gb_.indices[ss] , :]  = traj_mat_group[ii, :]

    del traj_mat_group
        
    # Populate the sample likelihood
    sample_likelihood = np.ones([mm.N_, num_samples])
    for ii, kk in enumerate(np.where(mm.sig_trajs_)[0]):    
        for dd in range(mm.D_):
            co_tmp = 1/(np.sqrt(2*np.pi)*sample_sig_mat[np.newaxis, dd, ii, :]*\
                        np.ones([mm.N_, num_samples]))
            exp_arg_tmp = -0.5*((Y[:, dd, np.newaxis] - \
                sample_mu_mat[:, dd, ii, :])/sample_sig_mat[np.newaxis, dd, ii, :]*\
                                np.ones([mm.N_, num_samples]))**2            
    
            likelihood_tmp = co_tmp*np.exp(exp_arg_tmp)
        
            ids = traj_mat_n == ii
            sample_likelihood[ids] *= likelihood_tmp[ids]    

    group_likelihood = np.zeros([mm.gb_.ngroups, num_samples])
    for ii, vv in enumerate(mm.gb_.indices.values()):
        group_likelihood[ii, :] = np.prod(sample_likelihood[vv, :], 0)

    # Set small values to an offset
    group_likelihood[group_likelihood < 1e-300] = 1e-300
    return group_likelihood

def compute_waic2(mm):
    """
    """
    waic2 = mm.compute_waic2()

    return waic2

def compute_psis_loo(mm):
    """
    """
    group_likelihood = np.zeros([mm.gb_.ngroups, 1000])

    for i in range(5):
        group_likelihood[:, (i*200):(i+1)*200] = \
            get_group_likelihood_samples(mm, num_samples=200)

    loo, loos, ks = psisloo(np.log(group_likelihood).T)

    return loo, loos, ks

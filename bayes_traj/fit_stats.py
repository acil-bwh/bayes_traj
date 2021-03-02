from bayes_traj.mult_dp_regression import MultDPRegression
import numpy as np

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
    for t in np.where(mm.sig_trajs_)[0]:
        ids = df_traj['traj'] == t
        ave_pps[t] = np.mean(mm.R_[ids, t])

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
    probs = np.sum(mm.R_, 0)/np.sum(mm.R_)
    
    df_traj = mm.to_df()
        
    prop_probs = {}
    for t in np.where(mm.sig_trajs_)[0]:
        ids = df_traj['traj'] == t
        prop = np.sum(ids)/float(mm.N_)

        prop_probs[t] = (prop, probs[t])

    return prop_probs

import numpy as np
import torch
import pdb

def get_restructured_data(df, predictors, targets, groupby):
    """

    Parameters
    ----------
    df : pandas data frame
        Contains the data that will be used for fitting. 

    predictors : list of strings
        Predictor names

    targets : list of targets
        Target names

    groupby : string
        Column in df corresponding to the subject identifier

    Returns
    -------
    X_re : array, shape ( T, G, M )
        Restructured predictor matrix. T corresponds to the maximum number of
        time points observed for any individual. G is the number of 
        individuals, and M is the number of predictors.

    X_mask : array, shape ( T, G, M )
        Boolean mask corresponding to X_re. True where X_re is non-nan, false 
        otherwise.

    Y_re : array, shape ( T, G, D )
        Restructured predictor matrix. T corresponds to the maximum number of
        time points observed for any individual. G is the number of 
        individuals, and D is the number of targets.

    Y_mask : array, shape ( T, G, D )
        Boolean mask corresponding to Y_re. True where Y_re is non-nan, false 
        otherwise.
    """
    num_observations_per_subject = \
        df.groupby('id').apply(lambda dd : dd.shape[0]).values
    
    M = len(predictors)
    D = len(targets)
    G = df.groupby('id').ngroups
    T = np.max(num_observations_per_subject)

    X_orig = torch.from_numpy(df[predictors].values).double()
    Y_orig = torch.from_numpy(df[targets].values).double()
    X_re = torch.full((T, G, M), float('nan')).double()
    Y_re = torch.full((T, G, D), float('nan')).double()

    start_idx = 0
    for g, num_obs in enumerate(num_observations_per_subject):
        X_re[:num_obs, g, :] = X_orig[start_idx:start_idx+num_obs, :]
        Y_re[:num_obs, g, :] = Y_orig[start_idx:start_idx+num_obs, :] 
        
        start_idx += num_obs

    X_mask = ~torch.isnan(X_re[:, :, 0])
    Y_mask = ~torch.isnan(Y_re[:, :, 0])
    
    return X_re, X_mask, Y_re, Y_mask

import numpy as np
import torch
from typing import Optional, TypedDict


class RestructuredData(TypedDict):
    """Dict containing data tensors used by `MultPyro`."""

    X: torch.Tensor
    X_mask: Optional[torch.Tensor]
    Y_real: Optional[torch.Tensor]
    Y_real_mask: Optional[torch.Tensor]
    Y_bool: Optional[torch.Tensor]
    Y_bool_mask: Optional[torch.Tensor]
    cohort: Optional[torch.Tensor]


def get_restructured_data(df, predictors, targets, groupby) -> RestructuredData:
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
    restructured_data : RestructuredData
        A dict containing the restructured data:

        X : array, shape ( T, G, M )
            Restructured predictor matrix. T corresponds to the maximum number
            of time points observed for any individual. G is the number of 
            individuals, and M is the number of predictors.

        X_mask : array, shape ( T, G, M )
            Boolean mask corresponding to X_re. True where X_re is non-nan,
            false otherwise.

        Y_real : array, shape ( T, G, D )
            Restructured predictor matrix. T corresponds to the maximum number
            of time points observed for any individual. G is the number of 
            individuals, and D is the number of real valued targets.
            
        Y_real_mask : array, shape ( T, G, D )
            Boolean mask corresponding to Y_real. True where Y_real is observed,
            false otherwise.

        Y_bool : array, shape ( T, G, B )
            Restructured predictor matrix. T corresponds to the maximum number
            of time points observed for any individual. G is the number of 
            individuals, and B is the number of boolean targets.

        Y_bool_mask : array, shape ( T, G, B )
            Boolean mask corresponding to Y_bool. True where Y_bool is observed,
            false otherwise.

        cohort : array, shape ( G, )
            Optional integer array containing the cohort of each individual.
    """
    num_observations_per_subject = \
        df.groupby(groupby).apply(lambda dd : dd.shape[0]).values

    bool_cols = []
    real_cols = []
    for tt in targets:
        if set(df[tt].values) == {0, 1}:
            bool_cols.append(tt)
        else:
            real_cols.append(tt)
    
    M = len(predictors)
    D = len(real_cols)
    B = len(bool_cols)
    G = df.groupby(groupby).ngroups
    T = np.max(num_observations_per_subject)
   
    X_orig = torch.from_numpy(df[predictors].values).double()

    Y_real_orig = None
    Y_bool_orig = None
    Y_real = None
    Y_real_mask = None
    Y_bool = None
    Y_bool_mask = None
    cohort: Optional[torch.Tensor] = None
    if len(real_cols) > 0:
        Y_real_orig = torch.from_numpy(df[real_cols].values).double()
        Y_real = torch.full((T, G, D), float('nan')).double()
    if len(bool_cols) > 0:
        Y_bool_orig = torch.from_numpy(df[bool_cols].values).double()    
        Y_bool = torch.full((T, G, B), float('nan')).double()
    if "cohort" in df.columns:
        cohort_ids = sorted(set(df["cohort"].values))
        cohort_orig = torch.tensor(
            [cohort_ids.index(cohort_id) for cohort_id in df["cohort"].values],
            dtype=torch.long,
        )
        cohort = torch.full((G,), -1, dtype=torch.long)
 
    X = torch.full((T, G, M), float('nan')).double()
            
    start_idx = 0
    for g, num_obs in enumerate(num_observations_per_subject):
        X[:num_obs, g, :] = X_orig[start_idx:start_idx+num_obs, :]
        if Y_real_orig is not None:
            assert Y_real is not None
            Y_real[:num_obs, g, :] = Y_real_orig[start_idx:start_idx+num_obs, :]
        if Y_bool_orig is not None:
            assert Y_bool is not None
            Y_bool[:num_obs, g, :] = Y_bool_orig[start_idx:start_idx+num_obs, :] 
        if cohort is not None:
            cohort[g] = cohort_orig[start_idx]
        
        start_idx += num_obs

    X_mask = ~(torch.isnan(X[:, :]).any(dim=-1))
    if Y_real is not None:
        Y_real_mask = ~torch.isnan(Y_real[:, :, 0])
    if Y_bool is not None:
        Y_bool = Y_bool.bool()
        Y_bool_mask = ~torch.isnan(Y_bool[:, :, 0])

    result = RestructuredData(
        X=X,
        X_mask=None if X_mask.all() else X_mask,
        Y_real=Y_real,
        Y_real_mask=Y_real_mask,
        Y_bool=Y_bool,
        Y_bool_mask=Y_bool_mask,
        cohort=cohort,
    )
    for k, v in result.items():
        print(f'{k: >12s}: {getattr(v, "shape", "n/a")}')
    return result

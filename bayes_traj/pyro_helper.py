import numpy as np
import torch
from typing import Optional, TypedDict
import pdb


class RestructuredData(TypedDict):
    """Dict containing data tensors used by `MultPyro`."""

    X: torch.Tensor
    X_mask: Optional[torch.Tensor]
    Y_real: Optional[torch.Tensor]
    Y_real_mask: Optional[torch.Tensor]
    Y_bool: Optional[torch.Tensor]
    Y_bool_mask: Optional[torch.Tensor]
    cohort: Optional[torch.Tensor]
    group_to_index : dict

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
    gb = df.groupby(groupby)
    num_observations_per_subject = gb.apply(lambda dd : dd.shape[0]).values

    bool_cols = []
    real_cols = []
    for tt in targets:
        if set(df[tt].values).issubset({0, 1}):
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
    X = None
    X_mask = None
    cohort: Optional[torch.Tensor] = None
    X = torch.full((T, G, M), float('nan')).double()
    if len(real_cols) > 0:
        Y_real = torch.full((T, G, D), float('nan')).double()
    if len(bool_cols) > 0:
        Y_bool = torch.full((T, G, B), float('nan')).double()
    if "cohort" in df.columns:
        cohort = torch.full((G,), torch.nan)

    for ii, gg in enumerate(gb.groups.keys()):
        df_tmp = gb.get_group(gg)
        X[:df_tmp.shape[0], ii, :] = \
            torch.from_numpy(df_tmp[predictors].values)
        if len(real_cols) > 0:        
            Y_real[:df_tmp.shape[0], ii, :] = \
                torch.from_numpy(df_tmp[real_cols].values)
        if len(bool_cols) > 0:
            Y_bool[:df_tmp.shape[0], ii, :] = \
                torch.from_numpy(df_tmp[bool_cols].values)
        if "cohort" in df_tmp.columns:
            tmp = torch.from_numpy(df_tmp['cohort'].values)
            assert torch.all(tmp==tmp[0]), \
                "Different cohorts assigned to same individual"
            cohort[ii] = tmp[0]

    X_mask = ~torch.isnan(X)
    if len(real_cols) > 0:
        Y_real_mask = ~torch.isnan(Y_real)
    if len(bool_cols) > 0:
        Y_bool_mask = ~torch.isnan(Y_bool)        

    result = RestructuredData(
        X=X,
        X_mask=X_mask,
        Y_real=Y_real,
        Y_real_mask=Y_real_mask,
        Y_bool=Y_bool,
        Y_bool_mask=Y_bool_mask,
        cohort=cohort,
        group_to_index=df.groupby(groupby).groups
    )
    for k, v in result.items():
        print(f'{k: >12s}: {getattr(v, "shape", "n/a")}')
    return result



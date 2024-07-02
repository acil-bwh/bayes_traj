from bayes_traj.pyro_helper import *
import pickle, torch
import numpy as np
import os
import pdb
import bayes_traj
import pandas as pd
import warnings

torch.set_default_dtype(torch.double)

def tensors_equal_with_nan(t1, t2):
    # Check if NaN positions are the same
    nan_mask_t1 = torch.isnan(t1)
    nan_mask_t2 = torch.isnan(t2)
    
    if not torch.equal(nan_mask_t1, nan_mask_t2):
        return False
    
    # Replace NaNs with zero
    t1_nonan = torch.where(nan_mask_t1, torch.zeros_like(t1), t1)
    t2_nonan = torch.where(nan_mask_t2, torch.zeros_like(t2), t2)
    
    return torch.allclose(t1_nonan, t2_nonan)

def test_get_restructured_data():
    data = {
        'id': ['a', 'a', 'b', 'a', 'b', 'b', 'b'],
        'x1': [10, 11, 12, 13, 14, 15, 16],
        'x2': [20, 21, 22, 23, 24, 25, 26],        
        'y1': [70, 71, 72, 73, 74, 75, 76],
        'y2': [80, 81, 82, 83, 84, 85, 86],
        'y3': [90, 91, 92, 93, 94, 95, 96],
        'y4': [0, 1, 1, 0, 0, 0, 0],        
        'cohort': [1, 1, 2, 1, 2, 2, 2]
    }
    df = pd.DataFrame(data)

    predictors = ['x1', 'x2']
    targets = ['y1', 'y2', 'y3', 'y4']
    re_data = get_restructured_data(df, predictors, targets, 'id')

    assert set(re_data['Y_bool_names']) == set(['y4']), \
        "Y_bool_names incorrect"
    
    assert set(re_data['Y_real_names']) == set(['y1', 'y2', 'y3']), \
        "Y_real_names incorrect"
    
    assert tensors_equal_with_nan(re_data['Y_real'][:, 0, :], \
        torch.tensor([[70., 80., 90.],
                      [71., 81., 91.],
                      [73., 83., 93.],
                      [torch.nan, torch.nan, torch.nan]],
                     dtype=torch.float64)), "Y_real error"

    assert tensors_equal_with_nan(re_data['Y_real'][:, 1, :], \
        torch.tensor([[72., 82., 92.],
                      [74., 84., 94.],
                      [75., 85., 95.],
                      [76., 86., 96.]],
                     dtype=torch.float64)), "Y_real error"

    assert tensors_equal_with_nan(re_data['Y_bool'][:, 0, :], \
        torch.tensor([[0.],
                      [1.],
                      [0.],
                      [torch.nan]],
                     dtype=torch.float64)), "Y_bool error"

    assert tensors_equal_with_nan(re_data['Y_bool'][:, 1, :], \
        torch.tensor([[1.],
                      [0.],
                      [0.],
                      [0.]],
                     dtype=torch.float64)), "Y_bool error"    

    assert tensors_equal_with_nan(re_data['Y_bool_mask'][:, 0, :], \
        torch.tensor([[ True],
                      [ True],
                      [ True],
                      [False]],
                     dtype=torch.bool)), "Y_bool_mask error"    

    assert tensors_equal_with_nan(re_data['Y_bool_mask'][:, 1, :], \
        torch.tensor([[ True],
                      [ True],
                      [ True],
                      [ True]],
                     dtype=torch.bool)), "Y_bool_mask error"    

    assert tensors_equal_with_nan(re_data['X'][:, 0, :], \
        torch.tensor([[10., 20.],
                      [11., 21.],
                      [13., 23.],
                      [torch.nan, torch.nan]],
                     dtype=torch.float64)), "X error"

    assert tensors_equal_with_nan(re_data['X'][:, 1, :], \
        torch.tensor([[12., 22.],
                      [14., 24.],
                      [15., 25.],
                      [16., 26.]],
                     dtype=torch.float64)), "X error"

    assert tensors_equal_with_nan(re_data['cohort'], \
        torch.tensor([1., 2.], dtype=torch.int32)), "cohort error"

    assert np.sum(re_data['group_to_index']['a'] == \
        np.array([0, 1, 3], dtype='int64')) == 3, \
        "group_to_index error"

    assert np.sum(re_data['group_to_index']['b'] == \
        np.array([2, 4, 5, 6], dtype='int64')) == 4, \
        "group_to_index error"

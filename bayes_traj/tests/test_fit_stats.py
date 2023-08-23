import torch
import numpy as np
import pandas as pd
import pdb
from bayes_traj.mult_dp_regression import MultDPRegression
from bayes_traj.fit_stats import *

def get_gt_model():
    """
    """
    df = pd.DataFrame(\
        {'sid': ['a', 'a', 'a', 'a', 'a', 'b', 'b', 'b', 'b'],
         'intercept': np.ones(9),
         'x': np.array([0, 2, 4, 6, 10, 0, 3, 7, 10])})
    
    K = 5
    M = 2
    D = 2
    N = df.shape[0]
    
    w_mu_gt = torch.zeros([M, D, K]).double()
    # Trajectory 0
    w_mu_gt[:, 0, 0] = torch.tensor([0, 1]).double()
    w_mu_gt[:, 1, 0] = torch.tensor([0, -1]).double()
    
    # Trajectory 1
    w_mu_gt[:, 1, 1] = torch.tensor([0, 1]).double()
    w_mu_gt[:, 0, 1] = torch.tensor([10, -1]).double()
        
    sig = 0.05
    # 'a' is from trajectory 0. 'b' is from trajectory 1. The following
    # target values were generating assuming a residual standard deviation
    # of 0.05.
    y1 = np.array([0.06280633,  2.06280633,  4.06280633,  6.06280633,
                   10.06280633, 9.94877976,  6.94877976,  2.94877976,
                   -0.05122024])    
    y2 = np.array([0.06889208, -1.93110792, -3.93110792, -5.93110792,
                   -9.93110792, 0.10712325,  3.10712325,  7.10712325,
                   10.10712325])
    df['y1'] = y1
    df['y2'] = y2
        
    w_var0 = np.ones([M, D])
    w_mu0 = np.zeros([M, D])
    lambda_a0 = np.ones(D)
    lambda_b0 = np.ones(D)
    alpha = 1.
    
    prec_mu = 1./sig**2
    prec_var = 1e-10
    prec_prior_weight = 1
    
    #-----------------------------------------------------------------------
    # Set up model 1
    #-----------------------------------------------------------------------
    mm = MultDPRegression(w_mu0, w_var0, lambda_a0, lambda_b0,
                          prec_prior_weight, alpha, K=K)
    mm.R_ = torch.zeros([N, K]).double()
    mm.R_[0:5, 0] = 1.
    mm.R_[5::, 1] = 1.
        
    mm.target_type_ = {}
    mm.target_type_[0] = 'gaussian'
    mm.target_type_[1] = 'gaussian'
        
    mm.w_mu_ = torch.zeros([M, D, K]).double()
    mm.w_var_ = 1e-10*torch.ones([M, D, K]).double()
    mm.w_mu_[:, 0, 0] = w_mu_gt[:, 0, 0]
    mm.w_mu_[:, 1, 0] = w_mu_gt[:, 1, 0]
    mm.w_mu_[:, 0, 1] = w_mu_gt[:, 0, 1]
    mm.w_mu_[:, 1, 1] = w_mu_gt[:, 1, 1]
        
    mm.lambda_a_ = torch.ones([D, K]).double()
    mm.lambda_b_ = torch.ones([D, K]).double()
    mm.lambda_a_[0, 0] = (prec_mu**2)/prec_var
    mm.lambda_b_[0, 0] = prec_mu/prec_var
    mm.lambda_a_[1, 0] = (prec_mu**2)/prec_var
    mm.lambda_b_[1, 0] = prec_mu/prec_var
    mm.lambda_a_[0, 1] = (prec_mu**2)/prec_var
    mm.lambda_b_[0, 1] = prec_mu/prec_var
    mm.lambda_a_[1, 1] = (prec_mu**2)/prec_var
    mm.lambda_b_[1, 1] = prec_mu/prec_var            

    mm.v_a_ = torch.ones(K)
    mm.v_b_ = alpha*torch.ones(K)
    
    mm.gb_ = df.groupby('sid')
    mm.X_ = torch.from_numpy(df[['intercept', 'x']].values).double()
    mm.Y_ = torch.from_numpy(df[['y1', 'y2']].values).double()
    mm.N_ = N 

    mm._set_group_first_index(df, mm.gb_)
    
    return mm


def test_get_group_likelihood_samples_1():
    # Create a model
    df = pd.DataFrame({'sid': ['a'],
                       'intercept': np.array([1]),
                       'x': np.array([0]),
                       'y': np.array([10])})
    M = 2
    D = 1
    N = df.shape[0]
    
    w_var0 = np.zeros([M, D])
    w_mu0 = np.zeros([M, D])
    lambda_a0 = np.ones(D)
    lambda_b0 = np.ones(D)
    alpha = 1
    K = 20

    prec_mu = 1
    prec_var = 1e-10
    prec_prior_weight = 1
    
    mm = MultDPRegression(w_mu0, w_var0, lambda_a0, lambda_b0,
                          prec_prior_weight, alpha, K=K)
    mm.R_ = torch.zeros([N, K]).double()
    mm.R_[0, 0] = 1
    
    mm.w_mu_ = torch.zeros([M, D, K]).double()
    mm.w_var_ = torch.ones([M, D, K]).double()
    mm.w_mu_[:, 0, 0] = torch.tensor([10, -1], dtype=torch.float64)
    mm.w_var_[:, 0, 0] = 1e-10*torch.tensor([1, 1], dtype=torch.float64)

    mm.w_mu_[:, 0, 1] = torch.tensor([11, -1], dtype=torch.float64)
    mm.w_var_[:, 0, 1] = 1e-10*torch.tensor([1, 1], dtype=torch.float64)
    
    mm.lambda_a_ = torch.ones([D, K]).double()
    mm.lambda_b_ = torch.ones([D, K]).double()
    mm.lambda_a_[0, 0] = (prec_mu**2)/prec_var
    mm.lambda_b_[0, 0] = prec_mu/prec_var    
    
    mm.gb_ = df.groupby('sid')
    mm.X_ = torch.from_numpy(df[['intercept', 'x']].values).double()
    mm.Y_ = torch.from_numpy(np.atleast_2d(df.y.values)).double()
    mm.N_ = N 
    
    tmp = get_group_likelihood_samples(mm, num_samples=1000)
    assert np.isclose(np.mean(tmp), 1/np.sqrt(2*np.pi)), \
        "Likelihood not as expected"

    mm.R_[0, [0, 1]] = torch.tensor([0, 1]).double()
    tmp_2 = get_group_likelihood_samples(mm, num_samples=1000)

    assert np.mean(tmp) > np.mean(tmp_2), "Unexpected likelihood comparison"

def test_get_group_likelihood_samples_2():
    # Create a model
    df = pd.DataFrame({'sid': ['a', 'a', 'a'],
                       'intercept': np.array([1., 1., 1.]),
                       'x': np.array([0., 5., 10.]),
                       'y1': np.array([10., 5., 0.]),
                       'y2': np.array([0., 5., 10.])})

    M = 2
    D = 2
    N = df.shape[0]
    
    w_var0 = np.zeros([M, D])
    w_mu0 = np.zeros([M, D])
    lambda_a0 = np.ones(D)
    lambda_b0 = np.ones(D)
    alpha = 1.
    K = 20

    prec_mu = 1.
    prec_var = 1e-10
    prec_prior_weight = 1
    
    mm = MultDPRegression(w_mu0, w_var0, lambda_a0, lambda_b0,
                          prec_prior_weight, alpha, K=K)
    mm.R_ = torch.zeros([N, K]).double()
    mm.R_[:, 0] = 1.

    mm.target_type_ = {}
    mm.target_type_[0] = 'gaussian'
    mm.target_type_[1] = 'gaussian'
    
    mm.w_mu_ = torch.zeros([M, D, K]).double()
    mm.w_var_ = 1e-10*torch.ones([M, D, K]).double()
    mm.w_mu_[:, 0, 0] = torch.tensor([10, -1]).double()
    mm.w_mu_[:, 1, 0] = torch.tensor([0, 1]).double()
    
    mm.lambda_a_ = torch.ones([D, K]).double()
    mm.lambda_b_ = torch.ones([D, K]).double()
    mm.lambda_a_[0, 0] = (prec_mu**2)/prec_var
    mm.lambda_b_[0, 0] = prec_mu/prec_var
    mm.lambda_a_[1, 0] = (prec_mu**2)/prec_var
    mm.lambda_b_[1, 0] = prec_mu/prec_var        
    
    mm.gb_ = df.groupby('sid')
    mm.X_ = torch.from_numpy(df[['intercept', 'x']].values).double()
    mm.Y_ = torch.from_numpy(df[['y1', 'y2']].values).double()
    mm.N_ = N 
    
    tmp_1 = get_group_likelihood_samples(mm, num_samples=1000)
    assert np.isclose(np.mean(tmp_1), (1/np.sqrt(2*np.pi))**6), \
        "Likelihood not as expected"
    
    # Internally, the missing target values will be imputed using the
    # posterior. 
    Y_ref = mm.Y_.clone().detach()
    mm.Y_[0, 0] = np.nan
    tmp_2 = get_group_likelihood_samples(mm, num_samples=1000)
    assert torch.allclose(mm.Y_, Y_ref) and mm.Y_[0, 0] != Y_ref[0,0], \
        "Interpolation error"


def test_compute_waic2():
    mm = get_gt_model()
    waic2_ref = compute_waic2(mm)
    print(waic2_ref)
    
    # Modify the slope of traj 1, dim 1 by 5%
    mm.w_mu_[1, 1, 1] = 1.05*mm.w_mu_[1, 1, 1]
    waic2_test = compute_waic2(mm)
    print("{} should be bigger than {}".format(waic2_test, waic2_ref))
    assert waic2_test > waic2_ref, "Error in WAIC computation"
    
    # Modify the intercept of traj 1, dim 1 by 5%
    mm = get_gt_model()    
    mm.w_mu_[0, 0, 1] = 1.05*mm.w_mu_[0, 0, 1]
    waic2_test = compute_waic2(mm)
    print("{} should be bigger than {}".format(waic2_test, waic2_ref))
    assert waic2_test > waic2_ref, "Error in WAIC computation"

    # Modify R
    mm = get_gt_model()
    for gg in mm.gb_.groups:
        mm.R_[mm.gb_.get_group(gg).index, :] = \
            mm.R_[mm.gb_.get_group(gg).index, :] + \
            0.01*np.random.uniform(0.001, 0.999, mm.K_)

    row_sums = mm.R_.sum(dim=1, keepdim=True)
    mm.R_ = mm.R_/row_sums
    waic2_test = compute_waic2(mm)
    print("{} should be bigger than {}".format(waic2_test, waic2_ref))    
    if waic2_test <= waic2_ref:
        pdb.set_trace()
    assert waic2_test > waic2_ref, "Error in WAIC computation"

    # Modify lambda_a and lambda_b for traj 0, dim 0
    mm = get_gt_model()
    prec_var = mm.lambda_a_[0, 0]/(mm.lambda_b_[0, 0]**2)
    mu = 1/(0.5**2)
    mm.lambda_b_[0, 0] = mu/prec_var
    mm.lambda_a_[0, 0] = mu*mm.lambda_b_[0, 0]
    
    waic2_test = compute_waic2(mm)
    print("{} should be bigger than {}".format(waic2_test, waic2_ref))    
    if waic2_test <= waic2_ref:
        pdb.set_trace()
    assert waic2_test > waic2_ref, "Error in WAIC computation"

    # Modify lambda_a and lambda_b for traj 1, dim 1
    mm = get_gt_model()
    prec_var = mm.lambda_a_[1, 1]/(mm.lambda_b_[1, 1]**2)
    mu = 1/(0.5**2)
    mm.lambda_b_[1, 1] = mu/prec_var
    mm.lambda_a_[1, 1] = mu*mm.lambda_b_[1, 1]
    
    waic2_test = compute_waic2(mm)
    print("{} should be bigger than {}".format(waic2_test, waic2_ref))    
    if waic2_test <= waic2_ref:
        pdb.set_trace()
    assert waic2_test > waic2_ref, "Error in WAIC computation"
    
    # Modify w_var_[0, 0, 0]
    mm = get_gt_model()
    mm.w_var_[0, 0, 0] = .001
    
    waic2_test = compute_waic2(mm)
    print("{} should be bigger than {}".format(waic2_test, waic2_ref))    
    if waic2_test <= waic2_ref:
        pdb.set_trace()
    assert waic2_test > waic2_ref, "Error in WAIC computation"
    
    # Modify w_var_[1, 1, 1]
    mm = get_gt_model()
    mm.w_var_[1, 1, 1] = .00001
    
    waic2_test = compute_waic2(mm)
    print("{} should be bigger than {}".format(waic2_test, waic2_ref))    
    if waic2_test <= waic2_ref:
        pdb.set_trace()
    assert waic2_test > waic2_ref, "Error in WAIC computation"


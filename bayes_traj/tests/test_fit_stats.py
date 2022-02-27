import numpy as np
import pandas as pd
import pdb
from bayes_traj.mult_dp_regression import MultDPRegression
from bayes_traj.fit_stats import *

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
    
    mm = MultDPRegression(w_mu0, w_var0, lambda_a0, lambda_b0, alpha, K=K)
    mm.R_ = np.zeros([N, K])
    mm.R_[0, 0] = 1
    
    mm.w_mu_ = np.zeros([M, D, K])
    mm.w_var_ = np.ones([M, D, K])
    mm.w_mu_[:, 0, 0] = np.array([10, -1])
    mm.w_var_[:, 0, 0] = 1e-10*np.array([1, 1])

    mm.w_mu_[:, 0, 1] = np.array([11, -1])
    mm.w_var_[:, 0, 1] = 1e-10*np.array([1, 1])
    
    mm.lambda_a_ = np.ones([D, K])
    mm.lambda_b_ = np.ones([D, K])
    mm.lambda_a_[0, 0] = (prec_mu**2)/prec_var
    mm.lambda_b_[0, 0] = prec_mu/prec_var    
    
    mm.gb_ = df.groupby('sid')
    mm.X_ = df[['intercept', 'x']].values
    mm.Y_ = np.atleast_2d(df.y.values)
    mm.N_ = N 
    
    tmp = get_group_likelihood_samples(mm, num_samples=1000)
    assert np.isclose(np.mean(tmp), 1/np.sqrt(2*np.pi)), \
        "Likelihood not as expected"

    mm.R_[0, [0, 1]] = np.array([0, 1])
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
    
    mm = MultDPRegression(w_mu0, w_var0, lambda_a0, lambda_b0, alpha, K=K)
    mm.R_ = np.zeros([N, K])
    mm.R_[:, 0] = 1.

    mm.w_mu_ = np.zeros([M, D, K])
    mm.w_var_ = 1e-10*np.ones([M, D, K])
    mm.w_mu_[:, 0, 0] = np.array([10, -1])
    mm.w_mu_[:, 1, 0] = np.array([0, 1])    
    
    mm.lambda_a_ = np.ones([D, K])
    mm.lambda_b_ = np.ones([D, K])
    mm.lambda_a_[0, 0] = (prec_mu**2)/prec_var
    mm.lambda_b_[0, 0] = prec_mu/prec_var
    mm.lambda_a_[1, 0] = (prec_mu**2)/prec_var
    mm.lambda_b_[1, 0] = prec_mu/prec_var        
    
    mm.gb_ = df.groupby('sid')
    mm.X_ = df[['intercept', 'x']].values
    mm.Y_ = df[['y1', 'y2']].values
    mm.N_ = N 
    
    tmp_1 = get_group_likelihood_samples(mm, num_samples=1000)
    assert np.isclose(np.mean(tmp_1), (1/np.sqrt(2*np.pi))**6), \
        "Likelihood not as expected"

    # Internally, the missing target values will be imputed using the
    # posterior. This will not be the optimal value in this toy
    # data set (which should be 10). As such, we expect the mean from tmp_1
    # to be greater than the mean for tmp_2.
    mm.Y_[0, 0] = np.nan
    tmp_2 = get_group_likelihood_samples(mm, num_samples=1000)

    assert np.mean(tmp_1) > np.mean(tmp_2), "Unexpected likelihood comparison"

def test_compute_waic2_1():
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
    
    mm_1 = MultDPRegression(w_mu0, w_var0, lambda_a0, lambda_b0, alpha, K=K)
    mm_1.R_ = np.zeros([N, K])
    mm_1.R_[:, 0] = 1.

    mm_1.w_mu_ = np.zeros([M, D, K])
    mm_1.w_var_ = 1e-10*np.ones([M, D, K])
    mm_1.w_mu_[:, 0, 0] = np.array([10, -1])
    mm_1.w_mu_[:, 1, 0] = np.array([0, 1])    
    
    mm_1.lambda_a_ = np.ones([D, K])
    mm_1.lambda_b_ = np.ones([D, K])
    mm_1.lambda_a_[0, 0] = (prec_mu**2)/prec_var
    mm_1.lambda_b_[0, 0] = prec_mu/prec_var
    mm_1.lambda_a_[1, 0] = (prec_mu**2)/prec_var
    mm_1.lambda_b_[1, 0] = prec_mu/prec_var        
    
    mm_1.gb_ = df.groupby('sid')
    mm_1.X_ = df[['intercept', 'x']].values
    mm_1.Y_ = df[['y1', 'y2']].values
    mm_1.N_ = N 

    waic2_1 = compute_waic2(mm_1)

    mm_2 = MultDPRegression(w_mu0, w_var0, lambda_a0, lambda_b0, alpha, K=K)
    mm_2.R_ = np.zeros([N, K])
    mm_2.R_[:, 0] = 1.

    mm_2.w_mu_ = np.zeros([M, D, K])
    mm_2.w_var_ = 1e-10*np.ones([M, D, K])
    mm_2.w_mu_[:, 0, 0] = np.array([11, -1]) # Poorer value
    mm_2.w_mu_[:, 1, 0] = np.array([0, 1])    
    
    mm_2.lambda_a_ = np.ones([D, K])
    mm_2.lambda_b_ = np.ones([D, K])
    mm_2.lambda_a_[0, 0] = (prec_mu**2)/prec_var
    mm_2.lambda_b_[0, 0] = prec_mu/prec_var
    mm_2.lambda_a_[1, 0] = (prec_mu**2)/prec_var
    mm_2.lambda_b_[1, 0] = prec_mu/prec_var        
    
    mm_2.gb_ = df.groupby('sid')
    mm_2.X_ = df[['intercept', 'x']].values
    mm_2.Y_ = df[['y1', 'y2']].values
    mm_2.N_ = N 

    waic2_2 = compute_waic2(mm_2)
    assert waic2_1 < waic2_2, "Unexpect WAIC2 comparison"


import torch
from bayes_traj.mult_dp_regression import MultDPRegression
import numpy as np
import pandas as pd
from bayes_traj.utils import *
import pdb, os, pickle

def test_update_u():
    """
    Create synthetic data: 5 individuals with 5 time points each. Two 
    trajectories, and each individual is associated with random effect
    vectors drawn from multivariate normal distributions.
    """
    np.random.seed(42)
    n_long_pts = 5
    D = 1
    M = 2
    K = 3
    G = 5
    N = G*n_long_pts

    mu1 = np.array([10, -3/5])
    mu2 = np.array([-10, -3/10])
    covmat = np.array([[ 0.03      , -0.02999999],
                       [-0.02999999,  0.03      ]])

    n_dom_pts = 100
    n_samples1 = 3
    n_samples2 = 2
    dom = np.linspace(0, 10, n_dom_pts)
    samples1 = np.random.multivariate_normal(mean=mu1,
        cov=covmat, size=n_samples1)
    samples2 = np.random.multivariate_normal(mean=mu2,
        cov=covmat, size=n_samples2)
    sig1 = .2
    sig2 = .15

    X = np.ones([n_long_pts, 2])
    X[:, 1] = np.linspace(0, 10, n_long_pts)
    y = np.zeros(N)
    for ii in range(n_samples1):
        y[ii*n_long_pts:(ii+1)*n_long_pts] = \
            np.dot(X, samples1[ii, :]) + sig1*np.random.randn(n_long_pts)

    for ii in range(n_samples2):
        y[(ii+n_samples1)*n_long_pts:(ii+n_samples1+1)*n_long_pts] = \
            np.dot(X, samples2[ii, :]) + sig2*np.random.randn(n_long_pts)

    target_names = ['y']
    predictor_names = ['intercept', 'x']
    df = pd.DataFrame()
    df['intercept'] = np.ones(N)
    df['id'] = ['a']*n_long_pts + ['b']*n_long_pts + ['c']*n_long_pts + \
        ['d']*n_long_pts + ['e']*n_long_pts
    df['x'] = [0, 2.5, 5, 7.5, 10]*G
    df['y'] = y

    R = np.zeros([N, K])
    R[0:15, 0] = 1
    R[15::, 1] = 1

    alpha = 1.242 # For 5 individuals and 2 exptected trajectories
    
    w_mu0 = np.zeros([M, D])
    w_mu0[0, 0] = np.mean(np.array([mu1[0], mu2[0]]))
    w_mu0[1, 0] = np.mean(np.array([mu1[1], mu2[1]]))

    w_var0 = np.ones([M, D])
    w_var0[0, 0] = np.var(np.array([mu1[0], mu2[0]]))
    w_var0[1, 0] = np.var(np.array([mu1[1], mu2[1]]))

    prec_weight = 1e1
    lambda_a0 = prec_weight*35*np.ones([D])
    lambda_b0 = prec_weight*np.ones([D])
    Sig0 = {'y': 1e0*torch.from_numpy(covmat)}
    ranef_indices = np.ones(M, dtype=bool)

    mm = MultDPRegression(w_mu0,
                          w_var0,
                          lambda_a0,
                          lambda_b0, 1,
                          alpha, K=K,
                          Sig0=Sig0,
                          ranef_indices=ranef_indices,
                          prob_thresh=0.001)

    groupby = 'id'
    iters = 0
    v_a = None
    v_b = None

    w_mu = np.zeros([M, D, K])
    w_mu[:, 0, 0] = mu1
    w_mu[:, 0, 1] = mu2

    w_var = 1e-10*np.ones([M, D, K])

    lambda_a = np.ones([D, K])
    lambda_b = np.ones([D, K])
    lambda_a[0, 0] = 100*(1/sig1**2)
    lambda_b[0, 0] = 100
    lambda_a[0, 1] = 100*(1/sig2**2)
    lambda_b[0, 1] = 100

    mm.fit(target_names = target_names,
           predictor_names = predictor_names,
           df = df,
           groupby = 'id',
           iters = 0,
           R = np.array(R),
           v_a = None, #v_a,
           v_b = None, #v_b,
           w_mu = np.array(w_mu),
           w_var = np.array(w_var),
           lambda_a = np.array(lambda_a),
           lambda_b = np.array(lambda_b),
           verbose=True)
    mm.update_u()

    ranefs_gt = \
        np.array([[[[-0.04736315,  0.04736312],
                    [ 0.        ,  0.        ],
                    [ 0.        ,  0.        ]]],


                  [[[-0.08581572,  0.08581566],
                    [ 0.        ,  0.        ],
                    [ 0.        ,  0.        ]]],


                  [[[ 0.06202493, -0.0620248 ],
                    [ 0.        ,  0.        ],
                    [ 0.        ,  0.        ]]],


                  [[[ 0.        ,  0.        ],
                    [-0.26497722,  0.26497726],
                    [ 0.        ,  0.        ]]],

                  
                  [[[ 0.        ,  0.        ],
                    [ 0.07680525, -0.07680522],
                    [ 0.        ,  0.        ]]]])

    assert np.allclose(ranefs_gt, mm.u_mu_.numpy()), \
        "Random effects not as expected"

def test_update_lambda():
    """
    Create synthetic data: 5 individuals with 5 time points each. Two 
    trajectories, and each individual is associated with random effect
    vectors drawn from multivariate normal distributions.
    """
    np.random.seed(42)
    n_long_pts = 5
    D = 1
    M = 2
    K = 3
    G = 5
    N = G*n_long_pts

    mu1 = np.array([10, -3/5])
    mu2 = np.array([-10, -3/10])
    covmat = np.array([[ 0.03      , -0.02999999],
                       [-0.02999999,  0.03      ]])

    n_dom_pts = 100
    n_samples1 = 3
    n_samples2 = 2
    dom = np.linspace(0, 10, n_dom_pts)
    samples1 = np.random.multivariate_normal(mean=mu1,
        cov=covmat, size=n_samples1)
    samples2 = np.random.multivariate_normal(mean=mu2,
        cov=covmat, size=n_samples2)
    sig1 = .2
    sig2 = .15

    X = np.ones([n_long_pts, 2])
    X[:, 1] = np.linspace(0, 10, n_long_pts)
    y = np.zeros(N)
    for ii in range(n_samples1):
        y[ii*n_long_pts:(ii+1)*n_long_pts] = \
            np.dot(X, samples1[ii, :]) + sig1*np.random.randn(n_long_pts)

    for ii in range(n_samples2):
        y[(ii+n_samples1)*n_long_pts:(ii+n_samples1+1)*n_long_pts] = \
            np.dot(X, samples2[ii, :]) + sig2*np.random.randn(n_long_pts)

    target_names = ['y']
    predictor_names = ['intercept', 'x']
    df = pd.DataFrame()
    df['intercept'] = np.ones(N)
    df['id'] = ['a']*n_long_pts + ['b']*n_long_pts + ['c']*n_long_pts + \
        ['d']*n_long_pts + ['e']*n_long_pts
    df['x'] = [0, 2.5, 5, 7.5, 10]*G
    df['y'] = y

    R = np.zeros([N, K])
    R[0:15, 0] = 1
    R[15::, 1] = 1

    alpha = 1.242 # For 5 individuals and 2 exptected trajectories
    
    w_mu0 = np.zeros([M, D])
    w_mu0[0, 0] = np.mean(np.array([mu1[0], mu2[0]]))
    w_mu0[1, 0] = np.mean(np.array([mu1[1], mu2[1]]))

    w_var0 = np.ones([M, D])
    w_var0[0, 0] = np.var(np.array([mu1[0], mu2[0]]))
    w_var0[1, 0] = np.var(np.array([mu1[1], mu2[1]]))

    prec_weight = 1e1
    lambda_a0 = prec_weight*35*np.ones([D])
    lambda_b0 = prec_weight*np.ones([D])
    Sig0 = {'y': 1e0*torch.from_numpy(covmat)}
    ranef_indices = np.ones(M, dtype=bool)

    mm = MultDPRegression(w_mu0,
                          w_var0,
                          lambda_a0,
                          lambda_b0, 1,
                          alpha, K=K,
                          Sig0=Sig0,
                          ranef_indices=ranef_indices,
                          prob_thresh=0.001)

    groupby = 'id'
    iters = 0
    v_a = None
    v_b = None

    w_mu = np.zeros([M, D, K])
    w_mu[:, 0, 0] = mu1
    w_mu[:, 0, 1] = mu2

    w_var = 1e-10*np.ones([M, D, K])

    lambda_a = np.ones([D, K])
    lambda_b = np.ones([D, K])
    lambda_a[0, 0] = 100*(1/sig1**2)
    lambda_b[0, 0] = 100
    lambda_a[0, 1] = 100*(1/sig2**2)
    lambda_b[0, 1] = 100

    u_mu = \
        torch.from_numpy(np.array([[[[-0.04736315,  0.04736312],
                                     [ 0.        ,  0.        ],
                                     [ 0.        ,  0.        ]]],
                               

                                   [[[-0.08581572,  0.08581566],
                                     [ 0.        ,  0.        ],
                                     [ 0.        ,  0.        ]]],
                           
                                   
                                   [[[ 0.06202493, -0.0620248 ],
                                     [ 0.        ,  0.        ],
                                     [ 0.        ,  0.        ]]],
                           

                                   [[[ 0.        ,  0.        ],
                                     [-0.26497722,  0.26497726],
                                     [ 0.        ,  0.        ]]],


                                   [[[ 0.        ,  0.        ],
                                     [ 0.07680525, -0.07680522],
                                     [ 0.        ,  0.        ]]]]))
    
    mm.fit(target_names = target_names,
           predictor_names = predictor_names,
           df = df,
           groupby = 'id',
           iters = 0,
           R = np.array(R),
           v_a = None, #v_a,
           v_b = None, #v_b,
           w_mu = np.array(w_mu),
           w_var = np.array(w_var),
           lambda_a = None, 
           lambda_b = None, 
           verbose=True)
    mm.u_mu_ = u_mu

    mm.update_lambda()

    # The following values were empirically computed
    traj_0_measured_var = 0.01768667556329983
    traj_1_measured_var = 0.01595407361096874

    # The effect of of the prior is subtracted away so we can evaluate how
    # well the update estimates the residual variance from the data
    traj_0_est_var = \
        ((mm.lambda_b_[0, 0] - mm.lambda_b0_[0])/(mm.lambda_a_[0, 0] - \
                                                  mm.lambda_a0_[0])).numpy()
    traj_1_est_var = \
        ((mm.lambda_b_[0, 1] - mm.lambda_b0_[0])/(mm.lambda_a_[0, 1] - \
                                                  mm.lambda_a0_[0])).numpy()
    assert np.isclose(traj_0_est_var, traj_0_measured_var, rtol=0, atol=1e-5), \
        "Estimate precision variance not as expected"
    assert np.isclose(traj_1_est_var, traj_1_measured_var, rtol=0, atol=1e-5), \
        "Estimate precision variance not as expected"    

def test_update_w_gaussian():
    """
    Create synthetic data: 5 individuals with 5 time points each. Two 
    trajectories, and each individual is associated with random effect
    vectors drawn from multivariate normal distributions.
    """
    np.random.seed(42)
    n_long_pts = 5
    D = 1
    M = 2
    K = 3
    G = 5
    N = G*n_long_pts

    mu1 = np.array([10, -3/5])
    mu2 = np.array([-10, -3/10])
    covmat = np.array([[ 0.03      , -0.02999999],
                       [-0.02999999,  0.03      ]])

    n_dom_pts = 100
    n_samples1 = 3
    n_samples2 = 2
    dom = np.linspace(0, 10, n_dom_pts)
    samples1 = np.random.multivariate_normal(mean=mu1,
        cov=covmat, size=n_samples1)
    samples2 = np.random.multivariate_normal(mean=mu2,
        cov=covmat, size=n_samples2)
    sig1 = .2
    sig2 = .15

    X = np.ones([n_long_pts, 2])
    X[:, 1] = np.linspace(0, 10, n_long_pts)
    y = np.zeros(N)
    for ii in range(n_samples1):
        y[ii*n_long_pts:(ii+1)*n_long_pts] = \
            np.dot(X, samples1[ii, :]) + sig1*np.random.randn(n_long_pts)

    for ii in range(n_samples2):
        y[(ii+n_samples1)*n_long_pts:(ii+n_samples1+1)*n_long_pts] = \
            np.dot(X, samples2[ii, :]) + sig2*np.random.randn(n_long_pts)

    target_names = ['y']
    predictor_names = ['intercept', 'x']
    df = pd.DataFrame()
    df['intercept'] = np.ones(N)
    df['id'] = ['a']*n_long_pts + ['b']*n_long_pts + ['c']*n_long_pts + \
        ['d']*n_long_pts + ['e']*n_long_pts
    df['x'] = [0, 2.5, 5, 7.5, 10]*G
    df['y'] = y

    R = np.zeros([N, K])
    R[0:15, 0] = 1
    R[15::, 1] = 1

    alpha = 1.242 # For 5 individuals and 2 exptected trajectories
    
    w_mu0 = np.zeros([M, D])
    w_mu0[0, 0] = np.mean(np.array([mu1[0], mu2[0]]))
    w_mu0[1, 0] = np.mean(np.array([mu1[1], mu2[1]]))

    w_var0 = np.ones([M, D])
    w_var0[0, 0] = np.var(np.array([mu1[0], mu2[0]]))
    w_var0[1, 0] = np.var(np.array([mu1[1], mu2[1]]))

    prec_weight = 1e1
    lambda_a0 = prec_weight*35*np.ones([D])
    lambda_b0 = prec_weight*np.ones([D])
    Sig0 = {'y': 1e0*torch.from_numpy(covmat)}
    ranef_indices = np.ones(M, dtype=bool)

    lambda_a = np.array([[42.5000, 40.0000, 35.0000]])
    lambda_b = np.array([[1.1327, 1.0798, 1.0000]])

    u_mu = \
        torch.from_numpy(np.array([[[[-0.04736315,  0.04736312],
                                     [ 0.        ,  0.        ],
                                     [ 0.        ,  0.        ]]],
                               

                                   [[[-0.08581572,  0.08581566],
                                     [ 0.        ,  0.        ],
                                     [ 0.        ,  0.        ]]],
                           
                                   
                                   [[[ 0.06202493, -0.0620248 ],
                                     [ 0.        ,  0.        ],
                                     [ 0.        ,  0.        ]]],
                           

                                   [[[ 0.        ,  0.        ],
                                     [-0.26497722,  0.26497726],
                                     [ 0.        ,  0.        ]]],


                                   [[[ 0.        ,  0.        ],
                                     [ 0.07680525, -0.07680522],
                                     [ 0.        ,  0.        ]]]]))
    mm = MultDPRegression(w_mu0,
                          w_var0,
                          lambda_a0,
                          lambda_b0, 1,
                          alpha, K=K,
                          Sig0=Sig0,
                          ranef_indices=ranef_indices,
                          prob_thresh=0.001)
    
    mm.fit(target_names = target_names,
           predictor_names = predictor_names,
           df = df,
           groupby = 'id',
           iters = 0,
           R = np.array(R),
           v_a = None, #v_a,
           v_b = None, #v_b,
           w_mu = None,
           w_var = None,
           lambda_a = lambda_a, 
           lambda_b = lambda_b, 
           verbose=True)
    mm.u_mu_ = u_mu

    for i in range(20):
        mm.update_w_gaussian()

    # The following GT values were evaluated against fixed effects determined by
    # linear mixed modeling (in which random slopes and random intercepts were
    # estimated). In fact, for trajectory 1, the slope was found to be non-
    # significant in that analysis, which is not surprising given that only two
    # individuals are in that trajectory. The slope in trajectory 1 in the
    # ground truth is with in the confidence interval.
    w_mu_gt = \
        np.array([[[ 9.99561852, -9.99730782,  0.        ]],
                  [[-0.5989    , -0.30046008, -0.45      ]]])
    
    assert np.allclose(w_mu_gt, mm.w_mu_.numpy()), \
        "w_mu values not as expected"

def test_update_z():
    """
    Create synthetic data: 5 individuals with 5 time points each. Two 
    trajectories, and each individual is associated with random effect
    vectors drawn from multivariate normal distributions.
    """
    np.random.seed(42)
    n_long_pts = 5
    D = 1
    M = 2
    K = 3
    G = 5
    N = G*n_long_pts

    mu1 = np.array([10, -3/5])
    mu2 = np.array([-10, -3/10])
    covmat = np.array([[ 0.03      , -0.02999999],
                       [-0.02999999,  0.03      ]])

    n_dom_pts = 100
    n_samples1 = 3
    n_samples2 = 2
    dom = np.linspace(0, 10, n_dom_pts)
    samples1 = np.random.multivariate_normal(mean=mu1,
        cov=covmat, size=n_samples1)
    samples2 = np.random.multivariate_normal(mean=mu2,
        cov=covmat, size=n_samples2)
    sig1 = .2
    sig2 = .15

    X = np.ones([n_long_pts, 2])
    X[:, 1] = np.linspace(0, 10, n_long_pts)
    y = np.zeros(N)
    for ii in range(n_samples1):
        y[ii*n_long_pts:(ii+1)*n_long_pts] = \
            np.dot(X, samples1[ii, :]) + sig1*np.random.randn(n_long_pts)

    for ii in range(n_samples2):
        y[(ii+n_samples1)*n_long_pts:(ii+n_samples1+1)*n_long_pts] = \
            np.dot(X, samples2[ii, :]) + sig2*np.random.randn(n_long_pts)

    target_names = ['y']
    predictor_names = ['intercept', 'x']
    df = pd.DataFrame()
    df['intercept'] = np.ones(N)
    df['id'] = ['a']*n_long_pts + ['b']*n_long_pts + ['c']*n_long_pts + \
        ['d']*n_long_pts + ['e']*n_long_pts
    df['x'] = [0, 2.5, 5, 7.5, 10]*G
    df['y'] = y

    alpha = 1.242 # For 5 individuals and 2 exptected trajectories
    
    w_mu0 = np.zeros([M, D])
    w_mu0[0, 0] = np.mean(np.array([mu1[0], mu2[0]]))
    w_mu0[1, 0] = np.mean(np.array([mu1[1], mu2[1]]))

    w_var0 = np.ones([M, D])
    w_var0[0, 0] = np.var(np.array([mu1[0], mu2[0]]))
    w_var0[1, 0] = np.var(np.array([mu1[1], mu2[1]]))

    prec_weight = 1e1
    lambda_a0 = prec_weight*35*np.ones([D])
    lambda_b0 = prec_weight*np.ones([D])
    Sig0 = {'y': 1e0*torch.from_numpy(covmat)}
    ranef_indices = np.ones(M, dtype=bool)

    lambda_a = np.array([[42.5000, 40.0000, 35.0000]])
    lambda_b = np.array([[1.1327, 1.0798, 1.0000]])

    w_mu = \
        np.array([[[ 9.99561852, -9.99730782,  0.  ]],
                  [[-0.5989    , -0.30046008, -0.45]]])

    w_var = 1e-10*np.ones([M, D, K])
    
    u_mu = \
        torch.from_numpy(np.array([[[[-0.04736315,  0.04736312],
                                     [ 0.        ,  0.        ],
                                     [ 0.        ,  0.        ]]],
                               

                                   [[[-0.08581572,  0.08581566],
                                     [ 0.        ,  0.        ],
                                     [ 0.        ,  0.        ]]],
                           
                                   
                                   [[[ 0.06202493, -0.0620248 ],
                                     [ 0.        ,  0.        ],
                                     [ 0.        ,  0.        ]]],
                           

                                   [[[ 0.        ,  0.        ],
                                     [-0.26497722,  0.26497726],
                                     [ 0.        ,  0.        ]]],


                                   [[[ 0.        ,  0.        ],
                                     [ 0.07680525, -0.07680522],
                                     [ 0.        ,  0.        ]]]]))

    v_a = torch.tensor([4., 3., 1.])
    v_b = torch.tensor([3.2420, 1.2420, 1.2420])
    
    mm = MultDPRegression(w_mu0,
                          w_var0,
                          lambda_a0,
                          lambda_b0, 1,
                          alpha, K=K,
                          Sig0=Sig0,
                          ranef_indices=ranef_indices,
                          prob_thresh=0.001)
    
    mm.fit(target_names = target_names,
           predictor_names = predictor_names,
           df = df,
           groupby = 'id',
           iters = 0,
           R = None,
           v_a = v_a,
           v_b = v_b,
           w_mu = np.array(w_mu),
           w_var = np.array(w_var),
           lambda_a = lambda_a, 
           lambda_b = lambda_b, 
           verbose=True)
    mm.u_mu_ = u_mu

    R = mm.update_z(mm.X_, mm.Y_).numpy()
    
    R_gt = np.zeros([N, K])
    R_gt[0:15, 0] = 1
    R_gt[15::, 1] = 1

    assert np.allclose(R, R_gt), \
        "R matrix not as expected"

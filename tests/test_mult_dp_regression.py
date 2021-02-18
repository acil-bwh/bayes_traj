from bayes_traj.mult_dp_regression import MultDPRegression
from bayes_traj.get_longitudinal_constraints_graph \
    import get_longitudinal_constraints_graph
import numpy as np
import pandas as pd
import pdb, os

def test_MultDPRegression():
    # Read data from resources dir
    data_file_name = os.path.split(os.path.realpath(__file__))[0] + \
        '/../resources/data/trajectory_data_1.csv'
    df = pd.read_csv(data_file_name)
    

    prior_file_name = os.path.split(os.path.realpath(__file__))[0] + \
        '/../resources/priors/trajectory_prior_1.p'
    
    # Read prior from resources dir
    
    
    #mm = MultDPRegression(prior_data['w_mu0'], prior_data['w_var0'],
    #                      prior_data['lambda_a0'], prior_data['lambda_b0'],
    #                      prior_data['alpha'], K=K)

    #mm.fit(X, Y, iters=iters, verbose=op.verbose,
    #       constraints=constraints_graph, data_names=data_names,
    #       target_names=targets, predictor_names=preds,
    #       traj_probs=prior_data['traj_probs'],
    #       traj_probs_weight=prior_data['probs_weight'],
    #       v_a=prior_data['v_a'], v_b=prior_data['v_b'], w_mu=prior_data['w_mu'],
    #       w_var=prior_data['w_var'], lambda_a=prior_data['lambda_a'],
    #       lambda_b=prior_data['lambda_b'])

def test_init_R_mat():
    """
    """
    # Construct some synthetic data: three trajectories with three different
    # intercepts and slopes
    num_per_traj = 10
    x = np.linspace(0, 10, num_per_traj)
    m1 = 1; b1 = 29; std1 = 0.3
    m2 = 0; b2 = 26; std2 = 0.01
    m3 = -1; b3 = 23; std3 = 0.5
    y1 = m1*x + b1 + std1*np.random.randn(num_per_traj)
    y2 = m2*x + b2 + std2*np.random.randn(num_per_traj)
    y3 = m3*x + b3 + std3*np.random.randn(num_per_traj)

    X_mat = np.ones([num_per_traj*3, 2])
    X_mat[:, 1] = np.vstack([x, x, x]).reshape(-1)
    Y_mat = np.atleast_2d(np.vstack([y1, y2, y3]).reshape(-1)).T
    
    M = X_mat.shape[1]
    D = Y_mat.shape[1]
    N = Y_mat.shape[0]
    
    w_mu0 = np.zeros([M, D])
    w_var0 = np.ones([M, D])
    lambda_a0 = np.ones(D)
    lambda_b0 = np.ones(D)
    alpha = 5
    K = 30
    mm = MultDPRegression(w_mu0, w_var0, lambda_a0, lambda_b0, alpha, K)

    lambda_a = np.ones([D, K])
    lambda_b = np.ones([D, K])
    w_mu = np.zeros([M, D, K])
    w_mu[:, 0, 0] = np.array([b1, m1])
    w_mu[:, 0, 1] = np.array([b2, m2])
    w_mu[:, 0, 2] = np.array([b3, m3])    
    w_var = np.ones([M, D, K])
    for d in range(D):
        lambda_a[d, :] = lambda_a0[d]
        lambda_b[d, :] = lambda_b0[d]
    lambda_a[0, 0] = 1
    lambda_b[0, 0] = .3**2
    lambda_a[0, 1] = 1
    lambda_b[0, 1] = .01**2
    lambda_a[0, 2] = 1
    lambda_b[0, 2] = .5**2     
        
    mm.w_mu_ = w_mu
    mm.w_var = w_var
    mm.lambda_a_ = lambda_a
    mm.lambda_b_ = lambda_b    
    mm.N_ = N
    mm.X_ = X_mat
    mm.Y_ = Y_mat
    constraints = None
    traj_probs = np.zeros(K)
    traj_probs[0] = .25
    traj_probs[1] = .25
    traj_probs[2] = .25
    traj_probs[3] = .25    

    mm.init_R_mat(constraints, traj_probs, traj_probs_weight=1)
    assert np.sum(np.isclose(np.ones(K), np.sum(mm.R_, 1))) == K, \
        "Unexpected R_ sum"

    traj_assignments = np.array([np.where(mm.R_[i, :] == \
        np.max(mm.R_[i, :]))[0][0] for i in range(N)])
    assert np.sum(traj_assignments == \
                  np.array([0]*10 + [1]*10 + [2]*10)) == 30, \
                  "Unexpected trajectory assignments"

def test_predict_proba():
    """
    """
    D = 1
    M = 2
    K = 2    
    w_mu0 = np.zeros([M, D])
    w_var0 = np.ones([M, D])
    lambda_a0 = np.ones(D)
    lambda_b0 = np.ones(D)
    alpha = 5
    mm = MultDPRegression(w_mu0, w_var0, lambda_a0, lambda_b0, alpha, K)

    mm.w_mu_ = np.zeros([M, D, K])
    mm.w_mu_[:, 0, 0] = np.array([2, 1])
    mm.w_mu_[:, 0, 1] = np.array([-2, -1])    
    mm.R_ = np.array([[.5, .5]])
    mm.lambda_b_ = np.ones([D, K])
    mm.lambda_a_ = np.ones([D, K])

    X = np.array([[1, 2], [1, 2], [1, 2]])
    Y = np.array([[3], [0], [-3]])
    
    R = mm.predict_proba(X, Y)
    R_ref = np.array([[1.00000000e+00, 3.77513454e-11],
                      [5.00000000e-01, 5.00000000e-01],
                      [3.77513454e-11, 1.00000000e+00]])
    assert np.sum(np.isclose(R, R_ref)) == 6, "Unexpected R value"

    constraints = get_longitudinal_constraints_graph(np.array([0, 2, 0]))
    R = mm.predict_proba(X, Y, constraints)
    R_ref = np.array([[0.5, 0.5],
                      [0.5, 0.5],
                      [0.5, 0.5]])
    assert np.sum(np.isclose(R, R_ref)) == 6, "Unexpected R value"


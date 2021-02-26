from bayes_traj.mult_dp_regression import MultDPRegression
import numpy as np
import pandas as pd
from bayes_traj.utils import *
import pdb, os

np.set_printoptions(precision = 10, suppress = True, threshold=1e6,
                    linewidth=300)

def test_MultDPRegression():
    # Read data from resources dir
    data_file_name = os.path.split(os.path.realpath(__file__))[0] + \
        '/../resources/data/trajectory_data_1.csv'
    df = pd.read_csv(data_file_name)
    
    prior_file_name = os.path.split(os.path.realpath(__file__))[0] + \
        '/../resources/priors/trajectory_prior_1.p'
    
    # Read prior from resources dir
    with open(prior_file_name, 'rb') as f:
        prior_file_info = pickle.load(f)
        
    preds = get_pred_names_from_prior_info(prior_file_info)
    targets = get_target_names_from_prior_info(prior_file_info)        

    D = len(targets)
    M = len(preds)
    K = 20

    prior_data = {}
    prior_data['w_mu0'] = np.zeros([M, D])
    prior_data['w_var0'] = np.ones([M, D])
    prior_data['lambda_a0'] = np.ones([D])
    prior_data['lambda_b0'] = np.ones([D])
    prior_data['alpha'] = prior_file_info['alpha']
    for (d, target) in enumerate(targets):
        prior_data['lambda_a0'][d] = prior_file_info['lambda_a0'][target]
        prior_data['lambda_b0'][d] = prior_file_info['lambda_b0'][target]            
            
        for (m, pred) in enumerate(preds):
            prior_data['w_mu0'][m, d] = prior_file_info['w_mu0'][target][pred]
            prior_data['w_var0'][m, d] = prior_file_info['w_var0'][target][pred]            
    
    mm = MultDPRegression(prior_data['w_mu0'], prior_data['w_var0'],
                          prior_data['lambda_a0'], prior_data['lambda_b0'],
                          prior_data['alpha'], K=K)

    mm.fit(target_names=targets, predictor_names=preds, df=df, groupby='id',
           iters=10, verbose=True)

    df_traj = mm.to_df()

    num_trajs_found = np.sum(np.where(pd.crosstab(df_traj.traj.values,
                                    df.traj.values).values == 250))

    assert num_trajs_found == 2, "Trajectory assignment error"
    pdb.set_trace()

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

def test_update_lambda_batch():
        # Read data from resources dir
    data_file_name = os.path.split(os.path.realpath(__file__))[0] + \
        '/../resources/data/trajectory_data_1.csv'
    df = pd.read_csv(data_file_name)
    
    prior_file_name = os.path.split(os.path.realpath(__file__))[0] + \
        '/../resources/priors/trajectory_prior_1.p'
    
    # Read prior from resources dir
    with open(prior_file_name, 'rb') as f:
        prior_file_info = pickle.load(f)
        
    preds = get_pred_names_from_prior_info(prior_file_info)
    targets = get_target_names_from_prior_info(prior_file_info)        

    D = len(targets)
    M = len(preds)
    K = 20

    prior_data = {}
    prior_data['w_mu0'] = np.zeros([M, D])
    prior_data['w_var0'] = np.ones([M, D])
    prior_data['lambda_a0'] = np.ones([D])
    prior_data['lambda_b0'] = np.ones([D])
    prior_data['alpha'] = prior_file_info['alpha']
    for (d, target) in enumerate(targets):
        prior_data['lambda_a0'][d] = prior_file_info['lambda_a0'][target]
        prior_data['lambda_b0'][d] = prior_file_info['lambda_b0'][target]            
            
        for (m, pred) in enumerate(preds):
            prior_data['w_mu0'][m, d] = prior_file_info['w_mu0'][target][pred]
            prior_data['w_var0'][m, d] = prior_file_info['w_var0'][target][pred]            
    
    mm = MultDPRegression(prior_data['w_mu0'], prior_data['w_var0'],
                          prior_data['lambda_a0'], prior_data['lambda_b0'],
                          prior_data['alpha'], K=K)

    mm.X_ = df[preds].values
    mm.Y_ = df[targets].values
    mm.N_ = df.shape[0]
    mm.batch_indices_ = np.ones(mm.N_, dtype=bool)
    mm.sig_trajs_ = np.zeros(mm.K_, dtype=bool)
    mm.sig_trajs_[0] = True
    mm.sig_trajs_[1] = True
    mm.R_ = np.zeros([mm.N_, mm.K_])
    mm.R_[0:250, 0] = 1
    mm.R_[250:500, 1] = 1
    
    mm.w_mu_ = np.zeros([mm.M_, mm.D_, mm.K_])
    mm.w_mu_[:, 0, 0] = np.array([10, -1])
    mm.w_mu_[:, 0, 1] = np.array([6, -2])

    mm.w_var_ = 1e-20*np.ones([mm.M_, mm.D_, mm.K_])

    mm.gb_ = df.groupby('id')
    
    mm.batch_size_ = 50 # Number of individuals
    indicator = np.zeros(mm.gb_.ngroups)
    indicator[0:mm.batch_size_] = 1

    mm.batch_indices_ = np.zeros(mm.N_, dtype=bool)
    mm.batch_subjects_ = np.array([ 6, 8, 9, 10, 11, 13, 14, 18, 19, 20, 21, \
                                    22, 25, 26, 29, 31, 35, 37, 38, 40, 41, \
                                    44, 45, 46, 47, 50, 52, 53, 55, 57, 58, \
                                    61, 62, 64, 66, 68, 71, 72, 75, 78, 80, \
                                    82, 83, 90, 92, 94, 95, 96, 98, 100])    
        
    for ww in mm.batch_subjects_:
            mm.batch_indices_[mm.gb_.get_group(ww).index] = True
    
    lambda_a, lambda_b = mm.update_lambda_batch(mm.w_mu_, mm.w_var_)
    pdb.set_trace()
    assert np.isclose(np.sqrt(lambda_b/lambda_a)[0, 0], 2, atol=.05), \
        "Residual variance not calculated correctly"
    assert np.isclose(np.sqrt(lambda_b/lambda_a)[0, 1], 1, atol=.05), \
        "Residual variance not calculated correctly"    

def test_update_w_batch():
    # Read data from resources dir
    data_file_name = os.path.split(os.path.realpath(__file__))[0] + \
        '/../resources/data/trajectory_data_1.csv'
    df = pd.read_csv(data_file_name)
    
    prior_file_name = os.path.split(os.path.realpath(__file__))[0] + \
        '/../resources/priors/trajectory_prior_1.p'
    
    # Read prior from resources dir
    with open(prior_file_name, 'rb') as f:
        prior_file_info = pickle.load(f)
        
    preds = get_pred_names_from_prior_info(prior_file_info)
    targets = get_target_names_from_prior_info(prior_file_info)        

    D = len(targets)
    M = len(preds)
    K = 20

    prior_data = {}
    prior_data['w_mu0'] = np.zeros([M, D])
    prior_data['w_var0'] = np.ones([M, D])
    prior_data['lambda_a0'] = np.ones([D])
    prior_data['lambda_b0'] = np.ones([D])
    prior_data['alpha'] = prior_file_info['alpha']
    for (d, target) in enumerate(targets):
        prior_data['lambda_a0'][d] = prior_file_info['lambda_a0'][target]
        prior_data['lambda_b0'][d] = prior_file_info['lambda_b0'][target]            
            
        for (m, pred) in enumerate(preds):
            prior_data['w_mu0'][m, d] = prior_file_info['w_mu0'][target][pred]
            prior_data['w_var0'][m, d] = prior_file_info['w_var0'][target][pred]            
    
    mm = MultDPRegression(prior_data['w_mu0'], prior_data['w_var0'],
                          prior_data['lambda_a0'], prior_data['lambda_b0'],
                          prior_data['alpha'], K=K)

    mm.X_ = df[preds].values
    mm.Y_ = df[targets].values
    mm.N_ = df.shape[0]
    mm.batch_indices_ = np.ones(mm.N_, dtype=bool)
    mm.sig_trajs_ = np.zeros(mm.K_, dtype=bool)
    mm.sig_trajs_[0] = True
    mm.sig_trajs_[1] = True
    mm.R_ = np.zeros([mm.N_, mm.K_])
    mm.R_[0:250, 0] = 1
    mm.R_[250:500, 1] = 1
    
    mm.w_mu_ = np.zeros([mm.M_, mm.D_, mm.K_])
    mm.w_mu_[:, 0, 0] = np.array([10, -1])
    mm.w_mu_[:, 0, 1] = np.array([6, -2])

    mm.w_var_ = np.ones([mm.M_, mm.D_, mm.K_])/10000.
   
    mm.gb_ = df.groupby('id')
    
    mm.batch_size_ = 50 # Number of individuals
    indicator = np.zeros(mm.gb_.ngroups)
    indicator[0:mm.batch_size_] = 1

    mm.batch_indices_ = np.zeros(mm.N_, dtype=bool)
    mm.batch_subjects_ = np.array([ 6, 8, 9, 10, 11, 13, 14, 18, 19, 20, 21, \
                                    22, 25, 26, 29, 31, 35, 37, 38, 40, 41, \
                                    44, 45, 46, 47, 50, 52, 53, 55, 57, 58, \
                                    61, 62, 64, 66, 68, 71, 72, 75, 78, 80, \
                                    82, 83, 90, 92, 94, 95, 96, 98, 100])    
        
    for ww in mm.batch_subjects_:
            mm.batch_indices_[mm.gb_.get_group(ww).index] = True
            
    mm.lambda_a_ = np.ones([mm.D_, mm.K_])
    mm.lambda_a_[0, 0:2] = 150
    mm.lambda_b_ = np.ones([mm.D_, mm.K_])
    mm.lambda_b_[0, 0] = 600
    mm.lambda_b_[0, 1] = 150

    w_mu, w_var = mm.update_w_batch(mm.lambda_a_, mm.lambda_b_)    

    assert np.sum(np.isclose(w_mu[:, 0, 0],
        np.array([ 9.9645200305, -0.9946675522]))) == 2, \
        "Coefficients not calculated correctly"
    assert np.sum(np.isclose(w_mu[:, 0, 1],
        np.array([ 5.9582239375, -2.0035498142]))) == 2, \
        "Coefficients not calculated correctly"
    assert np.sum(np.isclose(w_var[:, 0, 0],
        np.array([0.0151515152, 0.0001333488]))) == 2, \
        "Variances not calculated correctly"
    assert np.sum(np.isclose(w_var[:, 0, 1],
        np.array([0.0041493776, 0.0000417711]))) == 2, \
        "Variances not calculated correctly"
    

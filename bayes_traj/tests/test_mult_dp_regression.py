from bayes_traj.mult_dp_regression import MultDPRegression
import numpy as np
import pandas as pd
from bayes_traj.utils import *
import pdb, os, pickle

np.set_printoptions(precision = 10, suppress = True, threshold=1e6,
                    linewidth=300)

def test_update_w_logistic():
    data_file_name = os.path.split(os.path.realpath(__file__))[0] + \
        '/../resources/data/binary_data_1.csv'
    df = pd.read_csv(data_file_name)

    M = 2
    D = 1
    K = 1
    prior_data = {}
    prior_data['w_mu0'] = np.zeros([M, D])    
    prior_data['w_var0'] = 100*np.ones([M, D])
    prior_data['lambda_a0'] = np.ones([D])
    prior_data['lambda_b0'] = np.ones([D])
    prior_data['alpha'] = 1

    mm = MultDPRegression(prior_data['w_mu0'], prior_data['w_var0'],
                          prior_data['lambda_a0'], prior_data['lambda_b0'],
                          1/df.shape[0], prior_data['alpha'], K=K)

    mm.N_ = df.shape[0]
    mm.target_type_[0] = 'binary'
    mm.num_binary_targets_ = 1
    mm.w_var_ = None
    mm.w_covmat_ = np.nan*np.ones([M, M, D, K])
    mm.lambda_a_ = None    
    mm.lambda_b_ = None    
    mm.X_ = df[['intercept', 'pred']].values
    mm.Y_ = np.atleast_2d(df.target.values).T
    mm.gb_ = None
    
    mm.init_traj_params()

    mm.R_ = np.ones([mm.N_, K])
    
    mm.update_w_logistic(25)

    assert np.isclose(mm.w_mu_[0, 0, 0], 2.206, atol=0, rtol=.01), \
        "Intercept not as expected"
    assert np.isclose(mm.w_mu_[1, 0, 0], -2.3492, atol=0, rtol=.01), \
        "Slope not as expected"

    # Check that the function can handle nans
    mm.Y_[0, 0] = np.nan

    mm.init_traj_params()
    mm.update_w_logistic(25)

    # The intercept and slope that were used to create this synthetic data were
    # 2.5 and -2.5, respectively. When running standard logistic regression on
    # this data, the intercept and slope are found to be 2.2060 and -2.3492 
    assert np.isclose(mm.w_mu_[0, 0, 0], 2.206, atol=0, rtol=.01), \
        "Intercept not as expected"
    assert np.isclose(mm.w_mu_[1, 0, 0], -2.3492, atol=0, rtol=.01), \
        "Slope not as expected"

def test_update_w_logistic_2():
    data_file_name = os.path.split(os.path.realpath(__file__))[0] + \
        '/../resources/data/binary_data_2.csv'
    df = pd.read_csv(data_file_name)

    # Intercept, slope for group 1: 2.5, -2.5
    # Intercept, slope for group 1: -4, 4    
    
    M = 2
    D = 1
    K = 2
    prior_data = {}
    prior_data['w_mu0'] = np.zeros([M, D])
    #prior_data['w_mu0'][:, 0] = np.array([-50, 10])
    
    prior_data['w_var0'] = 100*np.ones([M, D])
    prior_data['lambda_a0'] = np.ones([D])
    prior_data['lambda_b0'] = np.ones([D])
    prior_data['alpha'] = 1

    mm = MultDPRegression(prior_data['w_mu0'], prior_data['w_var0'],
                          prior_data['lambda_a0'], prior_data['lambda_b0'],
                          1/df.shape[0],
                          prior_data['alpha'], K=K)

    mm.N_ = df.shape[0]
    mm.target_type_[0] = 'binary'
    mm.num_binary_targets_ = 1
    mm.w_var_ = None
    mm.w_covmat_ = np.nan*np.ones([M, M, D, K])
    mm.lambda_a_ = None
    mm.lambda_b_ = None    
    mm.X_ = df[['intercept', 'pred']].values
    mm.Y_ = np.atleast_2d(df.target.values).T
    mm.gb_ = None
    
    mm.init_traj_params()

    mm.R_ = np.zeros([mm.N_, K])
    mm.R_[0:int(mm.N_/2), 0] = 1
    mm.R_[int(mm.N_/2):-1, 1] = 1

    mm.R_ = np.zeros([mm.N_, K]) + 1e-4 #+ .00000000001
    mm.R_[0:int(mm.N_/2), 0] = 1-1e-4#.99999999999
    mm.R_[int(mm.N_/2)::, 1] = 1-1e-4#.99999999999

    mm.update_w_logistic(25)
    
    # The intercept and slope that were used to create this synthetic data were
    # 2.5 and -2.5 for the first group and -4 and 4 for the second group. When
    # running standard logistic regression on this data, the intercept and slope
    # are found to be 2.3702 and -2.1732 for the first group and -3.8925 and
    # 4.1095 for the second group    
    assert np.isclose(mm.w_mu_[0, 0, 0], 2.3702, atol=0, rtol=.01), \
        "Intercept not as expected"
    assert np.isclose(mm.w_mu_[1, 0, 0], -2.1732, atol=0, rtol=.01), \
        "Slope not as expected"

    assert np.isclose(mm.w_mu_[0, 0, 1], -3.8925, atol=0, rtol=.01), \
        "Intercept not as expected"
    assert np.isclose(mm.w_mu_[1, 0, 1], 4.1095, atol=0, rtol=.01), \
        "Slope not as expected"

def test_update_z_logistic():
    data_file_name = os.path.split(os.path.realpath(__file__))[0] + \
        '/../resources/data/binary_data_3.csv'
    df = pd.read_csv(data_file_name)

    # Intercept, slope for group 1: 0, 50
    # Intercept, slope for group 1: 0, -50
    
    M = 2
    D = 1
    K = 2
    prior_data = {}
    prior_data['w_mu0'] = np.zeros([M, D])
    
    prior_data['w_var0'] = 100*np.ones([M, D])
    prior_data['lambda_a0'] = np.ones([D])
    prior_data['lambda_b0'] = np.ones([D])
    prior_data['alpha'] = 1

    mm = MultDPRegression(prior_data['w_mu0'], prior_data['w_var0'],
                          prior_data['lambda_a0'], prior_data['lambda_b0'],
                          1/df.shape[0], prior_data['alpha'], K=K)

    mm.N_ = df.shape[0]
    mm.target_type_[0] = 'binary'
    mm.num_binary_targets_ = 1
    mm.w_var_ = None
    mm.lambda_a_ = None
    mm.lambda_b_ = None    
    mm.X_ = df[['intercept', 'pred']].values
    mm.Y_ = np.atleast_2d(df.target.values).T
    mm.gb_ = None
    mm.group_first_index_ = np.ones(mm.N_, dtype=bool)
    mm.w_covmat_ = np.ones([M, M, D, K])
    
    mm.init_traj_params()
    mm.v_a_ = np.ones(mm.K_)
    mm.v_b_ = mm.alpha_*np.ones(mm.K_)

    # Set w_mu_ to be correct
    mm.w_mu_[0, 0, 0] = 0
    mm.w_mu_[1, 0, 0] = 50
    mm.w_mu_[0, 0, 1] = 0
    mm.w_mu_[1, 0, 1] = -50
    
    # Set w_covmat_ to be correct
    mm.w_covmat_ = np.zeros([M, M, D, K])
    mm.w_covmat_[:, :, 0, 0] = 1e-50*np.diag([M, M])
    mm.w_covmat_[:, :, 0, 1] = 1e-50*np.diag([M, M])    
    
    # Set R to be correct
    mm.R_ = np.zeros([mm.N_, mm.K_])
    mm.R_[0:int(mm.N_/2), 0] = 1
    mm.R_[int(mm.N_/2)::, 1] = 1

    mm.update_v()

    # Scramble R
    mm.R_[:, 0] = np.random.uniform(0.001, .999, mm.N_)
    mm.R_[:, 1] = 1. - mm.R_[:, 0]
    
    # test update_z
    R_updated = mm.update_z(mm.X_, mm.Y_)

    assert np.isclose(np.mean(R_updated[int(mm.N_/2)::], 0)[1], .999,
                      atol=0, rtol=.01), "R not updated correctly"
    assert np.isclose(np.mean(R_updated[0:int(mm.N_/2)], 0)[0], .999,
                      atol=0, rtol=.01), "R not updated correctly"    
    
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
                          prior_data['lambda_a0'], prior_data['lambda_b0'], 1,
                          prior_data['alpha'], K=K)

    mm.fit(target_names=targets, predictor_names=preds, df=df, groupby='id',
           iters=20, verbose=True)

    df_traj = mm.to_df()

    num_trajs_found = np.sum(np.where(pd.crosstab(df_traj.traj.values,
                                    df.traj.values).values == 250))

    assert num_trajs_found == 2, "Trajectory assignment error"


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
    prec_prior_weight = 1
    alpha = 5
    K = 30
    mm = MultDPRegression(w_mu0, w_var0, lambda_a0, lambda_b0,
                          prec_prior_weight, alpha, K)

    mm.target_type_ = {}
    mm.target_type_[0] = 'gaussian'
    mm.target_type_[1] = 'gaussian'
    mm.gb_ = None
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
    traj_probs = np.zeros(K)
    traj_probs[0] = .25
    traj_probs[1] = .25
    traj_probs[2] = .25
    traj_probs[3] = .25    

    mm.init_R_mat(traj_probs, traj_probs_weight=1)
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
    prec_prior_weight = 1
    alpha = 5
    mm = MultDPRegression(w_mu0, w_var0, lambda_a0, lambda_b0,
                          prec_prior_weight, alpha, K)

    mm.target_type_ = {}
    mm.target_type_[0] = 'gaussian'

    mm.w_mu_ = np.zeros([M, D, K])
    mm.w_mu_[:, 0, 0] = np.array([2, 1])
    mm.w_mu_[:, 0, 1] = np.array([-2, -1])    
    mm.R_ = 0.5*np.ones([3, 2])
    mm.lambda_b_ = np.ones([D, K])
    mm.lambda_a_ = np.ones([D, K])
    mm.gb_ = None
    
    X = np.array([[1, 2], [1, 2], [1, 2]])
    Y = np.array([[3], [0], [-3]])

    mm.group_first_index_ = np.ones(X.shape[0], dtype=bool)
    
    R = mm.predict_proba_(X, Y)
    R_ref = np.array([[1.00000000e+00, 3.77513454e-11],
                      [5.00000000e-01, 5.00000000e-01],
                      [3.77513454e-11, 1.00000000e+00]])
    assert np.sum(np.isclose(R, R_ref)) == 6, "Unexpected R value"

    #constraints = get_longitudinal_constraints_graph(np.array([0, 2, 0]))
    #R = mm.predict_proba_(X, Y, constraints)
    #R_ref = np.array([[0.5, 0.5],
    #                  [0.5, 0.5],
    #                  [0.5, 0.5]])
    #assert np.sum(np.isclose(R, R_ref)) == 6, "Unexpected R value"

def test_init_traj_parmas():
    data_file_name = os.path.split(os.path.realpath(__file__))[0] + \
        '/../resources/data/trajectory_data_1.csv'
    df = pd.read_csv(data_file_name)
    
    prior_file_name = os.path.split(os.path.realpath(__file__))[0] + \
        '/../resources/priors/model_1_posterior.p'
    prior_info = pickle.load(open(prior_file_name, 'rb'))

    preds = get_pred_names_from_prior_info(prior_info)
    targets = get_target_names_from_prior_info(prior_info)        

    D = len(targets)
    M = len(preds)
    K = 20

    prec_prior_weight = 1

    prior_data = {}
    prior_data['w_mu0'] = np.zeros([M, D])
    prior_data['w_var0'] = np.ones([M, D])
    prior_data['lambda_a0'] = np.ones([D])
    prior_data['lambda_b0'] = np.ones([D])

    prior_data['w_mu'] = np.zeros([M, D, K])
    prior_data['w_var'] = np.ones([M, D, K])
    prior_data['lambda_a'] = np.ones([D, K])
    prior_data['lambda_b'] = np.ones([D, K])
    
    prior_data['alpha'] = prior_info['alpha']
    for (d, target) in enumerate(targets):
        prior_data['lambda_a0'][d] = prior_info['lambda_a0'][target]
        prior_data['lambda_b0'][d] = prior_info['lambda_b0'][target]            
        prior_data['lambda_a'][d, :] = prior_info['lambda_a'][target]
        prior_data['lambda_b'][d, :] = prior_info['lambda_b'][target]        
        for (m, pred) in enumerate(preds):
            prior_data['w_mu0'][m, d] = prior_info['w_mu0'][target][pred]
            prior_data['w_var0'][m, d] = prior_info['w_var0'][target][pred]
            prior_data['w_mu'][m, d, :] = prior_info['w_mu'][pred][target]
            prior_data['w_var'][m, d, :] = prior_info['w_var'][pred][target]

    traj_probs = prior_info['traj_probs']
    traj_probs_weight = 1.
    v_a = prior_info['v_a']
    v_b = prior_info['v_b']
    w_mu = None
    w_var = None
    lambda_a = None
    lambda_b = None
    
    mm = MultDPRegression(prior_data['w_mu0'], prior_data['w_var0'],
                          prior_data['lambda_a0'], prior_data['lambda_b0'],
                          prec_prior_weight, prior_data['alpha'], K)
    mm.fit(target_names=targets, predictor_names=preds, df=df, groupby='id',
           iters=0, verbose=True, traj_probs=traj_probs,
           traj_probs_weight=traj_probs_weight,
           v_a=prior_info['v_a'], v_b=prior_info['v_b'],
           w_mu=prior_data['w_mu'], w_var=prior_data['w_var'],
           lambda_a=prior_data['lambda_a'], lambda_b=prior_data['lambda_b'])

    assert np.sum(mm.w_mu_[:, :, traj_probs > 0] == \
                  prior_data['w_mu'][:, :, traj_probs > 0]) == 4, \
                  "Trajs params not initialized properly"
    
    assert np.sum(prior_data['w_var'] == mm.w_var_) == 40, \
        "Trajs params not initialized properly"

    assert np.sum(prior_data['lambda_a'] == mm.lambda_a_) == 20, \
        "Trajs params not initialized properly"

    assert np.sum(prior_data['lambda_b'] == mm.lambda_b_) == 20, \
        "Trajs params not initialized properly"    

    assert np.sum(prior_info['v_a'] == mm.v_a_) == 20, \
        "Trajs params not initialized properly"

    assert np.sum(prior_info['v_b'] == mm.v_b_) == 20, \
        "Trajs params not initialized properly"        
    
def test_init_R_mat():
    data_file_name = os.path.split(os.path.realpath(__file__))[0] + \
        '/../resources/data/trajectory_data_1.csv'
    df = pd.read_csv(data_file_name)
    
    prior_file_name = os.path.split(os.path.realpath(__file__))[0] + \
        '/../resources/priors/model_1_posterior.p'
    prior_info = pickle.load(open(prior_file_name, 'rb'))

    preds = get_pred_names_from_prior_info(prior_info)
    targets = get_target_names_from_prior_info(prior_info)        

    D = len(targets)
    M = len(preds)
    K = 20

    prec_prior_weight = 1

    prior_data = {}
    prior_data['w_mu0'] = np.zeros([M, D])
    prior_data['w_var0'] = np.ones([M, D])
    prior_data['lambda_a0'] = np.ones([D])
    prior_data['lambda_b0'] = np.ones([D])

    prior_data['w_mu'] = np.zeros([M, D, K])
    prior_data['w_var'] = np.ones([M, D, K])
    prior_data['lambda_a'] = np.ones([D, K])
    prior_data['lambda_b'] = np.ones([D, K])
    
    prior_data['alpha'] = prior_info['alpha']
    for (d, target) in enumerate(targets):
        prior_data['lambda_a0'][d] = prior_info['lambda_a0'][target]
        prior_data['lambda_b0'][d] = prior_info['lambda_b0'][target]            
        prior_data['lambda_a'][d, :] = prior_info['lambda_a'][target]
        prior_data['lambda_b'][d, :] = prior_info['lambda_b'][target]        
        for (m, pred) in enumerate(preds):
            prior_data['w_mu0'][m, d] = prior_info['w_mu0'][target][pred]
            prior_data['w_var0'][m, d] = prior_info['w_var0'][target][pred]
            prior_data['w_mu'][m, d, :] = prior_info['w_mu'][pred][target]
            prior_data['w_var'][m, d, :] = prior_info['w_var'][pred][target]

    traj_probs = prior_info['traj_probs']
    traj_probs_weight = 1.
    v_a = prior_info['v_a']
    v_b = prior_info['v_b']
    w_mu = None
    w_var = None
    lambda_a = None
    lambda_b = None
    
    mm = MultDPRegression(prior_data['w_mu0'], prior_data['w_var0'],
                          prior_data['lambda_a0'], prior_data['lambda_b0'],
                          prec_prior_weight, prior_data['alpha'], K)
    mm.fit(target_names=targets, predictor_names=preds, df=df, groupby='id',
           iters=0, verbose=True, traj_probs=traj_probs,
           traj_probs_weight=traj_probs_weight,
           v_a=prior_info['v_a'], v_b=prior_info['v_b'],
           w_mu=prior_data['w_mu'], w_var=prior_data['w_var'],
           lambda_a=prior_data['lambda_a'], lambda_b=prior_data['lambda_b'])

    assert np.sum((traj_probs > 0) | (np.sum(mm.R_, 0) > 0)) == 2, \
        "R_mat not initialized properly"
    
    mm.fit(target_names=targets, predictor_names=preds, df=df, groupby='id',
           iters=0, verbose=True, traj_probs=traj_probs,
           traj_probs_weight=0,
           v_a=prior_info['v_a'], v_b=prior_info['v_b'],
           w_mu=prior_data['w_mu'], w_var=prior_data['w_var'],
           lambda_a=prior_data['lambda_a'], lambda_b=prior_data['lambda_b'])

    # It's possible, thoush highly unlikely that the following sum is <=2. With
    # traj_probs_weight set to 0, a number of initialized trajectories should
    # have non-zero weight
    assert np.sum((traj_probs > 0) | (np.sum(mm.R_, 0) > 0)) > 2, \
        "R_mat may not be initialized properly"

    

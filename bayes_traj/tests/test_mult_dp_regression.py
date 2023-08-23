import torch
from bayes_traj.mult_dp_regression import MultDPRegression
import numpy as np
import pandas as pd
from bayes_traj.utils import *
import pdb, os, pickle

np.set_printoptions(precision = 10, suppress = True, threshold=1e6,
                    linewidth=300)

def get_gt_df():
    df = pd.DataFrame(\
        {'sid': ['a', 'a', 'a', 'a', 'a', 'b', 'b', 'b', 'b'],
         'intercept': np.ones(9),
         'x': np.array([0, 2, 4, 6, 10, 0, 3, 7, 10])})
        
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

    return df
    
def get_gt_model():
    """
    """
    df = get_gt_df()
    
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

    mm.num_binary_targets_ = 0
    
    mm.target_names_ = ['y1', 'y2']
    mm.predictor_names_ = ['intercept', 'x']
    
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


def test_update_w_logistic():
    data_file_name = os.path.split(os.path.realpath(__file__))[0] + \
        '/../resources/data/binary_data_1.csv'
    df = pd.read_csv(data_file_name)

    M = 2
    D = 1
    K = 1
    prior_data = {}
    prior_data['w_mu0'] = torch.zeros([M, D]).double()
    prior_data['w_var0'] = 100.*torch.ones([M, D]).double()
    prior_data['lambda_a0'] = torch.ones([D]).double()
    prior_data['lambda_b0'] = torch.ones([D]).double()
    prior_data['alpha'] = 1

    mm = MultDPRegression(prior_data['w_mu0'], prior_data['w_var0'],
                          prior_data['lambda_a0'], prior_data['lambda_b0'],
                          1/df.shape[0], prior_data['alpha'], K=K)

    mm.N_ = df.shape[0]
    mm.target_type_[0] = 'binary'
    mm.num_binary_targets_ = 1
    mm.w_var_ = None
    mm.w_covmat_ = torch.from_numpy(np.nan*np.ones([M, M, D, K])).double()
    mm.lambda_a_ = None    
    mm.lambda_b_ = None    
    mm.X_ = torch.from_numpy(df[['intercept', 'pred']].values).double()
    mm.Y_ = torch.from_numpy(np.atleast_2d(df.target.values).T).double()
    mm.gb_ = None

    mm.init_traj_params()

    mm.R_ = torch.ones([mm.N_, K]).double()
    
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
    mm.w_covmat_ = torch.from_numpy(np.nan*np.ones([M, M, D, K])).double()
    mm.lambda_a_ = None
    mm.lambda_b_ = None    
    mm.X_ = torch.from_numpy(df[['intercept', 'pred']].values).double()
    mm.Y_ = torch.from_numpy(np.atleast_2d(df.target.values).T).double()
    mm.gb_ = None
    
    mm.init_traj_params()

    mm.R_ = torch.zeros([mm.N_, K]).double()
    mm.R_[0:int(mm.N_/2), 0] = 1
    mm.R_[int(mm.N_/2):-1, 1] = 1

    mm.R_ = torch.zeros([mm.N_, K]).double() + 1e-4 #+ .00000000001
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
    df['id'] = np.arange(0, df.shape[0]) 
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
    mm.X_ = torch.from_numpy(df[['intercept', 'pred']].values).double()
    mm.Y_ = torch.from_numpy(np.atleast_2d(df.target.values).T).double()
    mm.gb_ = None
    mm.group_first_index_ = np.ones(mm.N_, dtype=bool)
    mm.w_covmat_ = torch.ones([M, M, D, K]).double()
    
    mm.init_traj_params()
    mm.v_a_ = torch.ones(mm.K_).double()
    mm.v_b_ = mm.alpha_*torch.ones(mm.K_).double()

    mm.df_helper_ = pd.DataFrame(df[['id']])
    for k in range(mm.K_):
        mm.df_helper_['like_accum_' + str(k)] = np.nan*np.zeros(mm.N_)
    mm.gb_ = mm.df_helper_.groupby('id')
        
    # Set w_mu_ to be correct
    mm.w_mu_[0, 0, 0] = 0
    mm.w_mu_[1, 0, 0] = 50
    mm.w_mu_[0, 0, 1] = 0
    mm.w_mu_[1, 0, 1] = -50
    
    # Set w_covmat_ to be correct
    mm.w_covmat_ = torch.zeros([M, M, D, K]).double()
    mm.w_covmat_[:, :, 0, 0] = 1e-50*torch.diag(torch.tensor([M, M])).double()
    mm.w_covmat_[:, :, 0, 1] = 1e-50*torch.diag(torch.tensor([M, M])).double()

    # Set R to be correct
    mm.R_ = torch.zeros([mm.N_, mm.K_]).double()
    mm.R_[0:int(mm.N_/2), 0] = 1
    mm.R_[int(mm.N_/2)::, 1] = 1

    mm.update_v()
    
    # Scramble R
    mm.R_[:, 0] = torch.distributions.Uniform(0.001, .999).sample((mm.N_, ))
    mm.R_[:, 1] = 1. - mm.R_[:, 0]
    
    # test update_z
    R_updated = mm.update_z(mm.X_, mm.Y_).numpy()

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

    
def test_to_df():
    targets = ['y1', 'y2']
    preds = ['intercept', 'x']

    D = len(targets)
    M = len(preds)
    K = 2

    prior_data = {}
    prior_data['w_mu0'] = np.zeros([M, D])
    prior_data['w_var0'] = np.ones([M, D])
    prior_data['lambda_a0'] = np.ones([D])
    prior_data['lambda_b0'] = np.ones([D])
    
    mm = MultDPRegression(prior_data['w_mu0'], prior_data['w_var0'],
                          prior_data['lambda_a0'], prior_data['lambda_b0'], 1,
                          1, K=K)
    mm.predictor_names_ = preds
    mm.target_names_ = targets    
    mm.X_ = torch.tensor([[1, 2], [1, 3], [1, 5]]).double()
    mm.Y_ = torch.tensor([[3, 7], [5.5, 3], [2, 8.1]]).double()
    mm.R_ = torch.tensor([[.8, .2], [0, 1], [.3, .7]]).double()

    df = mm.to_df()
    assert np.array_equal(df[preds].values, mm.X_), \
        "Predictor values not equal"
    assert np.array_equal(df[targets].values, mm.Y_), \
        "Target values not equal"
    assert 'traj' in df.columns and 'traj_0' in df.columns and \
        'traj_1' in df.columns, "Dataframe missing traj columns"
    assert df.traj.values[0] != df.traj.values[1] and \
        df.traj.values[1] == df.traj.values[2], \
        "Traj assignments incorrect"

    
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

    X_mat = torch.ones([num_per_traj*3, 2]).double()
    X_mat[:, 1] = torch.from_numpy(np.vstack([x, x, x]).reshape(-1)).double()
    Y_mat = torch.from_numpy(np.atleast_2d(\
            np.vstack([y1, y2, y3]).reshape(-1)).T).double()
    
    M = X_mat.shape[1]
    D = Y_mat.shape[1]
    N = Y_mat.shape[0]
    
    w_mu0 = torch.zeros([M, D]).double()
    w_var0 = torch.ones([M, D]).double()
    lambda_a0 = torch.ones(D).double()
    lambda_b0 = torch.ones(D).double()
    prec_prior_weight = 1
    alpha = 5
    K = 30

    mm = MultDPRegression(w_mu0, w_var0, lambda_a0, lambda_b0,
                          prec_prior_weight, alpha, K)

    mm.target_type_ = {}
    mm.target_type_[0] = 'gaussian'
    mm.target_type_[1] = 'gaussian'
    mm.gb_ = None
    lambda_a = torch.ones([D, K]).double()
    lambda_b = torch.ones([D, K]).double()
    w_mu = torch.zeros([M, D, K]).double()
    w_mu[:, 0, 0] = torch.tensor([b1, m1]).double()
    w_mu[:, 0, 1] = torch.tensor([b2, m2]).double()
    w_mu[:, 0, 2] = torch.tensor([b3, m3]).double()
    w_var = torch.ones([M, D, K]).double()
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
    traj_probs[0] = .2
    traj_probs[1] = .3
    traj_probs[2] = .4
    traj_probs[3] = .1    

    mm.init_R_mat(traj_probs, traj_probs_weight=1)
    assert np.sum(np.isclose(np.ones(K), torch.sum(mm.R_, 1).numpy())) == K, \
        "Unexpected R_ sum"

    traj_assignments = np.array([np.where(mm.R_[i, :].numpy() == \
            np.max(mm.R_[i, :].numpy()))[0][0] for i in range(N)])

    assert torch.allclose(torch.from_numpy(traj_probs).double(), \
                          torch.sum(mm.R_, 0)/N), \
                          "Unexpected R_ sum"


def test_init_traj_params():
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

    assert np.sum((mm.w_mu_).numpy()[:, :, traj_probs > 0] == \
                  prior_data['w_mu'][:, :, traj_probs > 0]) == 4, \
                  "Trajs params not initialized properly"

    assert np.sum(prior_data['w_var'] == (mm.w_var_).numpy()) == 40, \
        "Trajs params not initialized properly"

    assert np.sum(prior_data['lambda_a'] == (mm.lambda_a_).numpy()) == 20, \
        "Trajs params not initialized properly"

    assert np.sum(prior_data['lambda_b'] == (mm.lambda_b_).numpy()) == 20, \
        "Trajs params not initialized properly"    

    assert np.sum(prior_info['v_a'] == (mm.v_a_).numpy()) == 20, \
        "Trajs params not initialized properly"

    assert np.sum(prior_info['v_b'] == (mm.v_b_).numpy()) == 20, \
        "Trajs params not initialized properly"        
    
def test_init_R_mat_2():
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

    assert np.sum((traj_probs > 0) | (np.sum(mm.R_.numpy(), 0) > 0)) == 2, \
        "R_mat not initialized properly"

    mm2 = MultDPRegression(prior_data['w_mu0'], prior_data['w_var0'],
                          prior_data['lambda_a0'], prior_data['lambda_b0'],
                          prec_prior_weight, prior_data['alpha'], K)    
    mm2.fit(target_names=targets, predictor_names=preds, df=df, groupby='id',
           iters=0, verbose=True, traj_probs=traj_probs,
           traj_probs_weight=0, R=None,
           v_a=prior_info['v_a'], v_b=prior_info['v_b'],
           w_mu=prior_data['w_mu'], w_var=prior_data['w_var'],
           lambda_a=prior_data['lambda_a'], lambda_b=prior_data['lambda_b'])

    # It's possible, though highly unlikely that the following sum is <=2. With
    # traj_probs_weight set to 0, a number of initialized trajectories should
    # have non-zero weight
    assert np.sum((traj_probs > 0) | (np.sum(mm2.R_.numpy(), 0) > 0)) > 2, \
        "R_mat may not be initialized properly"

    
def test_get_traj_probs():
    mm = get_gt_model()

    assert np.sum(mm.get_traj_probs() == np.array([0.5, 0.5, 0, 0, 0])) == 5, \
        "Traj probs not correct"

    
def test_augment_df_with_traj_info():
    mm = get_gt_model()
    df = get_gt_df()
    df_aug = mm.augment_df_with_traj_info(df, 'sid')

    assert np.sum(df_aug['traj'].values == \
        np.array([0, 0, 0, 0, 0, 1, 1, 1, 1])) == 9, \
        "Incorrect traj assignment"

    for cc in df.columns:
        assert cc in df_aug.columns, "Dataframe incorrectly augmented"
    
    df_aug2 = mm.augment_df_with_traj_info(df)


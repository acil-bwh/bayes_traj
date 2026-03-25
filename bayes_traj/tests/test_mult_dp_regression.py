import torch
from bayes_traj.mult_dp_regression import MultDPRegression
import numpy as np
import pandas as pd
from bayes_traj.utils import *
import pdb, os, pickle
import matplotlib.pyplot as plt # TODO DEB

np.set_printoptions(precision = 10, suppress = True, threshold=1e6,
                    linewidth=300)

def get_basic_model(G, M, num_gaussian, num_binary, K, num_long_data_pts):
    """
    """
    np.random.seed(42)
    N = G*num_long_data_pts
    
    target_names = []
    target_type = [] 
    for dd in range(num_gaussian):
        target_names.append(f'gaussian_{dd}')
        target_type.append('gaussian')

    for dd in range(num_binary):
        target_names.append(f'binary_{dd}')
        target_type.append('binary')        
        
    predictor_names = []
    for pp in range(M):
        predictor_names.append(f'predictor_{pp}')

    D = num_gaussian + num_binary
    
    w_var0 = np.ones([M, D])
    w_mu0 = np.zeros([M, D])
    lambda_a0 = np.ones(D)
    lambda_b0 = np.ones(D)
    alpha = 1.
    mm = MultDPRegression(w_mu0, w_var0,
                          lambda_a0, lambda_b0,
                          1, 1, K=K)
    mm.target_names_ = target_names
    mm.target_type_ = target_type
    mm.predictor_names_ = predictor_names
    mm.M_ = M
    mm.N_ = N
    mm.G_ = G
    mm.K_ = K
    mm.num_binary_targets_ = num_binary

    mm.R_ = torch.zeros(N, K)
    inc = 0
    sids = []
    for gg in range(G):
        vec = np.exp(np.random.randn(K))
        vec_norm = vec/np.sum(vec)
        sid = f'sid_{gg}'
        for nn in range(num_long_data_pts):
            mm.R_[inc, :] = torch.from_numpy(vec_norm)
            sids.append(sid)
            inc += 1
    
    mm.X_ = torch.from_numpy(np.random.rand(N, M))
    mm.Y_ = torch.from_numpy(np.random.rand(N, D))
    for dd in range(num_gaussian, num_binary):
        tmp_ids = mm.Y_[:, dd] > 0.5
        mm.Y_[tmp_ids, dd] = 1
        mm.Y_[~tmp_ids, dd] = 0    

    mm.w_mu_ = torch.from_numpy(np.random.randn(M, D, K))
    mm.w_var_ = torch.from_numpy(np.exp(np.random.randn(M, D, K)))

    mm.lambda_a_ = torch.from_numpy(np.exp(np.random.randn(D, K)))
    mm.lambda_b_ = torch.from_numpy(np.exp(np.random.randn(D, K)))    

    mm.u_mu_ = torch.from_numpy(np.random.randn(G, D, K, M))

    df = pd.DataFrame()
    df['sid'] = sids
    for xx in range(M):
        df[predictor_names[xx]] = mm.X_[:, xx]
    for yy in range(D):
        df[target_names[yy]] = mm.Y_[:, yy]

    mm.gb_ = df.groupby('sid') 
    mm.N_to_G_index_map_ = np.arange(N)
    for ii, (kk, vv) in enumerate(mm.gb_.groups.items()):
        mm.N_to_G_index_map_[vv] = ii
    
    return mm
    
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

    target_names = ['y1', 'y2']
    predictor_names = ['intercept', 'x']

    Sig0 = {}
    for tt in target_names:
        Sig0[tt] = torch.eye(M)

    ranef_indices = np.zeros(M, dtype=bool)
    
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
                          prec_prior_weight, alpha, K=K, Sig0=Sig0,
                          ranef_indices=ranef_indices)
    mm.R_ = torch.zeros([N, K]).double()
    mm.R_[0:5, 0] = 1.
    mm.R_[5::, 1] = 1.
        
    mm.target_type_ = {}
    mm.target_type_[0] = 'gaussian'
    mm.target_type_[1] = 'gaussian'

    mm.num_binary_targets_ = 0
    
    mm.target_names_ = target_names
    mm.predictor_names_ = predictor_names
    
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
    G = mm.gb_.ngroups

    mm.N_to_G_index_map_ = np.arange(N)
    for ii, (kk, vv) in enumerate(mm.gb_.groups.items()):
        mm.N_to_G_index_map_[vv] = ii
    
    mm.X_ = torch.from_numpy(df[['intercept', 'x']].values).double()
    mm.Y_ = torch.from_numpy(df[['y1', 'y2']].values).double()
    mm.N_ = N 

    mm._set_group_first_index(df, mm.gb_)

    mm.u_mu_ = torch.zeros((G, D, K, M))
    mm.u_Sig_ = torch.zeros((G, D, K, M, M), dtype=torch.float64)
    
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
    
#def test_MultDPRegression():
#    # Read data from resources dir
#    data_file_name = os.path.split(os.path.realpath(__file__))[0] + \
#        '/../resources/data/trajectory_data_1.csv'
#    df = pd.read_csv(data_file_name)
#    df['cohort'] = 0
#    
#    prior_file_name = os.path.split(os.path.realpath(__file__))[0] + \
#        '/../resources/priors/trajectory_prior_1.p'
#    
#    # Read prior from resources dir
#    with open(prior_file_name, 'rb') as f:
#        prior_file_info = pickle.load(f)
#        
#    base_preds = get_pred_names_from_prior_info(prior_file_info)
#    preds = base_preds + ['cohort']
#    targets = get_target_names_from_prior_info(prior_file_info)        
#
#    D = len(targets)
#    M = len(preds)
#    K = 20
#
#    prior_data = {}
#    prior_data['w_mu0'] = np.zeros([M, D])
#    prior_data['w_var0'] = np.ones([M, D])
#    prior_data['lambda_a0'] = np.ones([D])
#    prior_data['lambda_b0'] = np.ones([D])
#    prior_data['alpha'] = prior_file_info['alpha']
#    for (d, target) in enumerate(targets):
#        prior_data['lambda_a0'][d] = prior_file_info['lambda_a0'][target]
#        prior_data['lambda_b0'][d] = prior_file_info['lambda_b0'][target]            
#            
#        for (m, pred) in enumerate(base_preds):
#            prior_data['w_mu0'][m, d] = prior_file_info['w_mu0'][target][pred]
#            prior_data['w_var0'][m, d] = prior_file_info['w_var0'][target][pred]
#
#        cohort_idx = preds.index('cohort')
#        prior_data['w_mu0'][cohort_idx, d] = 0.0
#        prior_data['w_var0'][cohort_idx, d] = 1.0            
#   
#    mm = MultDPRegression(prior_data['w_mu0'], prior_data['w_var0'],
#                          prior_data['lambda_a0'], prior_data['lambda_b0'], 1,
#                          prior_data['alpha'], K=K)
#
#    mm.fit(target_names=targets, predictor_names=preds, df=df, groupby='id',
#           iters=20, verbose=True, shared_predictor_names=None)
#    
#    assert mm.shared_indices_.shape[0] == 0
#    assert np.alltrue(mm.traj_indices_ == np.array([0, 1, 2]))
#
#    mm.fit(target_names=targets, predictor_names=preds,
#           df=df, groupby='id', iters=20, verbose=True,
#           shared_predictor_names=['cohort'])
    
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


def test_log_likelihood():
    G = 1
    M = 2
    num_gaussian = 1
    num_binary = 0    
    K = 2
    num_long_data_pts = 2
    mm = get_basic_model(G, M, num_gaussian, num_binary, K, num_long_data_pts)

    assert torch.isclose(mm.log_likelihood(),
        torch.tensor(-3.1031359432, dtype=torch.float64)), \
        "Incorrect log-likelihood value"

    Y = mm.Y_.clone().detach()
    mm.Y_[0, 0] = torch.nan
    ll1 = mm.log_likelihood()
    mm.Y_[0, 0] = Y[0, 0]
    mm.Y_[1, 0] = torch.nan
    ll2 = mm.log_likelihood()
    assert torch.isclose(ll1 + ll2,
        torch.tensor(-3.1031359432, dtype=torch.float64)), \
        "Incorrect log-likelihood value"

    num_gaussian = 0
    num_binary = 1  
    mm = get_basic_model(G, M, num_gaussian, num_binary, K, num_long_data_pts)

    assert torch.isclose(mm.log_likelihood(),
        torch.tensor(-1.8316561418, dtype=torch.float64)), \
        "Incorrect log-likelihood value"
    
def test_MultDPRegression():
    # Read data from resources dir
    data_file_name = os.path.split(os.path.realpath(__file__))[0] + \
        '/../resources/data/trajectory_data_1.csv'
    df = pd.read_csv(data_file_name)
    df['cohort'] = 0

    prior_file_name = os.path.split(os.path.realpath(__file__))[0] + \
        '/../resources/priors/trajectory_prior_1.p'

    # Read prior from resources dir
    with open(prior_file_name, 'rb') as f:
        prior_file_info = pickle.load(f)

    base_preds = get_pred_names_from_prior_info(prior_file_info)
    preds = base_preds + ['cohort']
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

    cohort_idx = preds.index('cohort')

    for (d, target) in enumerate(targets):
        prior_data['lambda_a0'][d] = prior_file_info['lambda_a0'][target]
        prior_data['lambda_b0'][d] = prior_file_info['lambda_b0'][target]

        for (m, pred) in enumerate(base_preds):
            prior_data['w_mu0'][m, d] = prior_file_info['w_mu0'][target][pred]
            prior_data['w_var0'][m, d] = prior_file_info['w_var0'][target][pred]

        prior_data['w_mu0'][cohort_idx, d] = 0.0
        prior_data['w_var0'][cohort_idx, d] = 1.0

    mm = MultDPRegression(prior_data['w_mu0'], prior_data['w_var0'],
                          prior_data['lambda_a0'], prior_data['lambda_b0'], 1,
                          prior_data['alpha'], K=K)

    # ------------------------------------------------------------------
    # No shared predictors: backward-compatible bookkeeping
    # ------------------------------------------------------------------
    mm.fit(target_names=targets, predictor_names=preds, df=df, groupby='id',
           iters=5, verbose=True, shared_predictor_names=None)

    assert mm.shared_indices_.shape[0] == 0
    assert np.array_equal(mm.traj_indices_, np.array([0, 1, 2]))
    assert mm.num_shared_preds_ == 0
    assert mm.num_traj_preds_ == 3
    assert mm.w_mu_shared_.shape == (0, D)
    assert mm.w_var_shared_.shape == (0, D)

    ll_no_shared = mm.log_likelihood()
    bic_no_shared = mm.bic()

    if isinstance(bic_no_shared, tuple):
        assert torch.isfinite(bic_no_shared[0])
        assert np.isfinite(bic_no_shared[1])
    else:
        assert torch.isfinite(bic_no_shared)

    assert torch.isfinite(ll_no_shared)

    # ------------------------------------------------------------------
    # One shared predictor
    # ------------------------------------------------------------------
    mm.fit(target_names=targets, predictor_names=preds,
           df=df, groupby='id', iters=5, verbose=True,
           shared_predictor_names=['cohort'])

    assert np.array_equal(mm.shared_indices_, np.array([cohort_idx]))
    assert np.array_equal(mm.traj_indices_, np.array([0, 1]))
    assert mm.num_shared_preds_ == 1
    assert mm.num_traj_preds_ == 2
    assert mm.w_mu_shared_.shape == (1, D)
    assert mm.w_var_shared_.shape == (1, D)

    # Shared Gaussian coefficients should be present and finite
    assert torch.all(torch.isfinite(mm.w_mu_shared_))
    assert torch.all(torch.isfinite(mm.w_var_shared_))

    # Shared predictors should be inactive in trajectory-specific Gaussian block
    gaussian_ids = [d for d in range(D) if mm.target_type_[d] == 'gaussian']
    for d in gaussian_ids:
        assert torch.allclose(
            mm.w_mu_[mm.shared_indices_, d, :],
            torch.zeros_like(mm.w_mu_[mm.shared_indices_, d, :]),
            atol=1e-8
        ), "Shared predictor should be inactive in trajectory-specific Gaussian block"

    ll_shared = mm.log_likelihood()
    bic_shared = mm.bic()

    assert torch.isfinite(ll_shared)
    if isinstance(bic_shared, tuple):
        assert torch.isfinite(bic_shared[0])
        assert np.isfinite(bic_shared[1])
    else:
        assert torch.isfinite(bic_shared)

def test_shared_predictor_partition_and_init():
    targets = ['y']
    preds = ['intercept', 'x', 'cohort']

    D = len(targets)
    M = len(preds)
    K = 3

    prior_data = {}
    prior_data['w_mu0'] = np.zeros([M, D])
    prior_data['w_var0'] = np.ones([M, D])
    prior_data['lambda_a0'] = np.ones([D])
    prior_data['lambda_b0'] = np.ones([D])

    mm = MultDPRegression(prior_data['w_mu0'], prior_data['w_var0'],
                          prior_data['lambda_a0'], prior_data['lambda_b0'],
                          1, 1, K=K)

    df = pd.DataFrame({
        'sid': ['a', 'a', 'b', 'b'],
        'intercept': [1, 1, 1, 1],
        'x': [0.0, 1.0, 0.0, 1.0],
        'cohort': [0, 0, 1, 1],
        'y': [0.0, 1.0, 0.0, 1.0]
    })

    mm.fit(target_names=targets, predictor_names=preds, df=df, groupby='sid',
           iters=0, verbose=False, shared_predictor_names=['cohort'])

    assert mm.shared_predictor_names_ == ['cohort']
    assert np.array_equal(mm.shared_indices_, np.array([2]))
    assert np.array_equal(mm.traj_indices_, np.array([0, 1]))
    assert mm.w_mu_shared_.shape == (1, 1)
    assert mm.w_var_shared_.shape == (1, 1)
    assert np.isclose(mm.w_mu0_shared_[0, 0].item(), 0.0)
    assert np.isclose(mm.w_var0_shared_[0, 0].item(), 1.0)        

def test_log_likelihood_with_shared_gaussian_effect():
    G = 1
    M = 3
    num_gaussian = 1
    num_binary = 0
    K = 2
    num_long_data_pts = 3

    mm = get_basic_model(G, M, num_gaussian, num_binary, K, num_long_data_pts)

    mm.target_type_[0] = 'gaussian'
    mm.num_binary_targets_ = 0

    mm.predictor_names_ = ['intercept', 'x', 'cohort']
    mm.shared_predictor_names_ = ['cohort']
    mm.shared_indices_ = np.array([2])
    mm.traj_indices_ = np.array([0, 1])
    mm.num_shared_preds_ = 1
    mm.num_traj_preds_ = 2

    mm.w_mu_shared_ = torch.zeros((1, 1), dtype=torch.float64)
    mm.w_var_shared_ = torch.ones((1, 1), dtype=torch.float64)
    mm.w_mu0_shared_ = torch.zeros((1, 1), dtype=torch.float64)
    mm.w_var0_shared_ = torch.ones((1, 1), dtype=torch.float64)

    # Make trajectory-specific coefficient at shared index inactive
    mm.w_mu_[2, 0, :] = 0.0
    mm.w_var_[2, 0, :] = 1.0

    ll0 = mm.log_likelihood()

    # Turn on a shared cohort effect
    mm.w_mu_shared_[0, 0] = 2.0
    ll1 = mm.log_likelihood()

    assert torch.isfinite(ll0)
    assert torch.isfinite(ll1)
    assert not torch.isclose(ll0, ll1), \
        "log_likelihood should change when shared Gaussian coefficient changes"
    
def test_bic_with_shared_predictors_runs():
    G = 2
    M = 3
    num_gaussian = 1
    num_binary = 0
    K = 3
    num_long_data_pts = 2

    mm = get_basic_model(G, M, num_gaussian, num_binary, K, num_long_data_pts)

    mm.predictor_names_ = ['intercept', 'x', 'cohort']
    mm.shared_predictor_names_ = ['cohort']
    mm.shared_indices_ = np.array([2])
    mm.traj_indices_ = np.array([0, 1])
    mm.num_shared_preds_ = 1
    mm.num_traj_preds_ = 2

    mm.w_mu_shared_ = torch.zeros((1, 1), dtype=torch.float64)
    mm.w_var_shared_ = torch.ones((1, 1), dtype=torch.float64)
    mm.w_mu0_shared_ = torch.zeros((1, 1), dtype=torch.float64)
    mm.w_var0_shared_ = torch.ones((1, 1), dtype=torch.float64)

    mm.v_a_ = torch.ones(K, dtype=torch.float64)
    mm.v_b_ = torch.ones(K, dtype=torch.float64)    
    
    mm.gb_ = pd.DataFrame({'sid': [f'sid_{i//num_long_data_pts}'
                                   for i in range(mm.N_)]}).groupby('sid')
    bic_val = mm.bic()

    assert isinstance(bic_val, tuple)
    assert torch.isfinite(bic_val[0])
    assert np.isfinite(bic_val[1])

def test_compute_waic2_with_shared_predictors_runs():
    G = 2
    M = 3
    num_gaussian = 1
    num_binary = 0
    K = 3
    num_long_data_pts = 2

    mm = get_basic_model(G, M, num_gaussian, num_binary, K, num_long_data_pts)

    mm.predictor_names_ = ['intercept', 'x', 'cohort']
    mm.shared_predictor_names_ = ['cohort']
    mm.shared_indices_ = np.array([2])
    mm.traj_indices_ = np.array([0, 1])
    mm.num_shared_preds_ = 1
    mm.num_traj_preds_ = 2

    mm.w_mu_shared_ = torch.zeros((1, 1), dtype=torch.float64)
    mm.w_var_shared_ = torch.ones((1, 1), dtype=torch.float64)
    mm.w_mu0_shared_ = torch.zeros((1, 1), dtype=torch.float64)
    mm.w_var0_shared_ = torch.ones((1, 1), dtype=torch.float64)

    waic_val = mm.compute_waic2(S=20, seed=123)

    assert np.isfinite(waic_val)    

def test_compute_waic2_changes_with_shared_gaussian_effect():
    G = 2
    M = 3
    num_gaussian = 1
    num_binary = 0
    K = 3
    num_long_data_pts = 2

    mm = get_basic_model(G, M, num_gaussian, num_binary, K, num_long_data_pts)

    mm.predictor_names_ = ['intercept', 'x', 'cohort']
    mm.shared_predictor_names_ = ['cohort']
    mm.shared_indices_ = np.array([2])
    mm.traj_indices_ = np.array([0, 1])
    mm.num_shared_preds_ = 1
    mm.num_traj_preds_ = 2

    mm.w_mu_shared_ = torch.zeros((1, 1), dtype=torch.float64)
    mm.w_var_shared_ = torch.ones((1, 1), dtype=torch.float64)
    mm.w_mu0_shared_ = torch.zeros((1, 1), dtype=torch.float64)
    mm.w_var0_shared_ = torch.ones((1, 1), dtype=torch.float64)

    waic0 = mm.compute_waic2(S=50, seed=123)

    mm.w_mu_shared_[0, 0] = 2.0

    waic1 = mm.compute_waic2(S=50, seed=123)

    assert np.isfinite(waic0)
    assert np.isfinite(waic1)
    assert not np.isclose(waic0, waic1), \
        "WAIC should change when shared Gaussian coefficient changes"    

def test_compute_waic2_shared_effect_not_double_counted():
    G = 2
    M = 3
    num_gaussian = 1
    num_binary = 0
    K = 3
    num_long_data_pts = 2

    mm = get_basic_model(G, M, num_gaussian, num_binary, K, num_long_data_pts)

    mm.predictor_names_ = ['intercept', 'x', 'cohort']
    mm.shared_predictor_names_ = ['cohort']
    mm.shared_indices_ = np.array([2])
    mm.traj_indices_ = np.array([0, 1])
    mm.num_shared_preds_ = 1
    mm.num_traj_preds_ = 2

    mm.w_mu_shared_ = torch.zeros((1, 1), dtype=torch.float64)
    mm.w_var_shared_ = torch.ones((1, 1), dtype=torch.float64)
    mm.w_mu0_shared_ = torch.zeros((1, 1), dtype=torch.float64)
    mm.w_var0_shared_ = torch.ones((1, 1), dtype=torch.float64)

    # Shared effect turned on
    mm.w_mu_shared_[0, 0] = 1.5

    # Deliberately put a large nonzero value into the trajectory-specific
    # coefficient at the shared predictor index. The patched compute_waic2()
    # should ignore this for Gaussian targets.
    mm.w_mu_[2, 0, :] = 100.0

    waic0 = mm.compute_waic2(S=50, seed=123)

    # Change the trajectory-specific coefficient at the shared index again.
    # If the implementation is correct, WAIC should remain effectively unchanged.
    mm.w_mu_[2, 0, :] = -250.0

    waic1 = mm.compute_waic2(S=50, seed=123)

    assert np.isfinite(waic0)
    assert np.isfinite(waic1)
    assert np.isclose(waic0, waic1), \
        "Trajectory-specific Gaussian coefficients at shared indices should be ignored in compute_waic2"

def test_compute_waic2_no_shared_predictors_runs():
    G = 2
    M = 2
    num_gaussian = 1
    num_binary = 0
    K = 3
    num_long_data_pts = 2

    mm = get_basic_model(G, M, num_gaussian, num_binary, K, num_long_data_pts)

    mm.shared_predictor_names_ = []
    mm.shared_indices_ = np.array([], dtype=int)
    mm.traj_indices_ = np.array([0, 1])
    mm.num_shared_preds_ = 0
    mm.num_traj_preds_ = 2

    mm.w_mu_shared_ = torch.zeros((0, 1), dtype=torch.float64)
    mm.w_var_shared_ = torch.ones((0, 1), dtype=torch.float64)
    mm.w_mu0_shared_ = torch.zeros((0, 1), dtype=torch.float64)
    mm.w_var0_shared_ = torch.ones((0, 1), dtype=torch.float64)

    waic_val = mm.compute_waic2(S=20, seed=123)

    assert np.isfinite(waic_val)    

def test_plot_with_shared_predictors_runs():
    G = 2
    M = 3
    num_gaussian = 1
    num_binary = 0
    K = 3
    num_long_data_pts = 2

    mm = get_basic_model(G, M, num_gaussian, num_binary, K, num_long_data_pts)

    mm.predictor_names_ = ['intercept', 'x', 'cohort']
    mm.target_names_ = ['y']
    mm.shared_predictor_names_ = ['cohort']
    mm.shared_indices_ = np.array([2])
    mm.traj_indices_ = np.array([0, 1])
    mm.num_shared_preds_ = 1
    mm.num_traj_preds_ = 2
    mm.target_type_[0] = 'gaussian'
    mm.num_binary_targets_ = 0

    mm.w_mu_shared_ = torch.zeros((1, 1), dtype=torch.float64)
    mm.w_var_shared_ = torch.ones((1, 1), dtype=torch.float64)
    mm.w_mu0_shared_ = torch.zeros((1, 1), dtype=torch.float64)
    mm.w_var0_shared_ = torch.ones((1, 1), dtype=torch.float64)

    # Give the dataframe meaningful columns for plotting
    vals_x = np.linspace(0.0, 1.0, mm.N_)
    vals_cohort = np.array([0., 1.] * (mm.N_ // 2 + 1))[:mm.N_]
    mm.df_ = pd.DataFrame({
        'x': vals_x,
        'cohort': vals_cohort,
        'y': np.linspace(0.0, 1.0, mm.N_)
    })

    mm.X_[:, 0] = 1.0
    mm.X_[:, 1] = torch.from_numpy(vals_x).double()
    mm.X_[:, 2] = torch.from_numpy(vals_cohort).double()
    mm.Y_[:, 0] = torch.from_numpy(mm.df_['y'].values).double()

    ax = mm.plot(x_axis='x', y_axis='y', show=False, hide_scatter=True)

    assert ax is not None
    assert hasattr(ax, 'plot')    

def test_plot_changes_with_shared_gaussian_effect():
    G = 2
    M = 3
    num_gaussian = 1
    num_binary = 0
    K = 3
    num_long_data_pts = 2

    mm = get_basic_model(G, M, num_gaussian, num_binary, K, num_long_data_pts)

    mm.predictor_names_ = ['intercept', 'x', 'cohort']
    mm.target_names_ = ['y']
    mm.shared_predictor_names_ = ['cohort']
    mm.shared_indices_ = np.array([2])
    mm.traj_indices_ = np.array([0, 1])
    mm.num_shared_preds_ = 1
    mm.num_traj_preds_ = 2
    mm.target_type_[0] = 'gaussian'
    mm.num_binary_targets_ = 0

    mm.w_mu_shared_ = torch.zeros((1, 1), dtype=torch.float64)
    mm.w_var_shared_ = torch.ones((1, 1), dtype=torch.float64)
    mm.w_mu0_shared_ = torch.zeros((1, 1), dtype=torch.float64)
    mm.w_var0_shared_ = torch.ones((1, 1), dtype=torch.float64)

    vals_x = np.linspace(0.0, 1.0, mm.N_)
    vals_cohort = np.ones(mm.N_)  # make shared effect visible everywhere
    mm.df_ = pd.DataFrame({
        'x': vals_x,
        'cohort': vals_cohort,
        'y': np.linspace(0.0, 1.0, mm.N_)
    })

    mm.X_[:, 0] = 1.0
    mm.X_[:, 1] = torch.from_numpy(vals_x).double()
    mm.X_[:, 2] = torch.from_numpy(vals_cohort).double()
    mm.Y_[:, 0] = torch.from_numpy(mm.df_['y'].values).double()

    ax0 = mm.plot(x_axis='x', y_axis='y', show=False, hide_scatter=True)
    ydata0 = ax0.lines[0].get_ydata().copy()
    plt.close(ax0.figure)

    mm.w_mu_shared_[0, 0] = 2.0

    ax1 = mm.plot(x_axis='x', y_axis='y', show=False, hide_scatter=True)
    ydata1 = ax1.lines[0].get_ydata().copy()
    plt.close(ax1.figure)

    assert np.all(np.isfinite(ydata0))
    assert np.all(np.isfinite(ydata1))
    assert not np.allclose(ydata0, ydata1), \
        "Plotted Gaussian trajectory mean should change when shared coefficient changes"    

def test_plot_shared_effect_not_double_counted():
    G = 2
    M = 3
    num_gaussian = 1
    num_binary = 0
    K = 3
    num_long_data_pts = 2

    mm = get_basic_model(G, M, num_gaussian, num_binary, K, num_long_data_pts)

    mm.predictor_names_ = ['intercept', 'x', 'cohort']
    mm.target_names_ = ['y']
    mm.shared_predictor_names_ = ['cohort']
    mm.shared_indices_ = np.array([2])
    mm.traj_indices_ = np.array([0, 1])
    mm.num_shared_preds_ = 1
    mm.num_traj_preds_ = 2
    mm.target_type_[0] = 'gaussian'
    mm.num_binary_targets_ = 0

    mm.w_mu_shared_ = torch.zeros((1, 1), dtype=torch.float64)
    mm.w_var_shared_ = torch.ones((1, 1), dtype=torch.float64)
    mm.w_mu0_shared_ = torch.zeros((1, 1), dtype=torch.float64)
    mm.w_var0_shared_ = torch.ones((1, 1), dtype=torch.float64)

    vals_x = np.linspace(0.0, 1.0, mm.N_)
    vals_cohort = np.ones(mm.N_)
    mm.df_ = pd.DataFrame({
        'x': vals_x,
        'cohort': vals_cohort,
        'y': np.linspace(0.0, 1.0, mm.N_)
    })

    mm.X_[:, 0] = 1.0
    mm.X_[:, 1] = torch.from_numpy(vals_x).double()
    mm.X_[:, 2] = torch.from_numpy(vals_cohort).double()
    mm.Y_[:, 0] = torch.from_numpy(mm.df_['y'].values).double()

    mm.w_mu_shared_[0, 0] = 1.5

    mm.w_mu_[2, 0, :] = 100.0
    ax0 = mm.plot(x_axis='x', y_axis='y', show=False, hide_scatter=True)
    ydata0 = ax0.lines[0].get_ydata().copy()
    plt.close(ax0.figure)

    mm.w_mu_[2, 0, :] = -250.0
    ax1 = mm.plot(x_axis='x', y_axis='y', show=False, hide_scatter=True)
    ydata1 = ax1.lines[0].get_ydata().copy()
    plt.close(ax1.figure)

    assert np.allclose(ydata0, ydata1), \
        "Shared predictor contribution in plot() should not be double-counted through trajectory-specific coefficients"    

def test_plot_no_shared_predictors_runs():
    G = 2
    M = 2
    num_gaussian = 1
    num_binary = 0
    K = 3
    num_long_data_pts = 2

    mm = get_basic_model(G, M, num_gaussian, num_binary, K, num_long_data_pts)

    mm.predictor_names_ = ['intercept', 'x']
    mm.target_names_ = ['y']
    mm.shared_predictor_names_ = []
    mm.shared_indices_ = np.array([], dtype=int)
    mm.traj_indices_ = np.array([0, 1])
    mm.num_shared_preds_ = 0
    mm.num_traj_preds_ = 2
    mm.target_type_[0] = 'gaussian'
    mm.num_binary_targets_ = 0

    mm.w_mu_shared_ = torch.zeros((0, 1), dtype=torch.float64)
    mm.w_var_shared_ = torch.ones((0, 1), dtype=torch.float64)
    mm.w_mu0_shared_ = torch.zeros((0, 1), dtype=torch.float64)
    mm.w_var0_shared_ = torch.ones((0, 1), dtype=torch.float64)

    vals_x = np.linspace(0.0, 1.0, mm.N_)
    mm.df_ = pd.DataFrame({
        'x': vals_x,
        'y': np.linspace(0.0, 1.0, mm.N_)
    })

    mm.X_[:, 0] = 1.0
    mm.X_[:, 1] = torch.from_numpy(vals_x).double()
    mm.Y_[:, 0] = torch.from_numpy(mm.df_['y'].values).double()

    ax = mm.plot(x_axis='x', y_axis='y', show=False, hide_scatter=True)

    assert ax is not None
    plt.close(ax.figure)


def test_get_R_matrix_changes_with_shared_gaussian_effect():
    G = 2
    M = 3
    num_gaussian = 1
    num_binary = 0
    K = 3
    num_long_data_pts = 2

    mm = get_basic_model(G, M, num_gaussian, num_binary, K, num_long_data_pts)

    mm.predictor_names_ = ['intercept', 'x', 'cohort']
    mm.target_names_ = ['y']
    mm.shared_predictor_names_ = ['cohort']
    mm.shared_indices_ = np.array([2])
    mm.traj_indices_ = np.array([0, 1])
    mm.num_shared_preds_ = 1
    mm.num_traj_preds_ = 2
    mm.target_type_[0] = 'gaussian'
    mm.num_binary_targets_ = 0

    mm.w_mu_shared_ = torch.zeros((1, 1), dtype=torch.float64)
    mm.w_var_shared_ = torch.ones((1, 1), dtype=torch.float64)
    mm.w_mu0_shared_ = torch.zeros((1, 1), dtype=torch.float64)
    mm.w_var0_shared_ = torch.ones((1, 1), dtype=torch.float64)

    mm.v_a_ = torch.ones(mm.K_, dtype=torch.float64)
    mm.v_b_ = torch.ones(mm.K_, dtype=torch.float64)

    vals_x = np.linspace(0.0, 1.0, mm.N_)
    vals_cohort = np.array([0., 1.] * (mm.N_ // 2 + 1))[:mm.N_]
    vals_y = np.linspace(0.0, 1.0, mm.N_)
    vals_sid = [f'sid_{i // num_long_data_pts}' for i in range(mm.N_)]

    mm.df_ = pd.DataFrame({
        'intercept': np.ones(mm.N_),
        'sid': vals_sid,
        'x': vals_x,
        'cohort': vals_cohort,
        'y': vals_y
    })

    mm.X_[:, 0] = 1.0
    mm.X_[:, 1] = torch.from_numpy(vals_x).double()
    mm.X_[:, 2] = torch.from_numpy(vals_cohort).double()
    mm.Y_[:, 0] = torch.from_numpy(vals_y).double()

    mm.gb_ = mm.df_.groupby('sid')
    mm._set_group_first_index(mm.df_, mm.gb_)
    mm._set_N_to_G_index_map()

    R0 = mm.get_R_matrix(df=mm.df_, gb_col='sid')

    mm.w_mu_shared_[0, 0] = 2.0
    R1 = mm.get_R_matrix(df=mm.df_, gb_col='sid')

    assert torch.all(torch.isfinite(R0))
    assert torch.all(torch.isfinite(R1))
    assert not torch.allclose(R0, R1), \
        "R matrix should change when shared Gaussian coefficient changes"

def test_get_R_matrix_shared_variance_not_double_counted():
    G = 2
    M = 3
    num_gaussian = 1
    num_binary = 0
    K = 3
    num_long_data_pts = 2

    mm = get_basic_model(G, M, num_gaussian, num_binary, K, num_long_data_pts)

    mm.predictor_names_ = ['intercept', 'x', 'cohort']
    mm.target_names_ = ['y']
    mm.shared_predictor_names_ = ['cohort']
    mm.shared_indices_ = np.array([2])
    mm.traj_indices_ = np.array([0, 1])
    mm.num_shared_preds_ = 1
    mm.num_traj_preds_ = 2
    mm.target_type_[0] = 'gaussian'
    mm.num_binary_targets_ = 0

    mm.w_mu_shared_ = torch.zeros((1, 1), dtype=torch.float64)
    mm.w_var_shared_ = torch.ones((1, 1), dtype=torch.float64)
    mm.w_mu0_shared_ = torch.zeros((1, 1), dtype=torch.float64)
    mm.w_var0_shared_ = torch.ones((1, 1), dtype=torch.float64)

    mm.v_a_ = torch.ones(mm.K_, dtype=torch.float64)
    mm.v_b_ = torch.ones(mm.K_, dtype=torch.float64)

    vals_x = np.linspace(0.0, 1.0, mm.N_)
    vals_cohort = np.array([0., 1.] * (mm.N_ // 2 + 1))[:mm.N_]
    vals_y = np.linspace(0.0, 1.0, mm.N_)
    vals_sid = [f'sid_{i // num_long_data_pts}' for i in range(mm.N_)]

    mm.df_ = pd.DataFrame({
        'intercept': np.ones(mm.N_),
        'sid': vals_sid,
        'x': vals_x,
        'cohort': vals_cohort,
        'y': vals_y
    })

    mm.X_[:, 0] = 1.0
    mm.X_[:, 1] = torch.from_numpy(vals_x).double()
    mm.X_[:, 2] = torch.from_numpy(vals_cohort).double()
    mm.Y_[:, 0] = torch.from_numpy(vals_y).double()

    mm.gb_ = mm.df_.groupby('sid')
    mm._set_group_first_index(mm.df_, mm.gb_)
    mm._set_N_to_G_index_map()

    # Give shared block a nontrivial variance contribution
    mm.w_var_shared_[0, 0] = 2.0

    # Deliberately perturb trajectory-specific variance at the shared index.
    # Correct behavior: this should NOT affect R.
    mm.w_var_[2, 0, :] = 100.0
    R0 = mm.get_R_matrix(df=mm.df_, gb_col='sid')

    mm.w_var_[2, 0, :] = 0.001
    R1 = mm.get_R_matrix(df=mm.df_, gb_col='sid')

    assert torch.all(torch.isfinite(R0))
    assert torch.all(torch.isfinite(R1))
    assert torch.allclose(R0, R1, atol=1e-8, rtol=1e-8), \
        "Trajectory-specific variance at a shared predictor index should be ignored in get_R_matrix"    

def test_update_lambda_shared_variance_not_double_counted():
    G = 2
    M = 3
    num_gaussian = 1
    num_binary = 0
    K = 3
    num_long_data_pts = 2

    mm = get_basic_model(G, M, num_gaussian, num_binary, K, num_long_data_pts)

    mm.predictor_names_ = ['intercept', 'x', 'cohort']
    mm.target_names_ = ['y']
    mm.shared_predictor_names_ = ['cohort']
    mm.shared_indices_ = np.array([2])
    mm.traj_indices_ = np.array([0, 1])
    mm.num_shared_preds_ = 1
    mm.num_traj_preds_ = 2
    mm.target_type_[0] = 'gaussian'
    mm.num_binary_targets_ = 0
    mm.sig_trajs_ = torch.tensor([True] * mm.K_)

    mm.w_mu_shared_ = torch.zeros((1, 1), dtype=torch.float64)
    mm.w_var_shared_ = torch.ones((1, 1), dtype=torch.float64)
    mm.w_mu0_shared_ = torch.zeros((1, 1), dtype=torch.float64)
    mm.w_var0_shared_ = torch.ones((1, 1), dtype=torch.float64)

    vals_x = np.linspace(0.0, 1.0, mm.N_)
    vals_cohort = np.array([0., 1.] * (mm.N_ // 2 + 1))[:mm.N_]
    vals_y = np.linspace(0.0, 1.0, mm.N_)

    mm.X_[:, 0] = 1.0
    mm.X_[:, 1] = torch.from_numpy(vals_x).double()
    mm.X_[:, 2] = torch.from_numpy(vals_cohort).double()
    mm.Y_[:, 0] = torch.from_numpy(vals_y).double()

    # Make sure prior-modified lambda params exist as they would after fit()
    mm.lambda_a0_mod_ = mm.lambda_a0_.clone()
    mm.lambda_b0_mod_ = mm.lambda_b0_.clone()

    # Give shared block a nontrivial variance contribution
    mm.w_var_shared_[0, 0] = 2.0

    mm.w_var_[2, 0, :] = 100.0
    mm.update_lambda()
    b0 = mm.lambda_b_[0, :].clone()

    mm.w_var_[2, 0, :] = 0.001
    mm.update_lambda()
    b1 = mm.lambda_b_[0, :].clone()

    assert torch.all(torch.isfinite(b0))
    assert torch.all(torch.isfinite(b1))
    assert torch.allclose(b0, b1, atol=1e-8, rtol=1e-8), \
        "Trajectory-specific variance at a shared predictor index should be ignored in update_lambda"    

def test_shared_gaussian_effect_simulation_recovery():
    np.random.seed(123)
    torch.manual_seed(123)

    # ------------------------------------------------------------------
    # Simulate longitudinal Gaussian data
    # ------------------------------------------------------------------
    n_subjects = 120
    n_timepoints = 5
    K_fit = 6

    subject_ids = []
    time_vals = []
    cohort_vals = []
    y_vals = []

    true_traj = np.random.binomial(1, 0.5, size=n_subjects)
    subject_cohort = np.random.binomial(1, 0.5, size=n_subjects)

    # True parameters
    beta_cohort = 2.0  # shared effect

    # trajectory-specific intercepts/slopes
    intercepts = {0: 0.0, 1: 1.0}
    slopes = {0: 0.2, 1: 1.0}

    sigma = 0.35

    for g in range(n_subjects):
        sid = f'subj_{g}'
        k = true_traj[g]
        cohort = subject_cohort[g]

        for t in range(n_timepoints):
            mu = (
                intercepts[k]
                + slopes[k] * t
                + beta_cohort * cohort
            )
            y = np.random.normal(mu, sigma)

            subject_ids.append(sid)
            time_vals.append(float(t))
            cohort_vals.append(float(cohort))
            y_vals.append(float(y))

    df = pd.DataFrame({
        'id': subject_ids,
        'intercept': 1.0,
        'time': time_vals,
        'cohort': cohort_vals,
        'y': y_vals
    })

    # ------------------------------------------------------------------
    # Priors / model setup
    # ------------------------------------------------------------------
    predictor_names = ['intercept', 'time', 'cohort']
    target_names = ['y']

    M = len(predictor_names)
    D = len(target_names)

    w_mu0 = np.zeros((M, D))
    w_var0 = np.ones((M, D)) * 10.0

    lambda_a0 = np.ones(D)
    lambda_b0 = np.ones(D)

    alpha = 1.0

    mm = MultDPRegression(
        w_mu0, w_var0,
        lambda_a0, lambda_b0,
        1, alpha,
        K=K_fit
    )

    mm.fit(
        target_names=target_names,
        predictor_names=predictor_names,
        df=df,
        groupby='id',
        iters=60,
        verbose=False,
        shared_predictor_names=['cohort']
    )

    # ------------------------------------------------------------------
    # Basic sanity checks
    # ------------------------------------------------------------------
    assert torch.all(torch.isfinite(mm.w_mu_shared_))
    assert torch.all(torch.isfinite(mm.w_var_shared_))
    assert torch.all(torch.isfinite(mm.w_mu_))
    assert torch.all(torch.isfinite(mm.w_var_))
    assert torch.all(torch.isfinite(mm.R_))

    # ------------------------------------------------------------------
    # Shared coefficient should be learned with correct sign and
    # nontrivial magnitude
    # ------------------------------------------------------------------
    cohort_idx = predictor_names.index('cohort')
    time_idx = predictor_names.index('time')
    y_idx = 0

    fitted_shared = mm.w_mu_shared_[0, y_idx].item()

    assert fitted_shared > 0.5, \
        f"Expected positive shared cohort effect; got {fitted_shared}"

    # ------------------------------------------------------------------
    # Shared predictor should remain inactive in trajectory-specific block
    # ------------------------------------------------------------------
    assert torch.allclose(
        mm.w_mu_[cohort_idx, y_idx, :],
        torch.zeros_like(mm.w_mu_[cohort_idx, y_idx, :]),
        atol=1e-6
    ), "Shared predictor should be inactive in trajectory-specific Gaussian block"

    # ------------------------------------------------------------------
    # Active trajectory-specific slopes should show heterogeneity
    # ------------------------------------------------------------------
    active_trajs = torch.where(mm.sig_trajs_)[0].cpu().numpy()

    # Need at least two active trajectories for this check to be meaningful
    assert len(active_trajs) >= 2, \
        "Expected at least two active trajectories in simulation recovery test"

    active_slopes = mm.w_mu_[time_idx, y_idx, active_trajs].detach().cpu().numpy()

    # Check that at least one pair differs substantially
    max_slope_diff = 0.0
    for i in range(len(active_slopes)):
        for j in range(i + 1, len(active_slopes)):
            max_slope_diff = max(max_slope_diff, abs(active_slopes[i] - active_slopes[j]))

    assert max_slope_diff > 0.3, \
        f"Expected trajectory-specific slope heterogeneity; max diff was {max_slope_diff}"

    # ------------------------------------------------------------------
    # Optional: log-likelihood should be finite
    # ------------------------------------------------------------------
    ll = mm.log_likelihood()
    assert torch.isfinite(ll), "Log-likelihood should be finite after fitting"    


def test_shared_gaussian_effect_simulation_null_vs_signal():
    def run_sim(beta_cohort, seed):
        np.random.seed(seed)
        torch.manual_seed(seed)

        n_subjects = 120
        n_timepoints = 5
        K_fit = 6

        subject_ids = []
        time_vals = []
        cohort_vals = []
        y_vals = []

        true_traj = np.random.binomial(1, 0.5, size=n_subjects)
        subject_cohort = np.random.binomial(1, 0.5, size=n_subjects)

        intercepts = {0: 0.0, 1: 1.0}
        slopes = {0: 0.2, 1: 1.0}
        sigma = 0.35

        for g in range(n_subjects):
            sid = f'subj_{g}'
            k = true_traj[g]
            cohort = subject_cohort[g]

            for t in range(n_timepoints):
                mu = intercepts[k] + slopes[k] * t + beta_cohort * cohort
                y = np.random.normal(mu, sigma)

                subject_ids.append(sid)
                time_vals.append(float(t))
                cohort_vals.append(float(cohort))
                y_vals.append(float(y))

        df = pd.DataFrame({
            'id': subject_ids,
            'intercept': 1.0,
            'time': time_vals,
            'cohort': cohort_vals,
            'y': y_vals
        })

        predictor_names = ['intercept', 'time', 'cohort']
        target_names = ['y']

        M = len(predictor_names)
        D = len(target_names)

        w_mu0 = np.zeros((M, D))
        w_var0 = np.ones((M, D)) * 10.0
        lambda_a0 = np.ones(D)
        lambda_b0 = np.ones(D)

        mm = MultDPRegression(
            w_mu0, w_var0,
            lambda_a0, lambda_b0,
            1, 1.0,
            K=K_fit
        )

        mm.fit(
            target_names=target_names,
            predictor_names=predictor_names,
            df=df,
            groupby='id',
            iters=60,
            verbose=False,
            shared_predictor_names=['cohort']
        )

        return mm

    mm_null = run_sim(beta_cohort=0.0, seed=456)
    mm_signal = run_sim(beta_cohort=2.0, seed=123)

    fitted_null = mm_null.w_mu_shared_[0, 0].item()
    fitted_signal = mm_signal.w_mu_shared_[0, 0].item()

    assert torch.all(torch.isfinite(mm_null.w_mu_shared_))
    assert torch.all(torch.isfinite(mm_signal.w_mu_shared_))

    assert fitted_signal > 0.5, \
        f"Expected positive shared effect in signal case; got {fitted_signal}"

    assert fitted_signal > fitted_null + 0.5, \
        f"Expected signal shared effect to exceed null by a meaningful margin; null={fitted_null}, signal={fitted_signal}"

    assert torch.allclose(
        mm_null.w_mu_[2, 0, :],
        torch.zeros_like(mm_null.w_mu_[2, 0, :]),
        atol=1e-6
    )
    assert torch.allclose(
        mm_signal.w_mu_[2, 0, :],
        torch.zeros_like(mm_signal.w_mu_[2, 0, :]),
        atol=1e-6
    )    

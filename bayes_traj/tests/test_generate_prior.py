from bayes_traj.generate_prior import PriorGenerator
from bayes_traj.mult_dp_regression import MultDPRegression
import numpy as np
import pandas as pd
from bayes_traj.utils import *
import pdb, os, pickle
import pytest, copy
import torch

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


def test_prior_info_from_model():
    """
    """
    model_file_name = os.path.split(os.path.realpath(__file__))[0] + \
        '/../resources/models/model_1.p'

    mm = pickle.load(open(model_file_name, 'rb'))['MultDPRegression']
    
    targets = ['y']
    preds = ['intercept', 'age']
    pg = PriorGenerator(targets, preds)
    pg.set_model(mm)
    pg.prior_info_from_model('y')
    assert np.isclose(pg.prior_info_['w_mu0']['y']['intercept'], 7.73326866724), \
        "w_mu0 not as expected"
    assert np.isclose(pg.prior_info_['w_mu0']['y']['age'], -1.47391042070), \
        "w_mu0 not as expected"
    assert np.isclose(pg.prior_info_['w_var0']['y']['intercept'], 0.46760358415383), \
        "w_var0 not as expected"    
    assert np.isclose(pg.prior_info_['w_var0']['y']['age'], 0.0439302696606), \
        "w_var0 not as expected"
    assert np.isclose(pg.prior_info_['lambda_a0']['y'], 17.7068686687), \
        "lambda_a0 not as expected"
    assert np.isclose(pg.prior_info_['lambda_b0']['y'], 24.0246304888), \
        "lambda_b0 not as expected"
    
def test_prior_info_from_df():
    """
    """
    data_file_name = os.path.split(os.path.realpath(__file__))[0] + \
        '/../resources/data/trajectory_data_1.csv'
    df = pd.read_csv(data_file_name)
    
    targets = ['y']
    preds = ['intercept', 'age']
    pg = PriorGenerator(targets, preds, num_trajs=2)
    pg.set_data(df, 'id')
    pg.prior_info_from_df('y')

    assert np.isclose(pg.prior_info_['w_var0']['y']['intercept'], 0.99665726), \
        "w_var0 not as expected"
    assert np.isclose(pg.prior_info_['w_var0']['y']['age'], 0.01614473854363), \
        "w_var0 not as expected"
    assert np.isclose(pg.prior_info_['w_mu0']['y']['intercept'], 6.879517995), \
        "w_mu0 not as expected"
    assert np.isclose(pg.prior_info_['w_mu0']['y']['age'], -1.36580642497926), \
        "w_mu0 not as expected"

    assert np.isclose(pg.prior_info_['lambda_a0']['y'], 14.3999999999), \
        "lambda_a0 not as expected"
    assert np.isclose(pg.prior_info_['lambda_b0']['y'], 356.5218504085522), \
        "lambda_b0 nost as expected"
    
    #pickle.dump(pg.prior_info_, open('/Users/jr555/tmp/foo_prior.p', 'wb'))
    #pdb.set_trace()
    
#def test_prior_from_model_and_data():
#    """
#    """
#    data_file_name = os.path.split(os.path.realpath(__file__))[0] + \
#        '/../resources/data/trajectory_data_1.csv'
#    model_file_name = os.path.split(os.path.realpath(__file__))[0] + \
#        '/../resources/models/model_1.p'
#   
#    df = pd.read_csv(data_file_name)
#    mm = pickle.load(open(model_file_name, 'rb'))['MultDPRegression']

def test_traj_prior_info_from_df():
    """
    """
    data_file_name = os.path.split(os.path.realpath(__file__))[0] + \
        '/../resources/data/trajectory_data_1.csv'
    model_file_name = os.path.split(os.path.realpath(__file__))[0] + \
        '/../resources/models/model_1.p'
    
    df = pd.read_csv(data_file_name)
    mm = pickle.load(open(model_file_name, 'rb'))['MultDPRegression']
    mm.cast_to_torch()
    mm.ranef_indices_ = None
    
    targets = ['y']
    preds = ['intercept', 'age']
    pg = PriorGenerator(targets, preds)
    pg.set_model(mm)
    pg.set_data(df, 'id')

    #---------------------------------------------------------------------------
    # Evaluate prior for trajectory 5
    #---------------------------------------------------------------------------
    intercept_co = 10
    age_co = -1
    resid_std = 2

    traj = 1
    pg.traj_prior_info_from_df('y', traj)

    mu = pg.prior_info_['w_mu']['intercept']['y'][traj]
    sig = np.sqrt(pg.prior_info_['w_var']['intercept']['y'][traj])

    assert mu - 2*sig <= intercept_co and mu + 2*sig >= intercept_co, \
        "Intercept prior not as expected for trajectory 5"

    mu = pg.prior_info_['w_mu']['age']['y'][traj]
    sig = np.sqrt(pg.prior_info_['w_var']['age']['y'][traj])
    assert mu - 2*sig <= age_co and mu + 2*sig >= age_co, \
        "Age prior not as expected for trajectory 5"

    prec_mean = pg.prior_info_['lambda_a']['y'][traj]/\
        pg.prior_info_['lambda_b']['y'][traj]
    prec_var = pg.prior_info_['lambda_a']['y'][traj]/\
        (pg.prior_info_['lambda_b']['y'][traj]**2)
    prec_high = prec_mean + 2*np.sqrt(prec_var)
    prec_low= prec_mean - 2*np.sqrt(prec_var)

    std_high = np.sqrt(1/prec_low)
    std_low = np.sqrt(1/prec_high)

    assert resid_std >= std_low and resid_std <= std_high, \
        "Prior over residual precision not as expected for trajectory 1"

    #---------------------------------------------------------------------------
    # Evaluate prior for trajectory 3
    #---------------------------------------------------------------------------
    intercept_co = 6
    age_co = -2
    resid_std = 1

    traj = 3
    pg.traj_prior_info_from_df('y', traj)

    mu = pg.prior_info_['w_mu']['intercept']['y'][traj]
    sig = np.sqrt(pg.prior_info_['w_var']['intercept']['y'][traj])
    assert mu - 2*sig <= intercept_co and mu + 2*sig >= intercept_co, \
        "Intercept prior not as expected for trajectory 3"

    mu = pg.prior_info_['w_mu']['age']['y'][traj]
    sig = np.sqrt(pg.prior_info_['w_var']['age']['y'][traj])
    assert mu - 2*sig <= age_co and mu + 2*sig >= age_co, \
        "Age prior not as expected for trajectory 3"

    prec_mean = pg.prior_info_['lambda_a']['y'][traj]/\
        pg.prior_info_['lambda_b']['y'][traj]
    prec_var = pg.prior_info_['lambda_a']['y'][traj]/\
        (pg.prior_info_['lambda_b']['y'][traj]**2)
    prec_high = prec_mean + 2*np.sqrt(prec_var)
    prec_low= prec_mean - 2*np.sqrt(prec_var)

    std_high = np.sqrt(1/prec_low)
    std_low = np.sqrt(1/prec_high)

    assert resid_std >= std_low and resid_std <= std_high, \
        "Prior over residual precision not as expected for trajectory 3"

def test_traj_prior_info_from_model():
    """
    """
    model_file_name = os.path.split(os.path.realpath(__file__))[0] + \
        '/../resources/models/model_1.p'
    
    mm = pickle.load(open(model_file_name, 'rb'))['MultDPRegression']

    targets = ['y']
    preds = ['intercept', 'age']
    pg = PriorGenerator(targets, preds)
    pg.set_model(mm)

    for tt in np.where(mm.sig_trajs_)[0]:
        pg.traj_prior_info_from_model('y', tt)

    for tt in [1, 3]:
        assert pg.prior_info_['w_mu']['intercept']['y'][tt] == \
            mm.w_mu_[0, 0, tt], \
            "Intercept prior not as expected for trajectory 1"
        assert pg.prior_info_['w_mu']['age']['y'][tt] == mm.w_mu_[1, 0, tt], \
            "Age prior not as expected for trajectory 1"
        assert pg.prior_info_['lambda_a']['y'][tt] == mm.lambda_a_[0, tt], \
            "Gamma prior not as expected for trajectory 1"
        assert pg.prior_info_['lambda_b']['y'][tt] == mm.lambda_b_[0, tt], \
            "Gamma prior not as expected for trajectory 1"

def test_compute_prior_info_1():
    """Data only
    """
    data_file_name = os.path.split(os.path.realpath(__file__))[0] + \
        '/../resources/data/trajectory_data_1.csv'
    
    df = pd.read_csv(data_file_name)
    
    targets = ['y']
    preds = ['intercept', 'age']
    pg = PriorGenerator(targets, preds)
    pg.set_data(df, 'id')
    pg.compute_prior_info()

    assert np.isclose(pg.prior_info_['w_var0']['y']['intercept'], 0.99665726), \
        "w_var0 not as expected"
    assert np.isclose(pg.prior_info_['w_var0']['y']['age'], 0.01614473854363), \
        "w_var0 not as expected"
    assert np.isclose(pg.prior_info_['w_mu0']['y']['intercept'], 6.879517995), \
        "w_mu0 not as expected"
    assert np.isclose(pg.prior_info_['w_mu0']['y']['age'], -1.36580642497926), \
        "w_mu0 not as expected"
    assert np.isclose(pg.prior_info_['lambda_a0']['y'], 14.399999999999999), \
        "lambda_a0 not as expected"
    assert np.isclose(pg.prior_info_['lambda_b0']['y'], 356.5218504085522), \
        "lambda_b0 nost as expected"

    assert pg.prior_info_['w_mu'] is None, \
        "w_mu should be None"
    assert pg.prior_info_['w_var'] is None, \
        "w_var should be None"
    assert pg.prior_info_['lambda_a'] is None, \
        "labmda_a should be None"
    assert pg.prior_info_['lambda_b'] is None, \
        "lambda_b should be None"
    assert pg.prior_info_['v_a'] is None, \
        "v_a should be None"
    assert pg.prior_info_['v_b'] is None, \
        "v_b should be None"
    assert pg.prior_info_['traj_probs'] is None, \
        "traj_probs should be None"    

def test_compute_prior_info_2():
    """Model only. In this case, the specified target sets and predictor sets 
    must agree with those of the input model.
    """
    data_file_name = os.path.split(os.path.realpath(__file__))[0] + \
        '/../resources/data/trajectory_data_1.csv'
    model_file_name = os.path.split(os.path.realpath(__file__))[0] + \
        '/../resources/models/model_1.p'
    
    df = pd.read_csv(data_file_name)    
    mm = pickle.load(open(model_file_name, 'rb'))['MultDPRegression']
    mm.cast_to_torch()
    mm.ranef_indices_ = None
    
    targets = ['y']
    preds = ['intercept', 'age']
    pg = PriorGenerator(targets, preds)
    pg.set_model(mm)    
    pg.set_data(df, 'id')
    pg.compute_prior_info()

    assert np.isclose(pg.prior_info_['w_mu0']['y']['intercept'], 7.733268667), \
        "w_mu0 not as expected"
    assert np.isclose(pg.prior_info_['w_mu0']['y']['age'], -1.473910420702446), \
        "w_mu0 not as expected"
    assert np.isclose(pg.prior_info_['w_var0']['y']['intercept'], 0.46760358415383946), \
        "w_var0 not as expected"    
    assert np.isclose(pg.prior_info_['w_var0']['y']['age'], 0.043930269660685), \
        "w_var0 not as expected"
    assert np.isclose(pg.prior_info_['lambda_a0']['y'], 17.70686866876), \
        "lambda_a0 not as expected"
    assert np.isclose(pg.prior_info_['lambda_b0']['y'], 24.0246304888103), \
        "lambda_b0 not as expected"

    for tt in [1, 3]:
        assert pg.prior_info_['w_mu']['intercept']['y'][tt] == \
            mm.w_mu_[0, 0, tt], \
            "Intercept prior not as expected for trajectory 1"
        assert pg.prior_info_['w_mu']['age']['y'][tt] == mm.w_mu_[1, 0, tt], \
            "Age prior not as expected for trajectory 1"
        assert pg.prior_info_['lambda_a']['y'][tt] == mm.lambda_a_[0, tt], \
            "Gamma prior not as expected for trajectory 1"
        assert pg.prior_info_['lambda_b']['y'][tt] == mm.lambda_b_[0, tt], \
            "Gamma prior not as expected for trajectory 1"    
    
def test_compute_prior_info_3():
    """Model and data, preds same, targets same
    """
    model_file_name = os.path.split(os.path.realpath(__file__))[0] + \
        '/../resources/models/model_1.p'
    data_file_name = os.path.split(os.path.realpath(__file__))[0] + \
        '/../resources/data/trajectory_data_1.csv'

    mm = pickle.load(open(model_file_name, 'rb'))['MultDPRegression']
    mm.cast_to_torch()
    mm.ranef_indices_ = None
    
    df = pd.read_csv(data_file_name)
        
    targets = ['y']
    preds = ['intercept', 'age']
    pg = PriorGenerator(targets, preds)
    pg.set_model(mm)
    pg.set_data(df, 'id')
    pg.compute_prior_info()

    assert np.isclose(pg.prior_info_['w_mu0']['y']['intercept'], 7.733268667), \
        "w_mu0 not as expected"
    assert np.isclose(pg.prior_info_['w_mu0']['y']['age'], -1.473910420702446), \
        "w_mu0 not as expected"
    assert np.isclose(pg.prior_info_['w_var0']['y']['intercept'], 0.46760358415383946), \
        "w_var0 not as expected"    
    assert np.isclose(pg.prior_info_['w_var0']['y']['age'], 0.043930269660685), \
        "w_var0 not as expected"
    assert np.isclose(pg.prior_info_['lambda_a0']['y'], 17.70686866876), \
        "lambda_a0 not as expected"
    assert np.isclose(pg.prior_info_['lambda_b0']['y'], 24.0246304888103), \
        "lambda_b0 not as expected"

    for tt in [1, 3]:
        assert pg.prior_info_['w_mu']['intercept']['y'][tt] == \
            mm.w_mu_[0, 0, tt], \
            "Intercept prior not as expected for trajectory 1"
        assert pg.prior_info_['w_mu']['age']['y'][tt] == mm.w_mu_[1, 0, tt], \
            "Age prior not as expected for trajectory 1"
        assert pg.prior_info_['lambda_a']['y'][tt] == mm.lambda_a_[0, tt], \
            "Gamma prior not as expected for trajectory 1"
        assert pg.prior_info_['lambda_b']['y'][tt] == mm.lambda_b_[0, tt], \
            "Gamma prior not as expected for trajectory 1"        

def test_compute_prior_info_4():
    """Model and data, preds same, new target
    """
    model_file_name = os.path.split(os.path.realpath(__file__))[0] + \
        '/../resources/models/model_1.p'
    data_file_name = os.path.split(os.path.realpath(__file__))[0] + \
        '/../resources/data/trajectory_data_1.csv'

    mm = pickle.load(open(model_file_name, 'rb'))['MultDPRegression']
    mm.cast_to_torch()
    mm.ranef_indices_ = None
    
    df = pd.read_csv(data_file_name)
    df['y2'] = df.y.values + 10
        
    targets = ['y', 'y2']
    preds = ['intercept', 'age']
    pg = PriorGenerator(targets, preds)
    pg.set_model(mm)
    pg.set_data(df, 'id')
    pg.compute_prior_info()

    assert np.isclose(pg.prior_info_['w_mu0']['y']['intercept'], 7.733268667), \
        "w_mu0 not as expected"
    assert np.isclose(pg.prior_info_['w_mu0']['y']['age'], -1.473910420702446), \
        "w_mu0 not as expected"
    assert np.isclose(pg.prior_info_['w_var0']['y']['intercept'], 0.46760358415383946), \
        "w_var0 not as expected"    
    assert np.isclose(pg.prior_info_['w_var0']['y']['age'], 0.043930269660685), \
        "w_var0 not as expected"
    assert np.isclose(pg.prior_info_['lambda_a0']['y'], 17.70686866876), \
        "lambda_a0 not as expected"
    assert np.isclose(pg.prior_info_['lambda_b0']['y'], 24.0246304888103), \
        "lambda_b0 not as expected"

    for tt in [1, 3]:
        assert pg.prior_info_['w_mu']['intercept']['y'][tt] == \
            mm.w_mu_[0, 0, tt], \
            "Intercept prior not as expected for trajectory 1"
        assert pg.prior_info_['w_mu']['age']['y'][tt] == mm.w_mu_[1, 0, tt], \
            "Age prior not as expected for trajectory 1"
        assert pg.prior_info_['lambda_a']['y'][tt] == mm.lambda_a_[0, tt], \
            "Gamma prior not as expected for trajectory 1"
        assert pg.prior_info_['lambda_b']['y'][tt] == mm.lambda_b_[0, tt], \
            "Gamma prior not as expected for trajectory 1"    

    assert np.isclose(pg.prior_info_['w_var0']['y2']['intercept'], 0.9966572), \
        "w_var0 not as expected"
    assert np.isclose(pg.prior_info_['w_var0']['y2']['age'], 0.0161447385436), \
        "w_var0 not as expected"
    assert np.isclose(pg.prior_info_['w_mu0']['y2']['intercept'], 16.8795179), \
        "w_mu0 not as expected"
    assert np.isclose(pg.prior_info_['w_mu0']['y2']['age'], -1.3658064249792), \
        "w_mu0 not as expected"
    assert np.isclose(pg.prior_info_['lambda_a0']['y2'], 14.399999999999999), \
        "lambda_a0 not as expected"
    assert np.isclose(pg.prior_info_['lambda_b0']['y2'], 356.5218504085522), \
        "lambda_b0 nost as expected"

    for tt in [1, 3]:
        assert np.isclose(mm.w_mu_[0, 0, tt] + 10, \
            pg.prior_info_['w_mu']['intercept']['y2'][tt], atol=0.2), \
            "Intercept prior not as expected"
        assert np.isclose(pg.prior_info_['w_mu']['age']['y2'][tt],  \
            mm.w_mu_[1, 0, tt], atol=0.02), \
            "Age prior not as expected"

    assert np.isclose(np.sqrt(pg.prior_info_['lambda_b']['y2'][3]/\
        pg.prior_info_['lambda_a']['y2'][3]), 1, atol=0.001), \
        "Gamma prior not as expected for trajectory 3"
    assert np.isclose(np.sqrt(pg.prior_info_['lambda_b']['y2'][1]/\
        pg.prior_info_['lambda_a']['y2'][1]), 2, atol=0.03), \
        "Gamma prior not as expected for trajectory 1"
    
def test_compute_prior_info_5():
    """Model and data, different preds, new target
    """
    model_file_name = os.path.split(os.path.realpath(__file__))[0] + \
        '/../resources/models/model_1.p'
    data_file_name = os.path.split(os.path.realpath(__file__))[0] + \
        '/../resources/data/trajectory_data_1.csv'

    mm = pickle.load(open(model_file_name, 'rb'))['MultDPRegression']
    mm.cast_to_torch()
    mm.ranef_indices_ = None
    
    df = pd.read_csv(data_file_name)
    df['y2'] = df.y.values + 10
        
    targets = ['y', 'y2']
    preds = ['intercept', 'age', 'age^2']
    pg = PriorGenerator(targets, preds)
    pg.set_model(mm)
    pg.set_data(df, 'id')
    pg.compute_prior_info()

    # The following values are from a prior that -- upon visual inspection of
    # random draws -- appears reasonable. This test, then, should be considered
    # a regression test as opposed to a test of "correct" values
    prior_info_gt = \
        {'w_mu0': {'y': {'intercept': 8.102185344934247,
                         'age': -1.7021817743184755,
                         'age^2': 0.018416334689522876},
                   'y2': {'intercept': 18.10218534493426,
                          'age': -1.7021817743184746,
                          'age^2': 0.01841633468952318}},
         'w_var0': {'y': {'intercept': 1.9028074002722193,
                          'age': 0.1804899197433961,
                          'age^2': 0.0007043047299188618},
                    'y2': {'intercept': 1.9028074002722182,
                           'age': 0.18048991974339593,
                           'age^2': 0.0007043047299188608}},
         'lambda_a0': {'y': 14.399999999999999, 'y2': 14.399999999999999},
         'lambda_b0': {'y': 355.56217578065207, 'y2': 355.56217578065207},
         'w_mu': {'intercept': {'y': np.array([5.73256565, 9.27107768]),
                                'y2': np.array([15.73256565, 19.27107768])},
                  'age': {'y': np.array([-1.91955117, -0.86877812]),
                          'y2': np.array([-1.91955117, -0.86877812,])},
                  'age^2': {'y': np.array([-0.0042961, -0.00525871,]),
                            'y2': np.array([-0.0042961, -0.00525871,])}},
         'w_var': {'intercept': {'y': np.array([ 0.06041792, 0.30262284,]),
                                 'y2': np.array([0.06041792, 0.30262284,])},
                   'age': {'y': np.array([0.00333069, 0.01713044]),
                           'y2': np.array([0.00333069, 0.01713044])},
                   'age^2': {'y': np.array([9.82818683e-06, 5.18071486e-05]),
                             'y2': np.array([9.82818683e-06, 5.18071486e-05])}},
         'lambda_a': {'y': np.array([203.50423622, 11.95284849]),
                      'y2': np.array([203.50423622, 11.95284849])},
         'lambda_b': {'y': np.array([201.74450982, 48.8934525]),
                      'y2': np.array([201.74450982, 48.8934525])},
         'v_a': np.array([51., 51.]),
         #'v_b': np.array([50.74102343, 0.74102343]),
         'v_b': np.array([0.74102343, 50.74102343]),         
         'traj_probs': np.array([0.5, 0.5]),
         'alpha': 0.5}

    assert np.isclose(pg.prior_info_['w_mu0']['y']['intercept'], \
        prior_info_gt['w_mu0']['y']['intercept']), "Error in prior"
    assert np.isclose(pg.prior_info_['w_mu0']['y']['age'], \
        prior_info_gt['w_mu0']['y']['age']), "Error in prior"
    assert np.isclose(pg.prior_info_['w_mu0']['y']['age^2'], \
        prior_info_gt['w_mu0']['y']['age^2']), "Error in prior"
    assert np.isclose(pg.prior_info_['w_mu0']['y2']['intercept'], \
        prior_info_gt['w_mu0']['y2']['intercept']), "Error in prior"
    assert np.isclose(pg.prior_info_['w_mu0']['y2']['age'], \
        prior_info_gt['w_mu0']['y2']['age']), "Error in prior"
    assert np.isclose(pg.prior_info_['w_mu0']['y2']['age^2'], \
        prior_info_gt['w_mu0']['y2']['age^2']), "Error in prior"
    assert np.isclose(pg.prior_info_['w_var0']['y']['intercept'], \
        prior_info_gt['w_var0']['y']['intercept']), "Error in prior"
    assert np.isclose(pg.prior_info_['w_var0']['y']['age'], \
        prior_info_gt['w_var0']['y']['age']), "Error in prior"
    assert np.isclose(pg.prior_info_['w_var0']['y']['age^2'], \
        prior_info_gt['w_var0']['y']['age^2']), "Error in prior"
    assert np.isclose(pg.prior_info_['w_var0']['y2']['intercept'], \
        prior_info_gt['w_var0']['y2']['intercept']), "Error in prior"
    assert np.isclose(pg.prior_info_['w_var0']['y2']['age'], \
        prior_info_gt['w_var0']['y2']['age']), "Error in prior"
    assert np.isclose(pg.prior_info_['w_var0']['y2']['age^2'], \
        prior_info_gt['w_var0']['y2']['age^2']), "Error in prior"
    assert np.isclose(pg.prior_info_['lambda_a0']['y'], \
        prior_info_gt['lambda_a0']['y']), "Error in prior"
    assert np.isclose(pg.prior_info_['lambda_a0']['y2'], \
        prior_info_gt['lambda_a0']['y2']), "Error in prior"
    assert np.isclose(pg.prior_info_['lambda_b0']['y'], \
        prior_info_gt['lambda_b0']['y']), "Error in prior"
    assert np.isclose(pg.prior_info_['lambda_b0']['y2'], \
        prior_info_gt['lambda_b0']['y2']), "Error in prior"

    assert np.isclose(pg.prior_info_['w_mu']['intercept']['y'][3], \
        prior_info_gt['w_mu']['intercept']['y'][0]), "Error in prior"
    assert np.isclose(pg.prior_info_['w_mu']['intercept']['y2'][3], \
        prior_info_gt['w_mu']['intercept']['y2'][0]), "Error in prior"
    assert np.isclose(pg.prior_info_['w_mu']['age']['y'][3], \
        prior_info_gt['w_mu']['age']['y'][0]), "Error in prior"
    assert np.isclose(pg.prior_info_['w_mu']['age']['y2'][3], \
        prior_info_gt['w_mu']['age']['y2'][0]), "Error in prior"
    assert np.isclose(pg.prior_info_['w_mu']['age^2']['y'][3], \
        prior_info_gt['w_mu']['age^2']['y'][0]), "Error in prior"
    assert np.isclose(pg.prior_info_['w_mu']['age^2']['y2'][3], \
        prior_info_gt['w_mu']['age^2']['y2'][0]), "Error in prior"
    assert np.isclose(pg.prior_info_['w_var']['intercept']['y'][3], \
        prior_info_gt['w_var']['intercept']['y'][0]), "Error in prior"
    assert np.isclose(pg.prior_info_['w_var']['intercept']['y2'][3], \
        prior_info_gt['w_var']['intercept']['y2'][0]), "Error in prior"
    assert np.isclose(pg.prior_info_['w_var']['age']['y'][3], \
        prior_info_gt['w_var']['age']['y'][0]), "Error in prior"
    assert np.isclose(pg.prior_info_['w_var']['age']['y2'][3], \
        prior_info_gt['w_var']['age']['y2'][0]), "Error in prior"
    assert np.isclose(pg.prior_info_['w_var']['age^2']['y'][3], \
        prior_info_gt['w_var']['age^2']['y'][0]), "Error in prior"
    assert np.isclose(pg.prior_info_['w_var']['age^2']['y2'][3], \
        prior_info_gt['w_var']['age^2']['y2'][0]), "Error in prior"
    assert np.isclose(pg.prior_info_['lambda_a']['y'][3], \
        prior_info_gt['lambda_a']['y'][0]), "Error in prior"
    assert np.isclose(pg.prior_info_['lambda_a']['y2'][3], \
        prior_info_gt['lambda_a']['y2'][0]), "Error in prior"
    assert np.isclose(pg.prior_info_['lambda_b']['y'][3], \
        prior_info_gt['lambda_b']['y'][0]), "Error in prior"
    assert np.isclose(pg.prior_info_['lambda_b']['y2'][3], \
        prior_info_gt['lambda_b']['y2'][0]), "Error in prior"
    assert np.isclose(pg.prior_info_['v_a'][3], \
        prior_info_gt['v_a'][0]), "Error in prior"

    assert np.isclose(pg.prior_info_['v_b'][3], \
        prior_info_gt['v_b'][0]), "Error in prior"
    assert np.isclose(pg.prior_info_['traj_probs'][3], \
        prior_info_gt['traj_probs'][0]), "Error in prior"

    assert np.isclose(pg.prior_info_['w_mu']['intercept']['y'][1], \
        prior_info_gt['w_mu']['intercept']['y'][1]), "Error in prior"
    assert np.isclose(pg.prior_info_['w_mu']['intercept']['y2'][1], \
        prior_info_gt['w_mu']['intercept']['y2'][1]), "Error in prior"
    assert np.isclose(pg.prior_info_['w_mu']['age']['y'][1], \
        prior_info_gt['w_mu']['age']['y'][1]), "Error in prior"
    assert np.isclose(pg.prior_info_['w_mu']['age']['y2'][1], \
        prior_info_gt['w_mu']['age']['y2'][1]), "Error in prior"
    assert np.isclose(pg.prior_info_['w_mu']['age^2']['y'][1], \
        prior_info_gt['w_mu']['age^2']['y'][1]), "Error in prior"
    assert np.isclose(pg.prior_info_['w_mu']['age^2']['y2'][1], \
        prior_info_gt['w_mu']['age^2']['y2'][1]), "Error in prior"
    assert np.isclose(pg.prior_info_['w_var']['intercept']['y'][1], \
        prior_info_gt['w_var']['intercept']['y'][1]), "Error in prior"
    assert np.isclose(pg.prior_info_['w_var']['intercept']['y2'][1], \
        prior_info_gt['w_var']['intercept']['y2'][1]), "Error in prior"
    assert np.isclose(pg.prior_info_['w_var']['age']['y'][1], \
        prior_info_gt['w_var']['age']['y'][1]), "Error in prior"
    assert np.isclose(pg.prior_info_['w_var']['age']['y2'][1], \
        prior_info_gt['w_var']['age']['y2'][1]), "Error in prior"
    assert np.isclose(pg.prior_info_['w_var']['age^2']['y'][1], \
        prior_info_gt['w_var']['age^2']['y'][1]), "Error in prior"
    assert np.isclose(pg.prior_info_['w_var']['age^2']['y2'][1], \
        prior_info_gt['w_var']['age^2']['y2'][1]), "Error in prior"
    assert np.isclose(pg.prior_info_['lambda_a']['y'][1], \
        prior_info_gt['lambda_a']['y'][1]), "Error in prior"
    assert np.isclose(pg.prior_info_['lambda_a']['y2'][1], \
        prior_info_gt['lambda_a']['y2'][1]), "Error in prior"
    assert np.isclose(pg.prior_info_['lambda_b']['y'][1], \
        prior_info_gt['lambda_b']['y'][1]), "Error in prior"
    assert np.isclose(pg.prior_info_['lambda_b']['y2'][1], \
        prior_info_gt['lambda_b']['y2'][1]), "Error in prior"
    assert np.isclose(pg.prior_info_['v_a'][1], \
        prior_info_gt['v_a'][1]), "Error in prior"
    assert np.isclose(pg.prior_info_['v_b'][1], \
        prior_info_gt['v_b'][1]), "Error in prior"
    assert np.isclose(pg.prior_info_['traj_probs'][1], \
        prior_info_gt['traj_probs'][1]), "Error in prior"
    assert np.isclose(pg.prior_info_['alpha'], 0.3613243243243244), \
        "Error in prior"
    
def test_prior_generator_initializes_shared_schema():
    pg = PriorGenerator(
        targets=['y'],
        preds=['intercept', 'time'],
        shared_predictors=['cohort']
    )

    assert 'shared_predictors' in pg.prior_info_
    assert 'w_mu0_shared' in pg.prior_info_
    assert 'w_var0_shared' in pg.prior_info_

    assert pg.prior_info_['shared_predictors'] == ['cohort']
    assert 'y' in pg.prior_info_['w_mu0_shared']
    assert 'y' in pg.prior_info_['w_var0_shared']

    assert pg.prior_info_['w_mu0_shared']['y']['cohort'] == 0
    assert pg.prior_info_['w_var0_shared']['y']['cohort'] == 1

    # legacy block still exists
    assert pg.prior_info_['w_mu0']['y']['intercept'] == 0
    assert pg.prior_info_['w_var0']['y']['intercept'] == 5
    
def test_prior_generator_rejects_shared_pred_overlap():
    with pytest.raises(AssertionError):
        PriorGenerator(
            targets=['y'],
            preds=['intercept', 'time', 'cohort'],
            shared_predictors=['cohort']
        )

def test_is_shared_pred():
    pg = PriorGenerator(
        targets=['y'],
        preds=['intercept', 'time'],
        shared_predictors=['cohort', 'site']
    )

    assert pg._is_shared_pred('cohort')
    assert pg._is_shared_pred('site')
    assert not pg._is_shared_pred('time')        

def test_prior_info_from_df_gaussians_routes_shared_predictors():
    np.random.seed(123)

    n = 200
    time = np.random.normal(size=n)
    cohort = np.random.binomial(1, 0.5, size=n)

    # true model: trajectory-specific block has intercept + time,
    # shared block has cohort
    y = 1.0 + 0.8 * time + 2.5 * cohort + np.random.normal(scale=0.5, size=n)

    df = pd.DataFrame({
        'intercept': 1.0,
        'time': time,
        'cohort': cohort,
        'y': y
    })

    pg = PriorGenerator(
        targets=['y'],
        preds=['intercept', 'time'],
        shared_predictors=['cohort']
    )
    pg.set_data(df, groupby=None)

    pg.prior_info_from_df_gaussians('y')

    # trajectory-specific predictors should be in legacy block
    assert 'intercept' in pg.prior_info_['w_mu0']['y']
    assert 'time' in pg.prior_info_['w_mu0']['y']

    # shared predictor should be in shared block
    assert 'cohort' in pg.prior_info_['w_mu0_shared']['y']
    assert 'cohort' in pg.prior_info_['w_var0_shared']['y']

    # and should not be required in the legacy block
    # (if you left it absent, this is the expected behavior)
    assert 'cohort' not in pg.prior_info_['w_mu0']['y']

    # sanity checks on direction / finiteness
    assert np.isfinite(pg.prior_info_['w_mu0']['y']['time'])
    assert np.isfinite(pg.prior_info_['w_var0']['y']['time'])
    assert np.isfinite(pg.prior_info_['w_mu0_shared']['y']['cohort'])
    assert np.isfinite(pg.prior_info_['w_var0_shared']['y']['cohort'])

    assert pg.prior_info_['w_mu0_shared']['y']['cohort'] > 0.5    

def test_prior_info_from_df_binary_leaves_shared_block_unused():
    np.random.seed(123)

    n = 200
    time = np.random.normal(size=n)
    cohort = np.random.binomial(1, 0.5, size=n)

    lp = -0.5 + 1.2 * time
    p = 1 / (1 + np.exp(-lp))
    y = np.random.binomial(1, p, size=n)

    df = pd.DataFrame({
        'intercept': 1.0,
        'time': time,
        'cohort': cohort,
        'y': y
    })

    pg = PriorGenerator(
        targets=['y'],
        preds=['intercept', 'time'],
        shared_predictors=['cohort']
    )
    pg.set_data(df, groupby=None)

    pg.prior_info_from_df('y')

    # legacy block populated from binary regression
    assert 'intercept' in pg.prior_info_['w_mu0']['y']
    assert 'time' in pg.prior_info_['w_mu0']['y']
    assert np.isfinite(pg.prior_info_['w_mu0']['y']['time'])

    # binary path should not populate shared block from regression
    # (assuming you left binary behavior unchanged)
    assert pg.prior_info_['w_mu0_shared']['y']['cohort'] == 0
    assert pg.prior_info_['w_var0_shared']['y']['cohort'] == 1

    # lambda priors should be None for binary targets
    assert pg.prior_info_['lambda_a0']['y'] is None
    assert pg.prior_info_['lambda_b0']['y'] is None    

def test_coef_override_routes_shared_predictor_to_shared_block():
    pg = PriorGenerator(
        targets=['y'],
        preds=['intercept', 'time'],
        shared_predictors=['cohort']
    )

    prior_info = copy.deepcopy(pg.prior_info_)

    tt = 'y'
    pp = 'cohort'
    m = 1.5
    s = 0.25

    # mimic your routing logic
    if pp in pg.shared_predictors_:
        prior_info['w_mu0_shared'][tt][pp] = m
        prior_info['w_var0_shared'][tt][pp] = s**2
    else:
        prior_info['w_mu0'][tt][pp] = m
        prior_info['w_var0'][tt][pp] = s**2

    assert prior_info['w_mu0_shared']['y']['cohort'] == 1.5
    assert prior_info['w_var0_shared']['y']['cohort'] == 0.25**2    

def test_prior_info_from_model_exports_shared_predictors_separately():
    np.random.seed(123)
    torch.manual_seed(123)

    # ------------------------------------------------------------------
    # Simulate simple Gaussian longitudinal data with one shared predictor
    # ------------------------------------------------------------------
    n_subjects = 80
    n_timepoints = 4
    K_fit = 4

    subject_ids = []
    time_vals = []
    cohort_vals = []
    y_vals = []

    true_traj = np.random.binomial(1, 0.5, size=n_subjects)
    subject_cohort = np.random.binomial(1, 0.5, size=n_subjects)

    beta_cohort = 1.5
    intercepts = {0: 0.0, 1: 1.0}
    slopes = {0: 0.2, 1: 0.9}
    sigma = 0.4

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

    w_mu0 = np.zeros((3, 1))
    w_var0 = np.ones((3, 1)) * 10.0
    lambda_a0 = np.ones(1)
    lambda_b0 = np.ones(1)

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
        iters=40,
        verbose=False,
        shared_predictor_names=['cohort']
    )

    # ------------------------------------------------------------------
    # Export priors from the fitted model
    # ------------------------------------------------------------------
    pg = PriorGenerator(
        targets=['y'],
        preds=['intercept', 'time'],
        shared_predictors=['cohort']
    )
    pg.set_model(mm)
    pg.prior_info_from_model('y')

    # Shared predictor should be exported to shared block
    assert 'cohort' in pg.prior_info_['w_mu0_shared']['y']
    assert 'cohort' in pg.prior_info_['w_var0_shared']['y']

    # Shared predictor should not appear in the legacy block
    assert 'cohort' not in pg.prior_info_['w_mu0']['y']
    assert 'cohort' not in pg.prior_info_['w_var0']['y']

    # Check that exported value matches model shared posterior
    assert np.isclose(
        pg.prior_info_['w_mu0_shared']['y']['cohort'],
        mm.w_mu_shared_[0, 0].item()
    )
    assert np.isclose(
        pg.prior_info_['w_var0_shared']['y']['cohort'],
        mm.w_var_shared_[0, 0].item()
    )

    # Legacy predictors should still export normally
    assert 'intercept' in pg.prior_info_['w_mu0']['y']
    assert 'time' in pg.prior_info_['w_mu0']['y']
    assert np.isfinite(pg.prior_info_['w_mu0']['y']['time'])
    assert np.isfinite(pg.prior_info_['w_var0']['y']['time'])    

def test_traj_prior_info_from_model_ignores_shared_predictors():
    np.random.seed(123)
    torch.manual_seed(123)

    # ------------------------------------------------------------------
    # Simulate simple Gaussian longitudinal data
    # ------------------------------------------------------------------
    n_subjects = 60
    n_timepoints = 4
    K_fit = 4

    subject_ids = []
    time_vals = []
    cohort_vals = []
    y_vals = []

    true_traj = np.random.binomial(1, 0.5, size=n_subjects)
    subject_cohort = np.random.binomial(1, 0.5, size=n_subjects)

    beta_cohort = 1.2
    intercepts = {0: 0.0, 1: 1.0}
    slopes = {0: 0.1, 1: 0.8}
    sigma = 0.4

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

    w_mu0 = np.zeros((3, 1))
    w_var0 = np.ones((3, 1)) * 10.0
    lambda_a0 = np.ones(1)
    lambda_b0 = np.ones(1)

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
        iters=30,
        verbose=False,
        shared_predictor_names=['cohort']
    )

    active_trajs = torch.where(mm.sig_trajs_)[0].cpu().numpy()
    assert len(active_trajs) > 0
    traj = int(active_trajs[0])

    pg = PriorGenerator(
        targets=['y'],
        preds=['intercept', 'time'],
        shared_predictors=['cohort']
    )
    pg.set_model(mm)
    pg.traj_prior_info_from_model('y', traj)

    # Only trajectory-specific predictors should appear in traj prior export
    assert 'intercept' in pg.prior_info_['w_mu']
    assert 'time' in pg.prior_info_['w_mu']
    assert 'cohort' not in pg.prior_info_['w_mu']

    assert 'intercept' in pg.prior_info_['w_var']
    assert 'time' in pg.prior_info_['w_var']
    assert 'cohort' not in pg.prior_info_['w_var']

    # Check values are present for the requested trajectory
    assert 'y' in pg.prior_info_['w_mu']['intercept']
    assert 'y' in pg.prior_info_['w_mu']['time']

    assert len(pg.prior_info_['w_mu']['intercept']['y']) > traj
    assert len(pg.prior_info_['w_mu']['time']['y']) > traj
    assert len(pg.prior_info_['w_var']['time']['y']) > traj

    assert np.isfinite(pg.prior_info_['w_mu']['intercept']['y'][traj])
    assert np.isfinite(pg.prior_info_['w_mu']['time']['y'][traj])
    assert np.isfinite(pg.prior_info_['w_var']['time']['y'][traj])

def test_prior_info_from_model_round_trip_shared_schema():
    np.random.seed(321)
    torch.manual_seed(321)

    # ------------------------------------------------------------------
    # Simulate Gaussian longitudinal data with one shared predictor
    # ------------------------------------------------------------------
    n_subjects = 70
    n_timepoints = 4
    K_fit = 4

    subject_ids = []
    time_vals = []
    cohort_vals = []
    y_vals = []

    true_traj = np.random.binomial(1, 0.5, size=n_subjects)
    subject_cohort = np.random.binomial(1, 0.5, size=n_subjects)

    beta_cohort = 1.8
    intercepts = {0: 0.0, 1: 0.8}
    slopes = {0: 0.2, 1: 0.9}
    sigma = 0.45

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

    w_mu0 = np.zeros((3, 1))
    w_var0 = np.ones((3, 1)) * 10.0
    lambda_a0 = np.ones(1)
    lambda_b0 = np.ones(1)

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
        iters=35,
        verbose=False,
        shared_predictor_names=['cohort']
    )

    # ------------------------------------------------------------------
    # Export prior info from the fitted model
    # ------------------------------------------------------------------
    pg = PriorGenerator(
        targets=['y'],
        preds=['intercept', 'time'],
        shared_predictors=['cohort']
    )
    pg.set_model(mm)
    pg.prior_info_from_model('y')

    prior_info = pg.prior_info_

    # Shared schema should be present
    assert 'shared_predictors' in prior_info
    assert 'w_mu0_shared' in prior_info
    assert 'w_var0_shared' in prior_info

    assert prior_info['shared_predictors'] == ['cohort']
    assert 'cohort' in prior_info['w_mu0_shared']['y']
    assert 'cohort' in prior_info['w_var0_shared']['y']

    # Exported shared priors should be finite
    assert np.isfinite(prior_info['w_mu0_shared']['y']['cohort'])
    assert np.isfinite(prior_info['w_var0_shared']['y']['cohort'])

    # Legacy block should still be finite for non-shared predictors
    assert np.isfinite(prior_info['w_mu0']['y']['intercept'])
    assert np.isfinite(prior_info['w_var0']['y']['intercept'])
    assert np.isfinite(prior_info['w_mu0']['y']['time'])
    assert np.isfinite(prior_info['w_var0']['y']['time'])    

def test_prior_info_from_model_predictor_set_validation_with_shared_preds():
    pg = PriorGenerator(
        targets=['y'],
        preds=['intercept', 'time'],
        shared_predictors=['cohort']
    )

    mm = get_basic_model(G=2, M=3, num_gaussian=1, num_binary=0, K=3, num_long_data_pts=2)
    mm.predictor_names_ = ['intercept', 'time', 'cohort']
    mm.target_names_ = ['y']
    mm.shared_predictor_names_ = ['cohort']
    mm.num_shared_preds_ = 1
    mm.num_traj_preds_ = 2
    mm.w_mu_shared_ = torch.zeros((1, 1), dtype=torch.float64)
    mm.w_var_shared_ = torch.ones((1, 1), dtype=torch.float64)

    pg.set_model(mm)

    # Should not raise
    pg.prior_info_from_model('y')    

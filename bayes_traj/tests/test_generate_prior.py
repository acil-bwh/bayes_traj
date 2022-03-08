from bayes_traj.generate_prior import PriorGenerator
import numpy as np
import pandas as pd
from bayes_traj.utils import *
import pdb, os, pickle


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

    assert np.isclose(pg.prior_info_['w_mu0']['y']['intercept'], 7.789066077), \
        "w_mu0 not as expected"

    assert np.isclose(pg.prior_info_['w_mu0']['y']['age'], -1.47880949563561), \
        "w_mu0 not as expected"

    assert np.isclose(pg.prior_info_['w_var0']['y']['intercept'], 0.50386898), \
        "w_var0 not as expected"
    
    assert np.isclose(pg.prior_info_['w_var0']['y']['age'], 0.04297104101760), \
        "w_var0 not as expected"

    
    pickle.dump(pg.prior_info_, open('/Users/jr555/tmp/foo_prior.p', 'wb'))
    pdb.set_trace()
    
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
    assert np.isclose(pg.prior_info_['lambda_a0']['y'], 16.0), \
        "lambda_a0 not as expected"
    assert np.isclose(pg.prior_info_['lambda_b0']['y'], 396.1353893428358), \
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
    
    targets = ['y']
    preds = ['intercept', 'age']
    pg = PriorGenerator(targets, preds)
    pg.set_model(mm)
    pg.set_data(df, 'id')

    #---------------------------------------------------------------------------
    # Evaluate prior for trajectory 1
    #---------------------------------------------------------------------------
    intercept_co = 10
    age_co = -1
    resid_std = 2

    traj = 1
    pg.traj_prior_info_from_df('y', traj)

    mu = pg.prior_info_['w_mu']['intercept']['y'][traj]
    sig = np.sqrt(pg.prior_info_['w_var']['intercept']['y'][traj])
    assert mu - 2*sig <= intercept_co and mu + 2*sig >= intercept_co, \
        "Intercept prior not as expected for trajectory 1"

    mu = pg.prior_info_['w_mu']['age']['y'][traj]
    sig = np.sqrt(pg.prior_info_['w_var']['age']['y'][traj])
    assert mu - 2*sig <= age_co and mu + 2*sig >= age_co, \
        "Age prior not as expected for trajectory 1"

    prec_mean = pg.prior_info_['lambda_a']['y'][traj]/\
        pg.prior_info_['lambda_b']['y'][traj]
    prec_var = pg.prior_info_['lambda_a']['y'][traj]/\
        (pg.prior_info_['lambda_b']['y'][1]**2)
    prec_high = prec_mean + 2*np.sqrt(prec_var)
    prec_low= prec_mean - 2*np.sqrt(prec_var)

    std_high = np.sqrt(1/prec_low)
    std_low = np.sqrt(1/prec_high)

    assert resid_std >= std_low and resid_std <= std_high, \
        "Prior over residual precision not as expected for trajectory 1"

    #---------------------------------------------------------------------------
    # Evaluate prior for trajectory 2
    #---------------------------------------------------------------------------
    intercept_co = 6
    age_co = -2
    resid_std = 1

    traj = 2
    pg.traj_prior_info_from_df('y', traj)

    mu = pg.prior_info_['w_mu']['intercept']['y'][traj]
    sig = np.sqrt(pg.prior_info_['w_var']['intercept']['y'][traj])
    assert mu - 2*sig <= intercept_co and mu + 2*sig >= intercept_co, \
        "Intercept prior not as expected for trajectory 2"

    mu = pg.prior_info_['w_mu']['age']['y'][traj]
    sig = np.sqrt(pg.prior_info_['w_var']['age']['y'][traj])
    assert mu - 2*sig <= age_co and mu + 2*sig >= age_co, \
        "Age prior not as expected for trajectory 2"

    prec_mean = pg.prior_info_['lambda_a']['y'][traj]/\
        pg.prior_info_['lambda_b']['y'][traj]
    prec_var = pg.prior_info_['lambda_a']['y'][traj]/\
        (pg.prior_info_['lambda_b']['y'][1]**2)
    prec_high = prec_mean + 2*np.sqrt(prec_var)
    prec_low= prec_mean - 2*np.sqrt(prec_var)

    std_high = np.sqrt(1/prec_low)
    std_low = np.sqrt(1/prec_high)

    assert resid_std >= std_low and resid_std <= std_high, \
        "Prior over residual precision not as expected for trajectory 2"


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

    traj = 1
    for tt in np.where(mm.sig_trajs_)[0]:
        pg.traj_prior_info_from_model('y', tt)

    assert pg.prior_info_['w_mu']['intercept']['y'][1] == mm.w_mu_[0, 0, 1], \
        "Intercept prior not as expected for trajectory 1"
    assert pg.prior_info_['w_mu']['intercept']['y'][2] == mm.w_mu_[0, 0, 2], \
        "Intercept prior not as expected for trajectory 2"

    assert pg.prior_info_['w_mu']['age']['y'][1] == mm.w_mu_[1, 0, 1], \
        "Age prior not as expected for trajectory 1"
    assert pg.prior_info_['w_mu']['age']['y'][2] == mm.w_mu_[1, 0, 2], \
        "Age prior not as expected for trajectory 2"

    assert pg.prior_info_['lambda_a']['y'][1] == mm.lambda_a_[0, 1], \
        "Gamma prior not as expected for trajectory 1"
    assert pg.prior_info_['lambda_b']['y'][1] == mm.lambda_b_[0, 1], \
        "Gamma prior not as expected for trajectory 1"

    assert pg.prior_info_['lambda_a']['y'][2] == mm.lambda_a_[0, 2], \
        "Gamma prior not as expected for trajectory 2"
    assert pg.prior_info_['lambda_b']['y'][2] == mm.lambda_b_[0, 2], \
        "Gamma prior not as expected for trajectory 2" 


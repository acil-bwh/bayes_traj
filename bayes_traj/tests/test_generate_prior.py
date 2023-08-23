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

    assert np.isclose(pg.prior_info_['w_mu0']['y']['intercept'], 7.810441593), \
        "w_mu0 not as expected"
    assert np.isclose(pg.prior_info_['w_mu0']['y']['age'], -1.48068743758711), \
        "w_mu0 not as expected"
    assert np.isclose(pg.prior_info_['w_var0']['y']['intercept'], 0.51800801), \
        "w_var0 not as expected"    
    assert np.isclose(pg.prior_info_['w_var0']['y']['age'], 0.04260915525159), \
        "w_var0 not as expected"
    assert np.isclose(pg.prior_info_['lambda_a0']['y'], 207.87669370617036), \
        "lambda_a0 not as expected"
    assert np.isclose(pg.prior_info_['lambda_b0']['y'], 67.86146951205156), \
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

    traj = 5
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
        "Prior over residual precision not as expected for trajectory 5"

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

    for tt in [3, 5]:
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
    
    targets = ['y']
    preds = ['intercept', 'age']
    pg = PriorGenerator(targets, preds)
    pg.set_model(mm)    
    pg.set_data(df, 'id')
    pg.compute_prior_info()

    assert np.isclose(pg.prior_info_['w_mu0']['y']['intercept'], 7.810441593), \
        "w_mu0 not as expected"
    assert np.isclose(pg.prior_info_['w_mu0']['y']['age'], -1.48068743758711), \
        "w_mu0 not as expected"
    assert np.isclose(pg.prior_info_['w_var0']['y']['intercept'], 0.51800801), \
        "w_var0 not as expected"    
    assert np.isclose(pg.prior_info_['w_var0']['y']['age'], 0.04260915525159), \
        "w_var0 not as expected"
    assert np.isclose(pg.prior_info_['lambda_a0']['y'], 207.87669370617036), \
        "lambda_a0 not as expected"
    assert np.isclose(pg.prior_info_['lambda_b0']['y'], 67.86146951205156), \
        "lambda_b0 not as expected"

    for tt in [3, 5]:
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
    df = pd.read_csv(data_file_name)
        
    targets = ['y']
    preds = ['intercept', 'age']
    pg = PriorGenerator(targets, preds)
    pg.set_model(mm)
    pg.set_data(df, 'id')
    pg.compute_prior_info()

    assert np.isclose(pg.prior_info_['w_mu0']['y']['intercept'], 7.810441593), \
        "w_mu0 not as expected"
    assert np.isclose(pg.prior_info_['w_mu0']['y']['age'], -1.48068743758711), \
        "w_mu0 not as expected"
    assert np.isclose(pg.prior_info_['w_var0']['y']['intercept'], 0.51800801), \
        "w_var0 not as expected"    
    assert np.isclose(pg.prior_info_['w_var0']['y']['age'], 0.04260915525159), \
        "w_var0 not as expected"
    assert np.isclose(pg.prior_info_['lambda_a0']['y'], 207.87669370617036), \
        "lambda_a0 not as expected"
    assert np.isclose(pg.prior_info_['lambda_b0']['y'], 67.86146951205156), \
        "lambda_b0 not as expected"

    for tt in [3, 5]:
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
    df = pd.read_csv(data_file_name)
    df['y2'] = df.y.values + 10
        
    targets = ['y', 'y2']
    preds = ['intercept', 'age']
    pg = PriorGenerator(targets, preds)
    pg.set_model(mm)
    pg.set_data(df, 'id')
    pg.compute_prior_info()

    assert np.isclose(pg.prior_info_['w_mu0']['y']['intercept'], 7.810441593), \
        "w_mu0 not as expected"
    assert np.isclose(pg.prior_info_['w_mu0']['y']['age'], -1.48068743758711), \
        "w_mu0 not as expected"
    assert np.isclose(pg.prior_info_['w_var0']['y']['intercept'], 0.51800801), \
        "w_var0 not as expected"    
    assert np.isclose(pg.prior_info_['w_var0']['y']['age'], 0.04260915525159), \
        "w_var0 not as expected"
    assert np.isclose(pg.prior_info_['lambda_a0']['y'], 207.87669370617036), \
        "lambda_a0 not as expected"
    assert np.isclose(pg.prior_info_['lambda_b0']['y'], 67.86146951205156), \
        "lambda_b0 not as expected"

    for tt in [3, 5]:
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

    for tt in [3, 5]:
        assert np.isclose(mm.w_mu_[0, 0, tt] + 10, \
            pg.prior_info_['w_mu']['intercept']['y2'][tt], atol=0.05), \
            "Intercept prior not as expected"
        assert np.isclose(pg.prior_info_['w_mu']['age']['y2'][tt],  \
            mm.w_mu_[1, 0, tt], atol=0.01), \
            "Age prior not as expected"

    assert np.isclose(np.sqrt(pg.prior_info_['lambda_b']['y2'][3]/\
        pg.prior_info_['lambda_a']['y2'][3]), 1, atol=0.001), \
        "Gamma prior not as expected for trajectory 3"
    assert np.isclose(np.sqrt(pg.prior_info_['lambda_b']['y2'][5]/\
        pg.prior_info_['lambda_a']['y2'][5]), 2, atol=0.03), \
        "Gamma prior not as expected for trajectory 5"
    
def test_compute_prior_info_5():
    """Model and data, different preds, new target
    """
    model_file_name = os.path.split(os.path.realpath(__file__))[0] + \
        '/../resources/models/model_1.p'
    data_file_name = os.path.split(os.path.realpath(__file__))[0] + \
        '/../resources/data/trajectory_data_1.csv'

    mm = pickle.load(open(model_file_name, 'rb'))['MultDPRegression']
    mm.cast_to_torch()
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
         'v_b': np.array([50.74102343, 0.74102343]),
         'traj_probs': np.array([0.5, 0.5]),
         'alpha': 0.5}

    assert pg.prior_info_['w_mu0']['y']['intercept'] == \
        prior_info_gt['w_mu0']['y']['intercept'], "Error in prior"
    assert pg.prior_info_['w_mu0']['y']['age'] == \
        prior_info_gt['w_mu0']['y']['age'], "Error in prior"
    assert pg.prior_info_['w_mu0']['y']['age^2'] == \
        prior_info_gt['w_mu0']['y']['age^2'], "Error in prior"
    assert pg.prior_info_['w_mu0']['y2']['intercept'] == \
        prior_info_gt['w_mu0']['y2']['intercept'], "Error in prior"
    assert pg.prior_info_['w_mu0']['y2']['age'] == \
        prior_info_gt['w_mu0']['y2']['age'], "Error in prior"
    assert pg.prior_info_['w_mu0']['y2']['age^2'] == \
        prior_info_gt['w_mu0']['y2']['age^2'], "Error in prior"
    assert pg.prior_info_['w_var0']['y']['intercept'] == \
        prior_info_gt['w_var0']['y']['intercept'], "Error in prior"
    assert pg.prior_info_['w_var0']['y']['age'] == \
        prior_info_gt['w_var0']['y']['age'], "Error in prior"
    assert pg.prior_info_['w_var0']['y']['age^2'] == \
        prior_info_gt['w_var0']['y']['age^2'], "Error in prior"
    assert pg.prior_info_['w_var0']['y2']['intercept'] == \
        prior_info_gt['w_var0']['y2']['intercept'], "Error in prior"
    assert pg.prior_info_['w_var0']['y2']['age'] == \
        prior_info_gt['w_var0']['y2']['age'], "Error in prior"
    assert pg.prior_info_['w_var0']['y2']['age^2'] == \
        prior_info_gt['w_var0']['y2']['age^2'], "Error in prior"
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

    assert np.isclose(pg.prior_info_['w_mu']['intercept']['y'][5], \
        prior_info_gt['w_mu']['intercept']['y'][1]), "Error in prior"
    assert np.isclose(pg.prior_info_['w_mu']['intercept']['y2'][5], \
        prior_info_gt['w_mu']['intercept']['y2'][1]), "Error in prior"
    assert np.isclose(pg.prior_info_['w_mu']['age']['y'][5], \
        prior_info_gt['w_mu']['age']['y'][1]), "Error in prior"
    assert np.isclose(pg.prior_info_['w_mu']['age']['y2'][5], \
        prior_info_gt['w_mu']['age']['y2'][1]), "Error in prior"
    assert np.isclose(pg.prior_info_['w_mu']['age^2']['y'][5], \
        prior_info_gt['w_mu']['age^2']['y'][1]), "Error in prior"
    assert np.isclose(pg.prior_info_['w_mu']['age^2']['y2'][5], \
        prior_info_gt['w_mu']['age^2']['y2'][1]), "Error in prior"
    assert np.isclose(pg.prior_info_['w_var']['intercept']['y'][5], \
        prior_info_gt['w_var']['intercept']['y'][1]), "Error in prior"
    assert np.isclose(pg.prior_info_['w_var']['intercept']['y2'][5], \
        prior_info_gt['w_var']['intercept']['y2'][1]), "Error in prior"
    assert np.isclose(pg.prior_info_['w_var']['age']['y'][5], \
        prior_info_gt['w_var']['age']['y'][1]), "Error in prior"
    assert np.isclose(pg.prior_info_['w_var']['age']['y2'][5], \
        prior_info_gt['w_var']['age']['y2'][1]), "Error in prior"
    assert np.isclose(pg.prior_info_['w_var']['age^2']['y'][5], \
        prior_info_gt['w_var']['age^2']['y'][1]), "Error in prior"
    assert np.isclose(pg.prior_info_['w_var']['age^2']['y2'][5], \
        prior_info_gt['w_var']['age^2']['y2'][1]), "Error in prior"
    assert np.isclose(pg.prior_info_['lambda_a']['y'][5], \
        prior_info_gt['lambda_a']['y'][1]), "Error in prior"
    assert np.isclose(pg.prior_info_['lambda_a']['y2'][5], \
        prior_info_gt['lambda_a']['y2'][1]), "Error in prior"
    assert np.isclose(pg.prior_info_['lambda_b']['y'][5], \
        prior_info_gt['lambda_b']['y'][1]), "Error in prior"
    assert np.isclose(pg.prior_info_['lambda_b']['y2'][5], \
        prior_info_gt['lambda_b']['y2'][1]), "Error in prior"
    assert np.isclose(pg.prior_info_['v_a'][5], \
        prior_info_gt['v_a'][1]), "Error in prior"
    assert np.isclose(pg.prior_info_['v_b'][5], \
        prior_info_gt['v_b'][1]), "Error in prior"
    assert np.isclose(pg.prior_info_['traj_probs'][5], \
        prior_info_gt['traj_probs'][1]), "Error in prior"
    
    assert pg.prior_info_['alpha'] == prior_info_gt['alpha'], "Error in prior"
    

    

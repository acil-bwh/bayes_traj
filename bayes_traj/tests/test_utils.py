from bayes_traj.utils import *
import pickle, torch
import numpy as np
import os
import pdb
import bayes_traj
import pandas as pd
import warnings

def test_load_model_pickle():
    model_file_name = os.path.split(os.path.realpath(__file__))[0] + \
        '/../resources/models/model_1.p'
        
    model = load_model(model_file_name)
    
    assert isinstance(model, bayes_traj.mult_dp_regression.MultDPRegression), \
        "Model not read correctly"


def test_load_model_torch():
    # TODO
    pass


def test_augment_df_with_traj_info_1():
    """
    """
    warnings.filterwarnings("ignore", category=DeprecationWarning,
                            module="pandas")
    # Read df
    data_file_name = os.path.split(os.path.realpath(__file__))[0] + \
        '/../resources/data/trajectory_data_1.csv'

    # Modify the input data a bit before testing the assignment
    df = pd.read_csv(data_file_name)
    df['y'] = df['y'].values + 0.01*np.random.randn(1)
    df[:] = df.sample(frac=1).values
    if 'traj' in df.columns:
        df.rename(columns={'traj': 'traj_gt'}, inplace=True)

    # Read MultDPRegression model
    model_file_name = os.path.split(os.path.realpath(__file__))[0] + \
        '/../resources/models/model_1.p'
    model = load_model(model_file_name)

    # Augment df and evaluate
    df_aug = augment_df_with_traj_info(df, model, traj_map=None)

    assert np.sum((df_aug.traj.values == 1) & (df.traj_gt.values == 1)) + \
        np.sum((df_aug.traj.values == 3) & (df.traj_gt.values == 2)), \
        "Error in trajectory assignment"

def test_augment_df_with_traj_info_2():
    """
    """
    warnings.filterwarnings("ignore", category=DeprecationWarning,
                            module="pandas")
    # Read df
    data_file_name = os.path.split(os.path.realpath(__file__))[0] + \
        '/../resources/data/trajectory_data_1.csv'

    # Modify the input data a bit before testing the assignment
    df = pd.read_csv(data_file_name)
    df['y'] = df['y'].values + 0.01*np.random.randn(1)
    df[:] = df.sample(frac=1).values
    if 'traj' in df.columns:
        df.rename(columns={'traj': 'traj_gt'}, inplace=True)
    
    # Read MultPyro model
    model_file_name = os.path.split(os.path.realpath(__file__))[0] + \
        '/../resources/models/pyro_model_1.pt'
    model = load_model(model_file_name)
    foo = model.classify(df)
    
    # Augment df and evaluate
    
def test_sample_cos():
    """
    """
    M = 2
    D = 1
    w_mu0 = np.atleast_2d(np.array([3, 7])).T
    w_var0 = np.atleast_2d(np.array([.1, 2])).T

    w = sample_cos(w_mu0, w_var0, num_samples=100000)

    assert np.sum(np.isclose(w_mu0, np.mean(w, 2), atol=.01)) == 2, \
        "Unexpected mean"
    assert np.sum(np.isclose(w_var0, np.var(w, 2), atol=.1)) == 2, \
        "Unexpected variance"

def test_sample_precs():
    lambda_a0 = np.array([1, 5])
    lambda_b0 = np.array([2, 13])
    precs = sample_precs(lambda_a0, lambda_b0, num_samples=100000)

    assert np.sum(np.isclose(lambda_a0/lambda_b0, \
                             np.mean(precs, 1), atol=.01)) == 2, \
        "Unexpected precision"

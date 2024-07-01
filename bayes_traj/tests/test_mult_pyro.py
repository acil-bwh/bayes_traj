from typing import Optional

import pdb
import pytest
import torch
import numpy as np
import pandas as pd
import warnings, os
from bayes_traj.load_model import load_model

from bayes_traj.mult_pyro import MultPyro

torch.set_default_dtype(torch.double)

def create_dummy_df(D, B, M, G, T, C):
    """
    Create a dummy DataFrame with D real target columns, B boolean target 
    # columns, M predictor columns, and a cohort column, for G individuals 
    # each with T time points.

    Args:
        D (int): Number of real-valued target columns.
        B (int): Number of boolean target columns.
        M (int): Number of predictor columns.
        G (int): Number of individuals.
        T (int): Number of time points per individual.
        C (int): Number of cohorts.

    Returns:
        pd.DataFrame: A DataFrame with the generated data.
    """
    assert C <= G, "Cannot have more cohorts than individuals"
    
    # Create a time index and individual ID for each time point
    time = np.tile(np.arange(T), G)
    individual = np.repeat(np.arange(G), T)

    # Create D columns of real values for targets drawn from a
    # normal distribution
    real_data = np.random.randn(T * G, D)

    # Create B columns of boolean values for targets drawn from a
    # Bernoulli distribution
    #bool_data = np.random.rand(T * G, B) < 0.5
    bool_data = \
        np.random.binomial(n=1, p=0.5, size=[T*G, B]).astype(float)
    
    # Create M columns of real values for predictors drawn from a
    # normal distribution
    predictors_data = np.random.randn(T * G, M)

    # Create cohort assignments
    cohort = np.random.randint(0, C, G)
    cohort[0:C] = range(0, C) # Ensures all cohorts are represented
    # Repeat cohort for each time point of an individual
    cohort_column = np.repeat(cohort, T)  

    # Combine into a single DataFrame
    df = pd.DataFrame(data={
        'time': time,
        'id': individual,
        'cohort': cohort_column
    })

    # Add real-valued target columns
    for i in range(D):
        df[f'real_target_{i}'] = real_data[:, i]

    # Add boolean target columns
    for i in range(B):
        df[f'bool_target_{i}'] = bool_data[:, i]

    # Add real-valued predictor columns
    for i in range(M):
        df[f'predictor_{i}'] = predictors_data[:, i]

    return df


@pytest.mark.parametrize("y_mask_dim", [2, 3])
@pytest.mark.parametrize("x_mask", [True, False])
@pytest.mark.parametrize("rand_eff", [False, True])
@pytest.mark.parametrize(
    "K, D, B, M, T, C, G, G_",
    [
        (1, 1, 0, 1, 1, 1, 1, 1),  # Real data only.
        (1, 0, 1, 1, 1, 1, 1, 1),  # Boolean data only.
        # The following use all distinct sizes to detect shape errors.
        (2, 3, 4, 5, 6, 1, 7, 8),  # Single cohort.
        (2, 3, 4, 5, 6, 7, 8, 9),  # Multi-cohort.
    ],
)
def test_smoke(K, D, B, M, T, C, G, G_, x_mask: bool, y_mask_dim: int,
               rand_eff: bool):

    #TODO: remove after testing:
    rand_eff = False
    
    # Set hyperparameters.
    alpha0 = torch.randn(K).exp()  # Ensure positive.
    w_mu0 = torch.randn(D + B, M)
    w_var0 = torch.randn(D + B, M).exp()  # Ensure positive.
    lambda_a0 = torch.randn(D).exp()  # Ensure positive.
    lambda_b0 = torch.randn(D).exp()  # Ensure positive.
    sig_u0 = torch.ones(D + B, M) if rand_eff else None

    # Create fake training data.
    df = create_dummy_df(D, B, M, G, T, C)

    # Create a model instance.
    model = MultPyro(
        alpha0=alpha0,
        w_mu0=w_mu0,
        w_var0=w_var0,
        lambda_a0=lambda_a0,
        lambda_b0=lambda_b0,
        sig_u0=sig_u0     
    )

    target_names = []
    for ii in range(D):
        target_names.append(f'real_target_{ii}')

    for ii in range(B):
        target_names.append(f'bool_target_{ii}')

    predictor_names = []
    for ii in range(M):
        predictor_names.append(f'predictor_{ii}')    

    # Fit the model.
    model.fit(df=df, target_names=target_names, predictor_names=predictor_names,
              groupby='id', num_steps=3)

    # Estimate parameters.
    params = model.estimate_params()
    assert isinstance(params, dict)
    if D:
        assert set(params) == {"W_mu", "W_var", "lambda_mu", "lambda_var"}
        assert params["lambda_mu"].shape == (K, D)
        assert params["lambda_var"].shape == (K, D)
    else:
        assert set(params) == {"W_mu", "W_var"}
    assert params["W_mu"].shape == (K, D + B, M)
    assert params["W_var"].shape == (K, D + B, M)

    # Create fake test data.
    df_test = create_dummy_df(D, B, M, G_, T, C)
    
    # Classify a novel batch of data of batch size B.
    probs = model.classify(df_test)
    assert probs.shape == (G_, K)
    assert probs.sum(-1).allclose(torch.ones(G_))

def test_augment_df_with_traj_info():
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

    # Augment df and evaluate
    df_aug = model.augment_df_with_traj_info(df)
    assert np.sum((df_aug.traj_gt.values == 1) & (df.traj.values == 1)) + \
        np.sum((df_aug.traj_gt.values == 2) & (df.traj.values == 0)), \
        "Error in trajectory assignment"

    for kk in range(model.K):
        assert np.sum(df_aug.groupby('id').\
            apply(lambda dd : np.all(dd[f'traj_{kk}'].values == \
                dd[f'traj_{kk}'].values[0]))) == df_aug.groupby('id').ngroups, \
                "Trajectory probabilities differ within individual"

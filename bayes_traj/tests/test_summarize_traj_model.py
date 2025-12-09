import torch
import numpy as np
import pandas as pd
import pdb
from bayes_traj.mult_dp_regression import MultDPRegression
from bayes_traj.summarize_traj_model import get_ranef_cov_mat_output_str

def random_covariance_matrix(M, seed=None):
    """
    Generates a random MxM symmetric positive semi-definite covariance matrix.

    Parameters:
    - M (int): Dimension of the covariance matrix
    - seed (int, optional): Random seed for reproducibility

    Returns:
    - cov_mat (np.ndarray): MxM covariance matrix
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate a random MxM matrix
    A = np.random.randn(M, M)
    
    # Compute a symmetric positive semi-definite matrix
    cov_mat = np.dot(A, A.T)

    # Optionally scale to unit variance for diagonal dominance
    # diag_scaler = np.diag(1 / np.sqrt(np.diag(cov_mat)))
    # cov_mat = diag_scaler @ cov_mat @ diag_scaler  # Normalize to correlation matrix

    return torch.from_numpy(cov_mat)

def test_print_ranef_cov_mat():
    """
    """
    seed = 42
    np.random.seed(seed)
    
    predictor_names = ['intercept', 'age', 'height']
    target_names = ['y1', 'y2']
    D = len(target_names)
    M = len(predictor_names)

    w_var0 = np.ones([M, D])
    w_mu0 = np.zeros([M, D])
    lambda_a0 = np.ones(D)
    lambda_b0 = np.ones(D)
    alpha = 1.
    K = 5
    prec_prior_weight = 1
    alpha = 1
    
    mm = MultDPRegression(w_mu0, w_var0, lambda_a0, lambda_b0,
                          prec_prior_weight, alpha, K=K)

    mm.predictor_names_ = predictor_names
    mm.target_names_ = target_names
    mm.M_ = M
    mm.ranef_indices_ = [True, False, True]
    mm.D_ = D
    mm.K_ = 2
    mm.G_ = 4
    num_long_pts = 3
    mm.R_ = torch.zeros((mm.G_*num_long_pts, mm.K_))
    mm.group_first_index_ = np.zeros(mm.G_*num_long_pts, dtype=bool)
    mm.u_Sig_ = torch.zeros((mm.G_, mm.D_, mm.K_, mm.M_, mm.M_))
    mm.u_mu_ = torch.zeros((mm.G_, mm.D_, mm.K_, mm.M_))

    probs_mat = np.random.uniform(0.01, 0.99, (mm.G_, mm.K_))
    probs_mat = (probs_mat.T/np.sum(probs_mat, 1)).T
    
    for gg in range(mm.G_):
        for ii in range(num_long_pts):
            mm.R_[gg*num_long_pts + ii, :] = torch.from_numpy(probs_mat[gg, :])
            if ii == 0:
                mm.group_first_index_[gg*num_long_pts + ii] = True
            
        for dd in range(mm.D_):
            for kk in range(mm.K_):
                mm.u_Sig_[gg, dd, kk, :, :] = \
                    random_covariance_matrix(mm.M_, (gg+1)*(dd+1)*(kk+1))
                np.random.seed((gg+1)*(dd+1)*(kk+1))
                mm.u_mu_[gg, dd, kk, :] = torch.from_numpy(\
                    np.random.multivariate_normal(np.zeros(mm.M_), np.eye(mm.M_)))
                                
    print_str = get_ranef_cov_mat_output_str(mm, 0, 0,
        precision=3, sci_notation_threshold=1e-3)

    print_str_ref = '                    intercept            height\nintercept               4.882             2.312\nheight                  2.312             3.060'

    assert print_str == print_str_ref


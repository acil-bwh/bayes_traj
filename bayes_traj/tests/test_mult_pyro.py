import pytest
import torch
from bayes_traj.mult_pyro import MultPyro


@pytest.mark.parametrize(
    "K, D, M, G, T, B",
    [
        (1, 1, 1, 1, 1, 1),
        (2, 3, 4, 5, 6, 7),   # All distinct to detect shape errors.
    ],
)
def test_smoke(K, D, M, G, T, B):
    # Create fake data.
    w_mu0 = torch.randn(D, M)
    w_var0 = torch.randn(D, M).exp()  # Ensure positive.
    lambda_a0 = torch.randn(D).exp()  # Ensure positive.
    lambda_b0 = torch.randn(D).exp()  # Ensure positive.
    X_train = torch.randn(T, G, M)
    Y_real_train = torch.randn(T, G, D)
    Y_real_mask_train = torch.ones(T, G).bernoulli().bool()

    # Create a model instance.
    model = MultPyro(
        K=K,
        w_mu0=w_mu0,
        w_var0=w_var0,
        lambda_a0=lambda_a0,
        lambda_b0=lambda_b0,
        X=X_train,
        Y_real=Y_real_train,
        Y_real_mask=Y_real_mask_train,
    )

    # Fit the model.
    model.fit(num_steps=3)

    # Estimate parameters.
    params = model.estimate_params()
    assert isinstance(params, dict)
    assert set(params) == {"W_mu", "W_var", "lambda_mu", "lambda_var"}
    assert params["W_mu"].shape == (K, D, M)
    assert params["W_var"].shape == (K, D, M)
    assert params["lambda_mu"].shape == (K, D)
    assert params["lambda_var"].shape == (K, D)

    # Classify a novel batch of data of batch size B.
    X_test = torch.randn(T, B, M)
    Y_real_test = torch.randn(T, B, D)
    Y_real_mask_test = torch.ones(T, B).bernoulli().bool()
    probs = model.classify(
        X=X_test, Y_real=Y_real_test, Y_real_mask=Y_real_mask_test
    )
    assert probs.shape == (B, K)
    assert probs.sum(-1).allclose(torch.ones(B))

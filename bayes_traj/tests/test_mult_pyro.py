import pytest
import torch
from bayes_traj.mult_pyro import MultPyro


@pytest.mark.parametrize(
    "K, D, M, G, T",
    [
        (1, 1, 1, 1, 1),
        (2, 3, 4, 5, 6),   # All distinct to detect shape errors.
    ],
)
def test_smoke(K, D, M, G, T):
    # Create fake data.
    w_mu0 = torch.randn(D, M)
    w_var0 = torch.randn(D, M).exp()  # Ensure positive.
    lambda_a0 = torch.randn(D).exp()  # Ensure positive.
    lambda_b0 = torch.randn(D).exp()  # Ensure positive.
    Y = torch.randn(T, G, D)
    X = torch.randn(T, G, M)
    obs_mask = torch.ones(T, G).bernoulli().bool()

    # Create a model instance.
    model = MultPyro(
        K=K,
        w_mu0=w_mu0,
        w_var0=w_var0,
        lambda_a0=lambda_a0,
        lambda_b0=lambda_b0,
        Y=Y,
        X=X,
        obs_mask=obs_mask,
    )

    # Fit the model.
    model.fit(num_steps=3)

    # Estimate parameters.
    params = model.estimate_params()
    assert isinstance(params, dict)
    assert set(params) == {"W_mu", "W_var", "lambda_mu", "lambda_var"}
    assert params["W_mu"].shape == (K, D, M)
    assert params["W_var"].shape == (K, D, M)
    assert params["lambda_mu"].shape == (K, D,)
    assert params["lambda_var"].shape == (K, D,)

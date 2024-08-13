from typing import Optional

import pytest
import torch

from bayes_traj.mult_pyro import MultPyro


@pytest.mark.parametrize("y_mask_dim", [2, 3])
@pytest.mark.parametrize("x_mask", [True, False])
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
def test_smoke(K, D, B, M, T, C, G, G_, x_mask: bool, y_mask_dim: int):
    # Set hyperparameters.
    alpha0 = torch.randn(K).exp()  # Ensure positive.
    w_mu0 = torch.randn(D + B, M)
    w_var0 = torch.randn(D + B, M).exp()  # Ensure positive.
    lambda_a0 = torch.randn(D).exp()  # Ensure positive.
    lambda_b0 = torch.randn(D).exp()  # Ensure positive.

    # Create fake training data.
    data_train = {}
    data_train["X"] = torch.randn(T, G, M)
    if x_mask:
        data_train["X_mask"] = torch.ones(T, G).bernoulli().bool()
    if D:
        data_train["Y_real"] = torch.randn(T, G, D)
        mask_shape = {2: (T, G), 3: (T, G, D)}[y_mask_dim]
        data_train["Y_real_mask"] = torch.ones(mask_shape).bernoulli().bool()
    if B:
        data_train["Y_bool"] = torch.ones(T, G, B).bernoulli().bool()
        mask_shape = {2: (T, G), 3: (T, G, B)}[y_mask_dim]
        data_train["Y_bool_mask"] = torch.ones(mask_shape).bernoulli().bool()
    if C > 1:
        data_train["cohort"] = torch.randint(C, (G,), dtype=torch.long)

    # Create a model instance.
    model = MultPyro(
        alpha0=alpha0,
        w_mu0=w_mu0,
        w_var0=w_var0,
        lambda_a0=lambda_a0,
        lambda_b0=lambda_b0,
        **data_train,
    )

    # Fit the model.
    model.fit(num_steps=3)

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
    data_test = {}
    data_test["X"] = torch.randn(T, G_, M)
    if x_mask:
        data_test["X_mask"] = torch.ones(T, G_).bernoulli().bool()
    if D:
        data_test["Y_real"] = torch.randn(T, G_, D)
        mask_shape = {2: (T, G_), 3: (T, G_, D)}[y_mask_dim]
        data_test["Y_real_mask"] = torch.ones(mask_shape).bernoulli().bool()
    if B:
        data_test["Y_bool"] = torch.ones(T, G_, B).bernoulli().bool()
        mask_shape = {2: (T, G_), 3: (T, G_, B)}[y_mask_dim]
        data_test["Y_bool_mask"] = torch.ones(mask_shape).bernoulli().bool()
    if C > 1:
        data_test["cohort"] = torch.randint(C, (G_,), dtype=torch.long)

    # Classify a novel batch of data of batch size B.
    probs = model.classify(**data_test)
    assert probs.shape == (G_, K)
    assert probs.sum(-1).allclose(torch.ones(G_))

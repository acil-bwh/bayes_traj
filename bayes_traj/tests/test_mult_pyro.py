import pytest
import torch
from bayes_traj.mult_pyro import MultPyro


@pytest.mark.parametrize(
    "K, D, M, G, N",
    [
        (1, 1, 1, 1, 1),
        (4, 2, 3, 6, 10),   # All distinct to detect shape errors. 
    ],
)
def test_fit_smoke(K, D, M, G, N):
    # Create fake data.
    w_mu0 = torch.randn(D, M)
    w_var0 = torch.randn(D, M).exp()  # Ensure positive.
    lambda_a0 = torch.randn(D).exp()  # Ensure positive.
    lambda_b0 = torch.randn(D).exp()  # Ensure positive.
    Y_sparse = torch.randn(N, D)
    X_sparse = torch.randn(N, M)
    # Ensure G can be inferred from sparse_individials.
    sparse_individuals = torch.randint(G, (N,))
    while sparse_individuals.max() != G - 1:
        sparse_individuals = torch.randint(G, (N,))

    # Create model.
    model = MultPyro(
        K=K,
        w_mu0=w_mu0,
        w_var0=w_var0,
        lambda_a0=lambda_a0,
        lambda_b0=lambda_b0,
        Y_sparse=Y_sparse,
        X_sparse=X_sparse,
        sparse_individuals=sparse_individuals,
    )

    model.fit(num_steps=3)

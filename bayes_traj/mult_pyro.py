from typing import Dict

import torch
import pyro
import pyro.distributions as dist
from pyro import poutine
from pyro.infer import SVI, TraceEnum_ELBO, config_enumerate
from pyro.infer.autoguide import AutoNormal

from pyro.optim import ClippedAdam


class MultPyro:
    def __init__(
        self,
        *,  # force passing by keyword to avoid confusion
        w_mu0: torch.Tensor,
        w_var0: torch.Tensor,
        lambda_a0: torch.Tensor,
        lambda_b0: torch.Tensor,
        Y: torch.Tensor,  # [T, G, D], real value
        X: torch.Tensor,  # [T, G, M], real value
        obs_mask: torch.Tensor,  # [T, G], boolean
        K: int,  # TODO configured from say --num-components
    ) -> None:
        """
        See MultDPRegression for parameter descriptions.

        Attributes:
            K: number of components
            D: number of targets
            G: number of individuals
            M: number of predictors (aka features)
            T: number of time points

        Args:
            Y (torch.Tensor): [T, G, D], real valued response tensor
            X (torch.Tensor): [T, G, M], real valued predictor tensor
            obs_mask (torch.Tensor): [T, G], boolean tensor indicating which
                entries of Y are observed. True means observed, False means
                missing.
        """
        # Collect sizes.
        assert Y.dim() == 3
        assert X.dim() == 3
        self.K = K
        self.T = T = Y.shape[0]
        self.G = G = Y.shape[1]
        self.D = D = Y.shape[2]
        self.M = M = X.shape[2]
        assert K > 0
        assert T > 0
        assert G > 0
        assert D > 0
        assert M > 0

        # Validate tensor shapes and dtypes.
        assert w_mu0.shape == (D, M)
        assert w_var0.shape == (D, M)
        assert lambda_a0.shape == (D,)
        assert lambda_b0.shape == (D,)
        assert Y.shape == (T, G, D)
        assert Y.dtype.is_floating_point
        assert X.shape == (T, G, M)
        assert X.dtype.is_floating_point
        assert obs_mask.shape == (T, G)
        assert obs_mask.dtype == torch.bool

        # Fixed parameters.
        self.w_mu0 = w_mu0
        self.w_var0 = w_var0
        self.lambda_a0 = lambda_a0
        self.lambda_b0 = lambda_b0

        # Data.
        self.Y = Y
        self.X = X
        self.obs_mask = obs_mask

        # Learned parameters.
        # self.R_ = R_
        # self.w_mu_ = w_mu_
        # self.w_var_ = w_var_
        # self.lambda_a = lambda_a
        # self.lambda_b = lambda_b

    def model(self) -> None:
        """
        The Bayesian model definition.

        This is called during each training step and during prediction.
        """
        D = self.D
        M = self.M
        K = self.K
        T = self.T
        G = self.G
        # We use two different dimensions for plates, which determines
        # max_plate_nesting.
        # See https://pyro.ai/examples/tensor_shapes.html
        components_plate = pyro.plate("components", K, dim=-1)  # (K,)
        individuals_plate = pyro.plate("individuals", G, dim=-1)  # (G,)
        time_plate = pyro.plate("time", T, dim=-2)  # (T, 1)

        assert self.lambda_a0.shape == (D,)
        assert self.lambda_b0.shape == (D,)
        with components_plate:
            # Sample the variance parameters.
            # We use .to_event(1) to sample a vectors of length D.
            lambda_ = pyro.sample(
                "lambda_",
                dist.Gamma(self.lambda_a0, self.lambda_b0).to_event(1)
            )
            assert isinstance(lambda_, torch.Tensor)
            assert lambda_.shape == (K, D)

            # Sample the regression coefficients.
            # We use .to_event(2) to sample matrices of shape [D, M].
            loc = self.w_mu0
            scale = self.w_var0.sqrt()
            assert loc.shape == (D, M)
            assert scale.shape == (D, M)
            W_ = pyro.sample("W_", dist.Normal(loc, scale).to_event(2))
            assert W_.shape == (K, D, M)

        # Sample the mixture component of each individual.
        # This will be enumerated out during SVI.
        # See https://pyro.ai/examples/enumeration.html
        with individuals_plate:
            class_probs = torch.ones(K) / K
            k = pyro.sample("k", dist.Categorical(class_probs))

            assert k.shape in {
                (G,),  # During prior sampling.
                (K, 1, 1),  # During training due to enumeration.
            }

            # Declare the likelihood, which is partially observed.
            with time_plate, poutine.mask(mask=self.obs_mask):
                # Compute the predicted mean.
                assert W_.shape == (K, D, M)
                assert k.shape in {(G,), (K, 1, 1)}
                assert k.max().item() < K
                W_n = W_[k]  # Determine each individual's W.
                assert W_n.shape in {(G, D, M), (K, 1, 1, D, M)}
                # We accomplish batched matrix-vector multiplication by
                # unsqueezing then squeezing.
                assert self.X.shape == (T, G, M)
                y_loc = (W_n @ self.X.unsqueeze(-1)).squeeze(-1)
                assert y_loc.shape in {(T, G, D), (K, T, G, D)}

                # Compute the predicted variance.
                assert lambda_.shape == (K, D)
                y_scale = lambda_.sqrt()[k]
                assert y_scale.shape in {(G, D), (K, 1, 1, D)}

                # Observers Y_sparse, i.e. the likelihood.
                assert self.Y.shape == (T, G, D)
                pyro.sample(
                    "Y", dist.Normal(y_loc, y_scale).to_event(1), obs=self.Y
                )

    def fit(
        self,
        *,
        learning_rate: float = 0.01,
        learning_rate_decay: float = 0.1,
        num_steps: int = 1001,
        seed: int = 20231215,
    ) -> None:
        """Fits the model via SVI."""
        # Reset state to ensure reproducibility.
        pyro.clear_param_store()
        pyro.set_rng_seed(seed)

        # Run SVI.
        # We need a guide only over the continuous latent variables since we're
        # marginalizing out the discrete latent variable k.
        self.guide = AutoNormal(poutine.block(self.model, hide=["k"]))
        optim = ClippedAdam(
            {"lr": learning_rate, "lrd": learning_rate_decay**(1 / num_steps)}
        )
        # We'll use TraceEnum_ELBO to marginalize out the discrete latent
        # variables.
        elbo = TraceEnum_ELBO(max_plate_nesting=2)
        marginal_model = config_enumerate(self.model, "parallel")
        svi = SVI(marginal_model, self.guide, optim, elbo)

        # We'll log a loss normalized per-observation, which is more
        # interpretable than total loss:
        # - normalized loss on the order of ~1 means the model is doing well
        # - normalized loss larger than ~100 means the model is doing poorly
        obs_count = int(self.obs_mask.long().sum())
        for step in range(num_steps):
            loss = svi.step()
            loss /= obs_count  # Per-observation loss is more interpretable.
            if step % 100 == 0:
                print(f"step {step: >4d} loss = {loss:.3f}")

    def estimate_params(
        self, num_samples: int = 1000
    ) -> Dict[str, torch.Tensor]:
        """Estimates posterior parameters."""
        # First draw samples from the guide.
        # Disable gradients and loss computations.
        with torch.no_grad(), pyro.poutine.mask(mask=False):
            # Draw many samples in parallel.
            with pyro.plate("particles", num_samples, dim=-3):
                samples = self.guide()

        # Compute moments from the samples.
        # Note this uses (mean,variance) of the lambda_ variable; we could
        # instead fit posterior parameters to a Gamma distribution.
        means = {k: v.mean(0).squeeze(0) for k, v in samples.items()}
        vars = {k: v.var(0).squeeze(0) for k, v in samples.items()}

        # Extract the parameters we care about.
        return {
            "W_mu": means["W_"],
            "W_var": vars["W_"],
            "lambda_mu": means["lambda_"],
            "lambda_var": vars["lambda_"],
        }

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Predicts `Y` from `X`."""
        raise NotImplementedError("TODO do we need this?")

    def classify(self, X: torch.Tensor) -> torch.Tensor:
        """
        Classifies `X`.

        Returns:
            probs: A `[G, K]`-shaped tensor of probabilities, normalized over
                the leftmost dimension.
        """
        raise NotImplementedError("TODO do we need this?")

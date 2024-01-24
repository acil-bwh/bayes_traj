from typing import Dict

import torch
import pyro
import pyro.distributions as dist
from pyro import poutine
from pyro.infer import SVI, TraceEnum_ELBO, config_enumerate, infer_discrete
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
        X: torch.Tensor,  # [T, G, M], real value
        Y_real: torch.Tensor,  # [T, G, D], real value
        Y_real_mask: torch.Tensor,  # [T, G], boolean
        K: int,  # TODO configured from say --num-components
    ) -> None:
        """
        See `MultDPRegression` for parameter descriptions.

        Attributes:
            K (int): number of components
            D (int): number of targets
            G (int): number of individuals
            M (int): number of predictors (aka features)
            T (int): number of time points

        Args:
            w_mu0 (torch.Tensor): [D, M], real valued prior mean for regression
                coefficients.
            w_var0 (torch.Tensor): [D, M], real valued prior variance for
                regression coefficients.
            lambda_a0 (torch.Tensor): [D], real valued prior shape for
                likelihood variance parameters.
            lambda_b0 (torch.Tensor): [D], real valued prior rate for
                likelihood variance parameters.
            X (torch.Tensor): [T, G, M], real valued predictor tensor.
            Y_real (torch.Tensor): [T, G, D], real valued response tensor.
            obs_mask (torch.Tensor): [T, G], boolean tensor indicating which
                entries of `Y_real` are observed. True means observed, False
                means missing.
            K (int): number of components.
        """
        # Collect sizes.
        assert Y_real.dim() == 3
        assert X.dim() == 3
        self.K = K
        self.T = T = Y_real.shape[0]
        self.G = G = Y_real.shape[1]
        self.D = D = Y_real.shape[2]
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
        assert Y_real.shape == (T, G, D)
        assert Y_real.dtype.is_floating_point
        assert X.shape == (T, G, M)
        assert X.dtype.is_floating_point
        assert Y_real_mask.shape == (T, G)
        assert Y_real_mask.dtype == torch.bool

        # Fixed parameters.
        self.w_mu0 = w_mu0
        self.w_var0 = w_var0
        self.lambda_a0 = lambda_a0
        self.lambda_b0 = lambda_b0

        # Data.
        self.X = X
        self.Y_real = Y_real
        self.Y_real_mask = Y_real_mask

    def model(
        self,
        X: torch.Tensor,
        Y_real: torch.Tensor,
        Y_real_mask: torch.Tensor,
    ) -> None:
        """
        The Bayesian model definition.

        This is called during each training step and during classification.
        """
        # Validate shapes.
        D = self.D
        M = self.M
        K = self.K
        T = self.T
        G = X.shape[1]
        assert X.shape == (T, G, M)
        assert Y_real.shape == (T, G, D)
        assert Y_real_mask.shape == (T, G)

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
            with time_plate, poutine.mask(mask=Y_real_mask):
                # Compute the predicted mean.
                assert W_.shape == (K, D, M)
                assert k.shape in {(G,), (K, 1, 1)}
                assert k.max().item() < K
                W_n = W_[k]  # Determine each individual's W.
                assert W_n.shape in {(G, D, M), (K, 1, 1, D, M)}
                # We accomplish batched matrix-vector multiplication by
                # unsqueezing then squeezing.
                assert X.shape == (T, G, M)
                y_loc = (W_n @ X.unsqueeze(-1)).squeeze(-1)
                assert y_loc.shape in {(T, G, D), (K, T, G, D)}

                # Compute the predicted variance.
                assert lambda_.shape == (K, D)
                y_scale = lambda_.sqrt()[k]
                assert y_scale.shape in {(G, D), (K, 1, 1, D)}

                # Observers Y_sparse, i.e. the likelihood.
                assert Y_real.shape == (T, G, D)
                pyro.sample(
                    "Y_real", dist.Normal(y_loc, y_scale).to_event(1), obs=Y_real
                )

    def fit(
        self,
        *,
        learning_rate: float = 0.01,
        learning_rate_decay: float = 0.1,
        num_steps: int = 1001,
        seed: int = 20231215,
    ) -> None:
        """Fits the model via Stochastic Variational Inference (SVI)."""
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

        # We'll log a loss that is normalized per-observation, which is more
        # interpretable than the total loss. We can use this loss to diagnose
        # model mismatch:
        # - Normalized loss on the order of ~1 means the model is doing well.
        # - Normalized loss larger than ~100 means the model is doing poorly.
        obs_count = int(self.Y_real_mask.long().sum())
        for step in range(num_steps):
            loss = svi.step(self.X, self.Y_real, self.Y_real_mask)
            loss /= obs_count  # Per-observation loss is more interpretable.
            if step % 100 == 0:
                print(f"step {step: >4d} loss = {loss:.3f}")

    def estimate_params(
        self, num_samples: int = 1000
    ) -> Dict[str, torch.Tensor]:
        """Estimates learned posterior parameters."""
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

    @torch.no_grad()
    def classify(
        self,
        *,
        X: torch.Tensor,
        Y_real: torch.Tensor,
        Y_real_mask: torch.Tensor,
        num_samples: int = 100,
    ) -> torch.Tensor:
        """
        Classifies a batch of individuals based on their predictors `X`and
        observed responses `Y_*`.

        Note the batch size `B` may differ from the training set size `G`.        

        Args:
            X (Tensor): A `[T, B, M]`-shaped tensor of predictors.
            Y_real (Tensor): A `[T, B, D]`-shaped tensor of responses.
            Y_real_mask (Tensor): A `[T, B]`-shaped boolean tensor indicating
                which entries of `Y_real` are observed. True means observed,
                False means missing.
            num_samples (int): Number of samples to draw in computing empirical
                class probabilities.
        Returns:
            (Tensor): A `[B, K]`-shaped tensor of empirical sample
                probabilities, normalized over the leftmost dimension.
        """
        # Validate shapes.
        assert X.dim() == 3
        B = X.shape[1]
        assert X.shape == (self.T, B, self.M)
        assert Y_real.shape == (self.T, B, self.D)

        # Draw samples sequentially to keep tensor shapes simple.
        probs = torch.zeros(B, self.K)
        b = torch.arange(B)
        for _ in range(num_samples):
            # Sample from the guide and condition the model on the guide.
            latent_sample = self.guide()  # Draw a sample from the guide.
            model = poutine.condition(self.model, data=latent_sample)

            # Use Pyro's infer_discrete to sample the discrete latent variable k.
            model = config_enumerate(model, "parallel")
            model = infer_discrete(model, first_available_dim=-3, temperature=1)
            trace = poutine.trace(model).get_trace(X, Y_real, Y_real_mask)

            # Accumulate the empirical sample probabilities.
            k = trace.nodes["k"]["value"]
            k = k.reshape((B,))
            probs[b, k] += 1
        probs /= num_samples
        return probs

from typing import Dict

import torch
import pyro
import pyro.distributions as dist
from pyro import poutine
from pyro.infer import SVI, TraceEnum_ELBO, config_enumerate, infer_discrete
from pyro.infer.autoguide import AutoNormal, init_to_sample

from pyro.optim import ClippedAdam


class MultPyro:
    def __init__(
        self,
        *,  # force passing by keyword to avoid confusion
        alpha0: torch.Tensor,
        w_mu0: torch.Tensor,
        w_var0: torch.Tensor,
        lambda_a0: torch.Tensor,
        lambda_b0: torch.Tensor,
        X: torch.Tensor,
        X_mask: torch.Tensor | None = None,
        Y_real: torch.Tensor | None = None,
        Y_real_mask: torch.Tensor | None = None,
        Y_bool: torch.Tensor | None = None,
        Y_bool_mask: torch.Tensor | None = None,
        cohort: torch.Tensor | None = None,
    ) -> None:
        """
        See `MultDPRegression` for parameter descriptions.

        Users should provide at least one of `Y_real` or `Y_bool`, together with
        their respective masks. Observations `Y_real` follow a linear-normal
        model with variance `lambda`, while observations `Y_bool` follow a
        linear-logistic model.

        This assumes one column of `X` is a constant ones column, which
        represents the intercept term.

        Attributes:
            K (int): number of components
            T (int): number of time points (max per individual)
            G (int): number of individuals
            M (int): number of predictors (aka features)
            D (int): number of real targets `Y_real`
            B (int): number of boolean targets `Y_bool`
            C (int): number of cohorts

        Args:
            alpha0 (torch.Tensor): [K], real valued Dirichlet prior
                concentration for mixture components. The shape of `alpha0`
                determines the number of mixture components.
            w_mu0 (torch.Tensor): [D + B, M], real valued prior mean for
                regression coefficients.
            w_var0 (torch.Tensor): [D + B, M], real valued prior variance for
                regression coefficients.
            lambda_a0 (torch.Tensor): [D], real valued prior shape for
                likelihood precision (1/variance) parameters.
            lambda_b0 (torch.Tensor): [D], real valued prior rate for
                likelihood precision (1/variance) parameters.
            X (torch.Tensor): [T, G, M], real valued predictor tensor.
            X_mask (torch.Tensor): [T, G], boolean tensor indicating which
                entries of `X` are observed. True means observed, False means
                missing.
            Y_real (optional torch.Tensor): [T, G, D], real valued response
                tensor.
            Y_real_mask (optional torch.Tensor): [T, G] or [T, G, D], boolean
                tensor indicating which entries of `Y_real` are observed. True
                means observed, False means missing.
            Y_bool (optional torch.Tensor): [T, G, B], boolean valued response
                tensor. This should have `.dtype == torch.bool`, although a
                floating point tensor is used internally.
            Y_bool_mask (optional torch.Tensor): [T, G] or [T, G, B], boolean
                tensor indicating which entries of `Y_bool` are observed. True
                means observed, False means missing.
            cohort (optional torch.Tensor): [G], integer array containing the
                cohort of each individual.
        """
        # Validate predictor data.
        assert X.dtype.is_floating_point
        assert X.dim() == 3
        self.X = X
        self.T = T = X.shape[0]
        self.G = G = X.shape[1]
        self.M = M = X.shape[2]
        assert T > 0
        assert G > 0
        assert M > 0

        # Validate predictor mask.
        self.X_mask = None
        if X_mask is not None:
            assert X_mask.dtype == torch.bool
            assert X_mask.shape == (T, G)
            self.X_mask = X_mask

        # Check for real observations.
        if Y_real is None:
            assert Y_real_mask is None
            self.D = D = 0
        else:
            assert Y_real_mask is not None
            assert Y_real.dim() == 3
            self.D = D = Y_real.shape[2]
            assert Y_real.shape == (T, G, D)
            assert Y_real.dtype.is_floating_point
            assert Y_real_mask.shape in {(T, G), (T, G, D)}
            assert Y_real_mask.dtype == torch.bool
            self.Y_real = Y_real
            if Y_real_mask.dim() == 2:
                Y_real_mask = Y_real_mask.unsqueeze(-1).expand(T, G, D)
                assert Y_real_mask.shape == (T, G, D)
            self.Y_real_mask = Y_real_mask

            # Validate likelihood parameter.
            assert lambda_a0.shape == (D,)
            assert lambda_b0.shape == (D,)
            self.lambda_a0 = lambda_a0
            self.lambda_b0 = lambda_b0
        assert D >= 0

        # Check for boolean observations.
        if Y_bool is None:
            assert Y_bool_mask is None
            self.B = B = 0
        else:
            assert Y_bool_mask is not None
            assert Y_bool.dim() == 3
            self.B = B = Y_bool.shape[2]
            assert Y_bool.shape == (T, G, B)
            assert Y_bool.dtype == torch.bool
            assert Y_bool_mask.shape in {(T, G), (T, G, B)}
            assert Y_bool_mask.dtype == torch.bool
            self.Y_bool = Y_bool
            if Y_bool_mask.dim() == 2:
                Y_bool_mask = Y_bool_mask.unsqueeze(-1).expand(T, G, B)
                assert Y_bool_mask.shape == (T, G, B)
            self.Y_bool_mask = Y_bool_mask
        assert B >= 0
        assert B or D, "Must provide at least one of Y_real or Y_bool."

        # Check for cohort data.
        if cohort is None:
            self.C = 1
        else:
            assert cohort.shape == (G,)
            self.C = int(cohort.max()) + 1
            self.cohort = cohort
        assert self.C > 0

        # Validate fixed parameters.
        assert alpha0.dim() == 1
        assert alpha0.dtype.is_floating_point
        assert (alpha0 > 0).all()
        self.K = K = alpha0.shape[0]
        assert K > 0
        assert w_mu0.shape == (D + B, M)
        assert w_var0.shape == (D + B, M)
        self.alpha0 = alpha0
        self.w_mu0 = w_mu0
        self.w_var0 = w_var0

    def model(
        self,
        X: torch.Tensor,
        *,
        X_mask: torch.BoolTensor | None = None,
        Y_real: torch.Tensor | None = None,
        Y_real_mask: torch.BoolTensor | None = None,
        Y_bool: torch.Tensor | None = None,  # Expected to be floating point.
        Y_bool_mask: torch.BoolTensor | None = None,
        cohort: torch.LongTensor | None = None,
    ) -> None:
        """
        The Bayesian model definition.

        This is called during each training step and during classification.
        """
        # Validate shapes.
        D = self.D
        B = self.B
        M = self.M
        K = self.K
        T = self.T
        C = self.C
        G = X.shape[1]
        assert X.shape == (T, G, M)

        # We use two different dimensions for plates, which determines
        # max_plate_nesting.
        # See https://pyro.ai/examples/tensor_shapes.html
        cohorts_plate = pyro.plate("cohorts", C, dim=-1)  # (C,)
        components_plate = pyro.plate("components", K, dim=-1)  # (K,)
        individuals_plate = pyro.plate("individuals", G, dim=-1)  # (G,)
        time_plate = pyro.plate("time", T, dim=-2)  # (T, 1)

        # Sample the distribution over classes.
        if C == 1:
            # Model a flat prior over classes.
            class_probs = pyro.sample("class_probs", dist.Dirichlet(self.alpha0))
        else:
            # Model cohorts hierarchically.
            assert cohort is not None
            alpha_cohort = pyro.sample("alpha_cohort", dist.Dirichlet(self.alpha0))
            with cohorts_plate:
                class_probs = pyro.sample("class_probs", dist.Dirichlet(alpha_cohort))
                assert class_probs.shape[-2:] == (C, K)
            class_probs = class_probs[cohort]
            assert class_probs.shape[-2:] == (G, K)

        # Sample class-dependent parameters.
        with components_plate:
            # Sample the regression coefficients.
            # We use .to_event(2) to sample matrices of shape [D, M].
            loc = self.w_mu0
            scale = self.w_var0.sqrt()
            assert loc.shape == (D + B, M)
            assert scale.shape == (D + B, M)
            W_ = pyro.sample("W_", dist.Normal(loc, scale).to_event(2))
            assert W_.shape == (K, D + B, M)

            if self.D:
                # Sample the real precision (1/variance) parameters.
                # We use .to_event(1) to sample a vectors of length D.
                assert self.lambda_a0.shape == (D,)
                assert self.lambda_b0.shape == (D,)
                lambda_ = pyro.sample(
                    "lambda_",
                    dist.Gamma(self.lambda_a0, self.lambda_b0).to_event(1)
                )
                assert isinstance(lambda_, torch.Tensor)
                assert lambda_.shape == (K, D)

        # Sample the mixture component of each individual.
        # This will be enumerated out during SVI.
        # See https://pyro.ai/examples/enumeration.html
        with individuals_plate:
            k = pyro.sample("k", dist.Categorical(class_probs))

            assert k.shape in {
                (G,),  # During prior sampling.
                (K, 1, 1),  # During training due to enumeration.
            }

            # Compute the predicted mean.
            assert W_.shape == (K, D + B, M)
            assert k.shape in {(G,), (K, 1, 1)}
            assert k.max().item() < K
            W_n = W_[k]  # Determine each individual's W.
            assert W_n.shape in {(G, D + B, M), (K, 1, 1, D + B, M)}
            assert X.shape == (T, G, M)
            if X_mask is not None:
                # Avoid NaNs in the masked-out entries.
                assert X_mask.shape == (T, G)
                X = X.masked_fill(~X_mask.unsqueeze(-1), 0)
            # We accomplish batched matrix-vector multiplication by
            # unsqueezing then squeezing.
            y = (W_n @ X.unsqueeze(-1)).squeeze(-1)
            assert y.shape in {(T, G, D + B), (K, T, G, D + B)}

        if D:  # Check for real observations.
            assert Y_real is not None
            assert Y_real_mask is not None
            assert Y_real.shape == (T, G, D)
            assert Y_real_mask.shape == (T, G, D)
            assert Y_real.dtype.is_floating_point
            assert Y_real_mask.dtype == torch.bool
            if X_mask is not None:
                Y_real_mask = Y_real_mask & X_mask.unsqueeze(-1)  # type: ignore[assignment]
            # Avoid NaNs in the masked-out entries.
            Y_real = Y_real.masked_fill(~Y_real_mask, 0)

            # Declare the real likelihood, which is partially observed.
            with individuals_plate, time_plate:
                # Extract the predicted mean.
                y_loc = y[..., :D]
                assert y_loc.shape in {(T, G, D), (K, T, G, D)}

                # Compute the predicted scales.
                assert lambda_.shape == (K, D)
                y_scale = lambda_.rsqrt()[k]
                assert y_scale.shape in {(G, D), (K, 1, 1, D)}

                # Observe Y_sparse, i.e. the real likelihood.
                assert Y_real.shape == (T, G, D)
                pyro.sample(
                    "Y_real",
                    dist.Normal(y_loc, y_scale).mask(Y_real_mask).to_event(1),
                    obs=Y_real,
                )

        if B:  # Check for boolean observations.
            assert Y_bool is not None
            assert Y_bool_mask is not None
            assert Y_bool.shape == (T, G, B)
            assert Y_bool_mask.shape == (T, G, B)
            assert Y_bool.dtype.is_floating_point
            assert Y_bool_mask.dtype == torch.bool
            if X_mask is not None:
                Y_bool_mask = Y_bool_mask & X_mask.unsqueeze(-1)  # type: ignore[assignment]

            # Declare the boolean likelihood, which is partially observed.
            with individuals_plate, time_plate:
                # Extract the predicted mean.
                y_loc = y[..., D:]
                assert y_loc.shape in {(T, G, B), (K, T, G, B)}

                # Observe Y_bool, i.e. the boolean likelihood.
                # TODO consider switching to a beta-Bernoulli model once we get
                # the simple Bernoulli model working.
                assert Y_bool.shape == (T, G, B)
                pyro.sample(
                    "Y_bool",
                    dist.Bernoulli(logits=y_loc).mask(Y_bool_mask).to_event(1),
                    obs=Y_bool,
                )

    def fit(
        self,
        *,
        learning_rate: float = 0.05,
        learning_rate_decay: float = 0.1,
        num_steps: int = 1001,
        seed: int = 20231215,
    ) -> None:
        """Fits the model via Stochastic Variational Inference (SVI)."""
        # Reset state to ensure reproducibility.
        pyro.set_rng_seed(seed)
        with pyro.get_param_store().scope() as self.params:
            # Run SVI.
            # We need a guide only over the continuous latent variables since we're
            # marginalizing out the discrete latent variable k.
            self.guide = AutoNormal(poutine.block(self.model, hide=["k"]),
                                    init_scale=0.01,
                                    init_loc_fn=init_to_sample)
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
            obs_count = 0  # computed below.

            # Handle real and boolean observations.
            data = {"X": self.X}
            if self.X_mask is not None:
                data["X_mask"] = self.X_mask
            if self.D:
                obs_count += int(self.Y_real_mask.long().sum())
                data["Y_real"] = self.Y_real
                data["Y_real_mask"] = self.Y_real_mask
            if self.B:
                obs_count += int(self.Y_bool_mask.long().sum())
                # Convert to float for Bernoulli.log_prob().
                data["Y_bool"] = self.Y_bool.to(dtype=self.X.dtype)
                data["Y_bool_mask"] = self.Y_bool_mask
            if self.C > 1:
                data["cohort"] = self.cohort

            self.losses = []
            for step in range(num_steps):
                loss = svi.step(**data)
                loss /= obs_count  # Per-observation loss is interpretable.
                self.losses.append(loss)
                if step % 100 == 0:
                    print(f"step {step: >4d} loss = {loss:.3f}")

    def estimate_params(
        self, num_samples: int = 1000
    ) -> Dict[str, torch.Tensor]:
        """Estimates learned posterior parameters."""
        with pyro.get_param_store().scope(self.params):
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

            # Extract relevant parameters.
            params = {"W_mu": means["W_"], "W_var": vars["W_"]}
            if self.D:
                params["lambda_mu"] = means["lambda_"]
                params["lambda_var"] = vars["lambda_"]
            return params

    @torch.no_grad()
    def classify(
        self,
        X: torch.Tensor,
        *,
        X_mask: torch.Tensor | None = None,
        Y_real: torch.Tensor | None = None,
        Y_real_mask: torch.Tensor | None = None,
        Y_bool: torch.Tensor | None = None,
        Y_bool_mask: torch.Tensor | None = None,
        cohort: torch.Tensor | None = None,
        num_samples: int = 100,
    ) -> torch.Tensor:
        """
        Classifies a batch of individuals based on their predictors `X` and
        observed responses `Y_*`.

        Note the batch size `G_` may differ from the training set size `G`.

        Args:
            X (Tensor): A `[T, G_, M]`-shaped tensor of predictors.
            X_mask (optional Tensor): A `[T, G_]`-shaped boolean tensor
            Y_real (optional Tensor): A `[T, G_, D]`-shaped tensor of responses.
            Y_real_mask (optional Tensor): A `[T, G_]` or `[T, G_, D]` shaped
                boolean tensor indicating which entries of `Y_real` are
                observed.  True means observed, False means missing.
            Y_bool (optional Tensor): A `[T, G_, B]`-shaped tensor of responses.
            Y_bool_mask (optional Tensor): A `[T, G_]` or `[T, G_, B]` shaped
                boolean tensor indicating which entries of `Y_bool` are
                observed.  True means observed, False means missing.
            cohort (optional Tensor): A `[G_]`-shaped long tensor of cohort
                indices.
            num_samples (int): Number of samples to draw in computing empirical
                class probabilities.
        Returns:
            (Tensor): A `[G_, K]`-shaped tensor of empirical sample
                probabilities, normalized over the leftmost dimension.
        """
        # Validate shapes.
        T = self.T
        D = self.D
        B = self.B
        M = self.M
        assert X.dim() == 3
        G_ = X.shape[1]
        assert X.shape == (T, G_, M)
        data = {"X": X}
        if X_mask is not None:
            assert X_mask.shape == (T, G_)
            assert X_mask.dtype == torch.bool
            data["X_mask"] = X_mask
        if Y_real is not None:
            assert Y_real_mask is not None
            assert Y_real.shape == (T, G_, D)
            assert Y_real_mask.shape in {(T, G_), (T, G_, D)}
            assert Y_real.dtype.is_floating_point
            assert Y_real_mask.dtype == torch.bool
            data["Y_real"] = Y_real
            if Y_real_mask.dim() == 2:
                Y_real_mask = Y_real_mask.unsqueeze(-1).expand(T, G_, D)
                assert Y_real_mask.shape == (T, G_, D)
            data["Y_real_mask"] = Y_real_mask
        if Y_bool is not None:
            assert Y_bool_mask is not None
            assert Y_bool.shape == (T, G_, B)
            assert Y_bool_mask.shape in {(T, G_), (T, G_, B)}
            assert Y_bool.dtype == torch.bool
            assert Y_bool_mask.dtype == torch.bool
            # Convert to float for Bernoulli.log_prob().
            data["Y_bool"] = Y_bool.to(dtype=self.X.dtype)
            if Y_bool_mask.dim() == 2:
                Y_bool_mask = Y_bool_mask.unsqueeze(-1).expand(T, G_, B)
                assert Y_bool_mask.shape == (T, G_, B)
            data["Y_bool_mask"] = Y_bool_mask
        if cohort is not None:
            assert cohort.dim() == 1
            assert cohort.shape[0] == G_
            data["cohort"] = cohort

        with pyro.get_param_store().scope(self.params):
            # Draw samples sequentially to keep tensor shapes simple.
            probs = torch.zeros(G_, self.K)
            g = torch.arange(G_)
            for _ in range(num_samples):
                # Sample from the guide and condition the model on the guide.
                latent_sample = self.guide()  # Draw a sample from the guide.
                model = poutine.condition(self.model, data=latent_sample)

                # Use Pyro's infer_discrete to sample the discrete latent variable k.
                model = config_enumerate(model, "parallel")
                model = infer_discrete(model, first_available_dim=-3, temperature=1)
                trace = poutine.trace(model).get_trace(**data)

                # Accumulate the empirical sample probabilities.
                k = trace.nodes["k"]["value"]
                assert k is not None
                k = k.reshape((G_,))
                probs[g, k] += 1
            probs /= num_samples
            return probs

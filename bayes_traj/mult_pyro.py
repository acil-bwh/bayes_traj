from typing import Dict

import torch
import pyro
import pyro.distributions as dist
from pyro import poutine
from pyro.infer import SVI, TraceEnum_ELBO, config_enumerate, infer_discrete
from pyro.infer.autoguide import AutoNormal, init_to_sample
from pyro.optim import ClippedAdam
import pandas as pd
from bayes_traj.pyro_helper import *
import pdb
from bayes_traj.base_model import BaseModel

class MultPyro(BaseModel):
    def __init__(
        self,
        *,  # force passing by keyword to avoid confusion
        alpha0: torch.Tensor,
        w_mu0: torch.Tensor,
        w_var0: torch.Tensor,
        lambda_a0: torch.Tensor,
        lambda_b0: torch.Tensor,
        sig_u0: torch.Tensor | None = None,
        conc_u0: float | None = None,
    ) -> None:
        """
        See `MultDPRegression` for parameter descriptions.

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
            sig_u0 (optional torch.Tensor): [D + B, M], positive valued prior
                over random effect scales. TODO allow this to be sparse in D+B.
                (@Fritz: I think we want this to (also) be sparse in M. E.g.,
                in practice, we may only want to consider random effects
                for the intercept, or some subset of the predictors)
            conc_u0 (optional float): positive concentration parameter for the
                LKJ prior over the random effects correlation matrix. Defaults
                to a uniform prior value of 1.0.
                (@Fritz: a default value of 1 seems fine. This probably ought
                to be a [D + B] dimensional vector, though, to support per-
                target specification)
        """
        # Check for random effects.
        if sig_u0 is None:
            assert conc_u0 is None
        else:
            if conc_u0 is None:
                conc_u0 = 1.0
            assert isinstance(conc_u0, float)
            assert conc_u0 > 0
        self.sig_u0 = sig_u0
        self.conc_u0 = conc_u0

        # Validate fixed parameters.
        assert alpha0.dim() == 1
        assert alpha0.dtype.is_floating_point
        assert (alpha0 > 0).all()
        self.K = K = alpha0.shape[0]
        assert K > 0
        self.alpha0 = alpha0
        self.w_mu0 = w_mu0
        self.w_var0 = w_var0
        self.lambda_a0 = lambda_a0
        self.lambda_b0 = lambda_b0

        self.X = None
        self.X_mask = None
        self.Y_real = None
        self.Y_real_mask = None
        self.Y_bool = None
        self.Y_bool_mask = None
        self.cohort = None
        
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

        TODO: we'll want to extend the model definition to accommodate 
        specification of random effects. Whether or not the user desires to use
        random effects will be indicated in the structure of sig_u0, the prior
        over the zero-centered, multivariate covariance matrix: if all elements
        of sig_u0 are 0, then random effects are to be ignored. Generally,
        the presence of 0 at any matrix location will indicate not to model that
        matrix element. We'll have a covariance matrix prior for each of the
        possible D+B targets. During inference, we'll want to estimate each
        trajectory's random effects covariance matrix (so (D+B)xK matrices).
        In the most general case, this is a lot of parameters to estimate --
        in practical usage, I expect we can initialize these matrices close
        to a local minimum... In terms of what prior to use, I guess 
        inverse Wishart is conjugate, but I think whatever pyro handles most
        naturally is fine...

        Users should provide at least one of `Y_real` or `Y_bool`, together with
        their respective masks. Observations `Y_real` follow a linear-normal
        model with variance `lambda`, while observations `Y_bool` follow a
        linear-logistic model.

        This assumes one column of `X` is a constant ones column, which
        represents the intercept term.

        Args:
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

            if self.sig_u0 is not None:
                # Sample the random effects covariance matrices
                #   cov = chol_cov_u @ chol_cov_u.mT
                # where the Cholesky factor chol_cov_u is decomposed from an LKJ
                # correlation matrix chol_corr_u and a scale vector scale_u.
                # See https://pyro.ai/examples/lkj.html
                assert self.sig_u0.shape == (D + B, M)
                scale_u = pyro.sample(
                    "scale_u", dist.HalfCauchy(self.sig_u0).to_event(2)
                )
                assert scale_u.shape == (K, D + B, M)
                if M == 1:  # The trivial case is deterministic.
                    chol_cov_u = scale_u[..., None]
                else:
                    assert self.conc_u0 is not None
                    assert isinstance(self.conc_u0, float)
                    conc_u = scale_u.new_full((D + B,), self.conc_u0)
                    chol_corr_u = pyro.sample(
                        "chol_corr_u", dist.LKJCholesky(M, conc_u).to_event(1)
                    )
                    assert chol_corr_u.shape == (K, D + B, M, M)
                    chol_cov_u = scale_u[..., None] * chol_corr_u
                assert chol_cov_u.shape == (K, D + B, M, M)

        # Sample the mixture component of each individual.
        # This will be enumerated out during SVI.
        # See https://pyro.ai/examples/enumeration.html
        with individuals_plate:
            k = pyro.sample("k", dist.Categorical(class_probs))
            assert k.shape in {
                (G,),  # During prior sampling.
                (K, 1, 1),  # During training due to enumeration.
            }

            # Determine the individual's coefficient W.
            assert W_.shape == (K, D + B, M)
            assert k.shape in {(G,), (K, 1, 1)}
            assert k.max().item() < K
            W_n = W_[k]  # Determine each individual's W.
            assert W_n.shape in {(G, D + B, M), (K, 1, 1, D + B, M)}

            # Optionally add random effects.
            if self.sig_u0 is not None:
                u0 = torch.zeros(M)
                assert chol_cov_u.shape == (K, D + B, M, M)
                u = pyro.sample(
                    "u",
                    dist.MultivariateNormal(u0, scale_tril=chol_cov_u[k]).to_event(1),
                )
                assert u.shape in {(G, D + B, M), (K, 1, G, D + B, M)}
                W_n = W_n + u
                assert W_n.shape in {(G, D + B, M), (K, 1, G, D + B, M)}

            # Compute the predicted mean.
            assert X.shape == (T, G, M)
            if X_mask is not None:
                # Avoid NaNs in the masked-out entries.
                assert X_mask.shape == (T, G, M)
                #X = X.masked_fill(~X_mask.unsqueeze(-1), 0)
                X = X.masked_fill(~X_mask, 0)
                
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
        target_names,
        predictor_names,
        df,
        groupby,            
        learning_rate: float = 0.05,
        learning_rate_decay: float = 0.1,
        num_steps: int = 1001,
        seed: int = 20231215,
    ) -> None:
        """Fits the model via Stochastic Variational Inference (SVI).
        """        

        assert len(set(target_names)) == len(target_names), \
            "Duplicate target name found"
        self.target_names_ = target_names
    
        assert len(set(predictor_names)) == len(predictor_names), \
            "Duplicate predictor name found"
        self.predictor_names_ = predictor_names

        self.groupby_col_ = groupby
        restructured_data = \
            get_restructured_data(df, predictor_names, target_names, groupby)
        self.group_to_index = restructured_data['group_to_index']
        
        # Validate predictor data.
        assert restructured_data['X'].dtype.is_floating_point
        assert restructured_data['X'].dim() == 3
        self.X = restructured_data['X']
        self.T = T = restructured_data['X'].shape[0]
        self.G = G = restructured_data['X'].shape[1]
        self.M = M = restructured_data['X'].shape[2]
        assert T > 0
        assert G > 0
        assert M > 0

        self.real_target_names_ = restructured_data['Y_real_names']
        self.bool_target_names_ = restructured_data['Y_bool_names']        

        # Validate predictor mask.
        self.X_mask = None
        if restructured_data['X_mask'] is not None:
            assert restructured_data['X_mask'].dtype == torch.bool
            assert restructured_data['X_mask'].shape == (T, G, M)
            self.X_mask = restructured_data['X_mask']

        # Check for real observations.
        if restructured_data['Y_real'] is None:
            assert restructured_data['Y_real_mask'] is None
            self.D = D = 0
        else:
            assert restructured_data['Y_real_mask'] is not None
            assert restructured_data['Y_real'].dim() == 3
            self.D = D = restructured_data['Y_real'].shape[2]
            assert restructured_data['Y_real'].shape == (T, G, D)
            assert restructured_data['Y_real'].dtype.is_floating_point
            assert restructured_data['Y_real_mask'].shape in {(T, G), (T, G, D)}
            assert restructured_data['Y_real_mask'].dtype == torch.bool
            self.Y_real = restructured_data['Y_real']
            if restructured_data['Y_real_mask'].dim() == 2:
                restructured_data['Y_real_mask'] = \
                    restructured_data['Y_real_mask'].unsqueeze(-1).expand(T, G, D)
                assert restructured_data['Y_real_mask'].shape == (T, G, D)
            self.Y_real_mask = restructured_data['Y_real_mask']

            # Validate likelihood parameter.
            assert self.lambda_a0.shape == (D,)
            assert self.lambda_b0.shape == (D,)
        assert D >= 0

        # Check for boolean observations.
        if restructured_data['Y_bool'] is None:
            assert restructured_data['Y_bool_mask'] is None
            self.B = B = 0
        else:
            assert restructured_data['Y_bool_mask'] is not None
            assert restructured_data['Y_bool'].dim() == 3
            self.B = B = restructured_data['Y_bool'].shape[2]
            assert restructured_data['Y_bool'].shape == (T, G, B)
            assert restructured_data['Y_bool'].dtype.is_floating_point
            assert restructured_data['Y_bool_mask'].shape in {(T, G), (T, G, B)}
            assert restructured_data['Y_bool_mask'].dtype == torch.bool
            self.Y_bool = restructured_data['Y_bool']
            if restructured_data['Y_bool_mask'].dim() == 2:
                restructured_data['Y_bool_mask'] = \
                    restructured_data['Y_bool_mask'].unsqueeze(-1).expand(T, G, B)
                assert restructured_data['Y_bool_mask'].shape == (T, G, B)
            self.Y_bool_mask = restructured_data['Y_bool_mask']
        assert B >= 0
        assert B or D, "Must provide at least one of Y_real or Y_bool."

        # Check for cohort data.
        if restructured_data['cohort'] is None:
            self.C = 1
        else:
            assert restructured_data['cohort'].shape == (G,)
            self.C = int(restructured_data['cohort'].max()) + 1
            self.cohort = restructured_data['cohort']
        assert self.C > 0

        # Validate random effect prior shape if necessary
        if self.sig_u0 is not None:
            assert self.sig_u0.shape == (D + B, M)

        # Validate prior shapes
        assert self.w_mu0.shape == (D + B, M)
        assert self.w_var0.shape == (D + B, M)
        
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
        df = None,
        num_samples: int = 100,
    ) -> torch.Tensor:
        """
        Classifies a batch of individuals based on their predictors `X` and
        observed responses `Y_*`.

        Note the batch size `G_` may differ from the training set size `G`.

        TODO: make sure classification can apply to test / out-of-sample data.
        Specifically, make sure that this routine can classify individuals who
        have a different number of time points compared to the training data.

        Parameters
        ----------
        df : pandas DataFrame, optional
            Data frame containing data to classify. This data frame should have 
            columns that correspond to the predictor names, target names, and 
            subject identifier (groupby) column that were used to train the 
            model. If none specified, this function will by default classify 
            the data that was used to train the model

        Returns
        -------
        probs : Tensor
            A `[G_, K]`-shaped tensor of empirical sample probabilities, 
            normalized over the leftmost dimension.
        """
        if df is None:
            # By default, use data trained on
            X = self.X
            X_mask = self.X_mask
            Y_real = self.Y_real
            Y_real_mask = self.Y_real_mask
            Y_bool = self.Y_bool
            Y_bool_mask = self.Y_bool_mask
            cohort = self.cohort
        else:
            restructured_data = \
                get_restructured_data(df, self.predictor_names_,
                                      self.target_names_, self.groupby_col_)
            X = restructured_data['X']
            X_mask = restructured_data['X_mask']
            Y_real = restructured_data['Y_real']
            Y_real_mask = restructured_data['Y_real_mask']
            Y_bool = restructured_data['Y_bool']
            Y_bool_mask = restructured_data['Y_bool_mask']
            cohort = restructured_data['cohort']

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
            assert X_mask.shape == (T, G_, M)
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
            assert Y_bool.dtype.is_floating_point
            assert Y_bool_mask.dtype == torch.bool
            data["Y_bool"] = Y_bool
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

    def augment_df_with_traj_info(self, df):
        """
        """
        probs = self.classify(df)
        re_data = \
            get_restructured_data(df, self.predictor_names_,
                    self.target_names_, self.groupby_col_)

        traj_probs = {}
        for ii in range(self.K):
            traj_probs[f'traj_{ii}'] = np.array([np.nan]*df.shape[0])
        
        traj = np.array([np.nan]*df.shape[0])
        for ii, gg in enumerate(re_data['group_to_index'].keys()):
            which_traj = \
                np.where(probs[ii].numpy() == np.max(probs[ii].numpy()))[0][0]            
            traj[re_data['group_to_index'][gg]] = which_traj
            for tt in range(self.K):
                traj_probs[f'traj_{tt}'][re_data['group_to_index'][gg]] = \
                    probs[ii].numpy()[tt]

        df['traj'] = traj
        for ii in range(self.K):
            df[f'traj_{ii}'] = traj_probs[f'traj_{ii}']
        
        return df

    def plot(self, x_axis, y_axis, x_label=None, y_label=None, which_trajs=None,
             show=True, min_traj_prob=0, max_traj_prob=1, traj_map=None,
             hide_traj_details=False, hide_scatter=False, traj_markers=None,
             traj_colors=None, fill_alpha=0.3):
        """Generates a 2D plot of trajectory results. The original data will be
        shown as a scatter plot, color-coded according to trajectory membership.
        Trajectories will be plotted with line plots indicating the expected 
        target value given predictor values. The user has control over what
        variables will appear on the x- and y-axes. This plotting function
        expects that predictors raised to a power use the ^ character. E.g.
        predictors might be: 'age' and 'age^2'. In this case, if the user
        wants to plot 'age' on the x-axis, he/she need only specify 'age' (the
        plotting routine will take care of the higher order terms). Predictor
        variables not specified to be on the x-axis will be set to their mean
        values for plotting.     
        
        TODO: Update to accomodate binary target variables as necessary

        Parameters
        ----------
        x_axis : str
            Predictor name corresponding to x-axis.

        y_axis : str
            Target variable name corresponding to y-axis.

        x_label : str, optional
            Label to display on x-axis. If none given, the variable name 
            specified with x_axis will be used

        y_label : str, optional
            Label to display on y-axis. If none given, the variable name 
            specified with y_axis will be used

        which_trajs : int or array, optional
            If specified, only these trajectories will be plotted. If not 
            specified, all trajectories will be plotted.

        show : bool, optional
            By default, invocation of this function will show the plot. If set
            to false, the handle to the axes will be returned, but the plot will
            not be displayed

        min_traj_prob : float, optional
            The probability of a given trajectory must be at least this value in
            order to be rendered. Value should be between 0 and 1 inclusive.

        max_traj_prob : float, optional
            The probability of a given trajectory can not be larger than this 
            value in order to be rendered. Value should be between 0 and 1 
            inclusive.

        traj_map : dict, optional
            Int-to-int mapping of trajectories, where keys are the original 
            (default) trajectory numbers, and the values are the new trajectory
            numbers. This for display purposes only. Supercedes which_trajs.

        hide_traj_details : bool, optional
            If true, trajectory details (N and percentage of sample) will not 
            appear in the legend.

        hide_scatter : bool, optional
            If true, data scatter plot will not render

        traj_markers : list of strings, optional
            List of markers to use for each trajectory's line plot. Length of
            list should match number of trajectories to plot.

        traj_colors : list of strings, optional
            List of colors to use for each trajectory's line plot. Length of
            list should match number of trajectories to plot.

        fill_alpha : float, optional
            Value between 0 and 1 that controls opacity of each trajectorys 
            fill region (which indicates +\- 2 residual standard deviations 
            about the mean)
        """
        # Compute the probability vector for each trajectory
        traj_probs = self.get_traj_probs()
        
        df_traj = self.to_df()
            
        num_dom_locs = 100
        x_dom = np.linspace(np.min(df_traj[x_axis].values),
                            np.max(df_traj[x_axis].values),
                            num_dom_locs)
    
        target_index = np.where(np.array(self.target_names_) == y_axis)[0][0]
    
        X_tmp = np.ones([num_dom_locs, self.M_])
        for (inc, pp) in enumerate(self.predictor_names_):
            tmp_pow = pp.split('^')
            tmp_int = pp.split('*')
            
            if len(tmp_pow) > 1:
                if x_axis in tmp_pow:                
                    X_tmp[:, inc] = x_dom**(int(tmp_pow[-1]))
                else:                
                    X_tmp[:, inc] = np.mean(df_traj[tmp_pow[0]].values)**\
                        (int(tmp_pow[-1]))
            elif len(tmp_int) > 1:
                if x_axis in tmp_int:                
                    X_tmp[:, inc] = \
                        x_dom**np.mean(df_traj[tmp_int[np.where(\
                            np.array(tmp_int) != x_axis)[0][0]]].values)
                else:
                    X_tmp[:, inc] = np.mean(df_traj[tmp_int[0]])*\
                        np.mean(df_traj[tmp_int[1]])                    
            elif pp == x_axis:
                X_tmp[:, inc] = x_dom
            else:
                X_tmp[:, inc] = np.nanmean(df_traj[tmp_pow[0]].values)

        # Create a trajeectory mapping for internal uses. By default, this is
        # the trivial mapping whereby every trajectory maps to itself. Using
        # this trajectory mapping consistently will facilitate the use case
        # when a user specifies a specific mapping. Note that traj_map_ is only
        # for plotting color selection and legend numbering
        traj_map_ = {}
        for ii in range(self.K_):
            traj_map_[ii] = ii
                
        if traj_map is not None:
            traj_ids = np.array(list(traj_map.keys()))
            traj_map_ = traj_map
        elif which_trajs is not None:
            if type(which_trajs) == int:
                traj_ids = np.array([which_trajs])
            else:
                traj_ids = which_trajs
        else:
            traj_ids = np.where(self.sig_trajs_)[0]
    
        cmap = plt.cm.get_cmap('tab20')
            
        # The following just maps trajectories to sequential integers starting
        # at 0. Otherwise, trajectory numbers greater than 19 will all be given
        # the same color. With the chosen colormap, we still only have access
        # to 20 unique colors, but this should suffice in most cases.
        # If a traj_map is specified, there will be a one-to-one mapping between
        # the mapped values and colors        
        traj_id_to_cmap_index = {}
        if traj_map is not None:
            for vv in traj_map.values():
                traj_id_to_cmap_index[vv] = vv
        else:
            for (ii, tt) in enumerate(np.where(self.sig_trajs_)[0]):
                traj_id_to_cmap_index[tt] = ii
            
        fig, ax = plt.subplots(figsize=(6, 6))
        if not hide_scatter:
            ax.scatter(df_traj[x_axis].values,
                       df_traj[y_axis].values,
                       edgecolor='k', color='None', alpha=0.1)

        if torch.is_tensor(self.lambda_a_):
            lambda_a = self.lambda_a_.numpy()
            lambda_b = self.lambda_b_.numpy()
        else:
            lambda_a = self.lambda_a_
            lambda_b = self.lambda_b_
            
        for (traj_inc, tt) in enumerate(traj_ids):
            if traj_probs[tt] >= min_traj_prob and \
               traj_probs[tt] <= max_traj_prob:
                
                ids_tmp = df_traj.traj.values == tt
                if not hide_scatter:
                    if traj_colors is not None:
                        color = traj_colors[traj_inc]
                    else:
                        color = cmap(traj_id_to_cmap_index[traj_map_[tt]])
                    ax.scatter(df_traj[ids_tmp][x_axis].values,
                               df_traj[ids_tmp][y_axis].values,
                               edgecolor='k',
                               color=color,
                               alpha=0.5)

                if self.gb_ is None:
                    n_traj = np.sum(df_traj.traj.values == tt)
                    perc_traj = 100*n_traj/df_traj.shape[0]
                else:
                    groupby_col = self.gb_.count().index.name                
                    n_traj = df_traj[df_traj.traj.values == tt].\
                        groupby(groupby_col).ngroups
                    perc_traj = 100*n_traj/self.gb_.ngroups

                co = self.w_mu_[:, target_index, tt]
                if self.target_type_[target_index] == 'gaussian':
                    std = np.sqrt(lambda_b[target_index][tt]/\
                                  lambda_a[target_index][tt])
                    y_tmp = np.dot(co, X_tmp.T)

                    if traj_colors is not None:
                        color = traj_colors[traj_inc]
                    else:
                        color = cmap(traj_id_to_cmap_index[traj_map_[tt]])
                    
                    ax.fill_between(x_dom, y_tmp-2*std, y_tmp+2*std,
                            color=color, alpha=fill_alpha)
                else:
                    # Target assumed binary
                    y_tmp = np.exp(np.dot(co, X_tmp.T))/\
                        (1 + np.exp(np.dot(co, X_tmp.T)))

                if hide_traj_details:
                    label = 'Traj {}'.format(traj_map_[tt])
                else:
                    label = 'Traj {} (N={}, {:.1f}%)'.\
                        format(traj_map_[tt], n_traj, perc_traj)

                marker = None
                if traj_markers is not None:
                    marker = traj_markers[traj_inc]

                if traj_colors is not None:
                    color = traj_colors[traj_inc]
                else:
                    color = cmap(traj_id_to_cmap_index[traj_map_[tt]])
                ax.plot(x_dom, y_tmp,
                        color=color,
                        linewidth=3,
                        label=label, marker=marker, ms=8, markevery=5)

        
        ax.set_xlabel(x_axis if x_label is None else x_label, fontsize=16)
        ax.set_ylabel(y_axis if y_label is None else y_label, fontsize=16)    
        plt.tight_layout()
        ax.legend(loc='upper right', framealpha=1)

        if show:
            plt.show()

        return ax        

    def to_df(self) -> pd.DataFrame:
        """
        Reconstitutes the contents of the restructured data into a DataFrame.
        
        Returns
        -------
        pd.DataFrame
            A DataFrame containing the original data.
        """
        data = {}
        for pp in self.predictor_names_:
            data[pp] = np.array([])
        for tt in self.target_names_:
            data[tt] = np.array([])            
        data[self.groupby_col_] = []
        
        if self.cohort is not None:
            data['cohort'] = []
            
        for jj, (gg, idx) in enumerate(self.group_to_index.items()):
            for ii, mm in enumerate(self.predictor_names_):                
                data[mm] = np.append(data[mm], self.X[0:idx.shape[0], jj, ii])

            for ii, rr in enumerate(self.real_target_names_):                
                data[rr] = np.append(data[rr],
                                     self.Y_real[0:idx.shape[0], jj, ii])

            for ii, bb in enumerate(self.bool_target_names_):                
                data[bb] = np.append(data[bb],
                                     self.Y_bool[0:idx.shape[0], jj, ii])

            data[self.groupby_col_] = data[self.groupby_col_] + \
                [gg]*idx.shape[0]

            if self.cohort is not None:
                data['cohort'] = data['cohort'] + [self.cohort[jj]]*idx.shape[0]
            
        df_tmp = pd.DataFrame(data)
        df = self.augment_df_with_traj_info(df_tmp)
        
        return df
    

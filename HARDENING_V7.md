# Structured-DP / random-effect hardening (v7 development patch)

This development patch hardens the Gaussian longitudinal random-effect path in
`bayes_traj`.  It is intentionally bounded: it does not add new likelihood
families, graph regularization, finite-K mixtures, or inference for the DP
concentration parameter.

## Corrected structured random-effect inference

For Gaussian targets with random effects, the corrected implementation uses

`q(z_i) q(b_i | z_i)`

rather than the historical independent mean-field `q(z_i)q(b_i)`.  The local
random-effect posterior for every subject and candidate trajectory is updated
without multiplying its precision/information by the subject's current class
responsibility.  The corresponding class score includes the expected Gaussian
log likelihood and the KL term for the class-conditional random effect.

A historical group-order bug was also corrected.  Older code sometimes used a
boolean first-row mask to extract one responsibility vector per subject; that
produced dataframe-row order, whereas the random-effect state was stored in
pandas GroupBy order.  Group-level operations now use an explicit ordered
representative-row mapping.  Compatibility code reconstructs this mapping for
older pickles when possible.

## DP truncation and occupancy

In structured mode, `K` is a truncation ceiling.  Components are not hard-pruned
by `prob_thresh`, and `sig_trajs_` therefore no longer means "occupied
trajectories".  Reportable trajectories are determined from posterior effective
membership.  Model objects expose diagnostics including:

- effective membership `sum_i q(z_i=k)`;
- posterior probability that each component is occupied;
- expected occupied K;
- MAP occupied K;
- expected stick weights;
- residual/tail stick mass.

Utilities for summaries, priors, visualization, and external assignment use
these occupancy semantics for structured fits.

## ELBO, convergence, and repeat selection

Structured fitting records a valid ELBO and separate convergence concepts:

- **objective convergence**: relative ELBO change below tolerance;
- **parameter convergence**: responsibility/coefficient/precision/covariance
  changes below their stricter tolerances;
- **joint convergence**: both conditions.

Objective convergence is the practical fit-completion criterion.  Parameter
convergence remains a QC diagnostic.

Repeated fits of the *same* specification are selected only after eligibility
QC.  Eligible repeats must have a finite, objective-converged, complete fit;
normalized finite responsibilities; valid variational/global parameters;
positive-definite estimated random-effect covariance; and no materially
negative ELBO steps.  The eligible repeat with the highest final ELBO is
selected.  ELBO is not intended here as a general criterion for comparing
models with different priors, truncations, or likelihood specifications.

A repeat-summary CSV records selection status, eligibility failures, convergence
metrics, occupancy/tail diagnostics, and the fit seed.  If no repeat is
eligible, no model is silently selected.

## Random-effect covariance D

`ranef_cov_mode` supports:

- `fixed`: hold D fixed;
- `estimate`: empirical-Bayes/point-estimate D during fitting;
- `staged`: optimize with D fixed until objective convergence, then release D
  and continue with safeguarded covariance updates.

In staged mode `--iters` is a **per-phase** maximum: up to that many fixed-D
iterations followed by up to that many estimated-D iterations.  The model
records the initial D, release iteration, fixed-phase terminal ELBO, final D,
and final convergence status.  D updates are constrained positive definite and
rejected if they materially lower the structured ELBO.

D is a point estimate, not a variational random variable with its own prior.

## External/test-cohort inference

`assign_trajectory` and `MultDPRegression.infer_new_data` support two explicit
structured modes:

- `strict` (default): freeze all trained global quantities, including source
  stick weights, and infer only new-subject `q(z_i)` and `q(b_i | z_i)`;
- `adapt_prevalence`: freeze trajectory coefficients, residual precisions, and
  D, but estimate a new cohort's stick posterior together with local
  assignments/random effects.

Strict inference is the appropriate default for external validation.  Neither
mode mutates the fitted training model.  The API exposes both class-conditional
random-effect posteriors and posterior-marginal random-effect moments averaged
over trajectory uncertainty.

Target columns may be absent in a test dataframe and are treated as entirely
missing evidence; predictor columns remain required.  This is forward-compatible
with future models that contain additional target dimensions.

### Future multimodal/qCT note

The current random-effect predictor mask is shared across Gaussian target
variables.  Before using a cross-sectional qCT variable as a target that needs a
different random-effect structure, target-specific random-effect masks should be
implemented.  This patch deliberately does not add that feature before a
concrete scientific use case requires it.

## Model summaries and fit statistics

`summarize_traj_model` now reports group-level (subject-level) assignment
metrics rather than weighting subjects by number of visits.  It includes
posterior entropy, max-posterior summaries, AvePP, OCC, posterior/MAP class
sizes, occupied-K/tail diagnostics, and structured convergence state.

For all-Gaussian models it also reports **plug-in marginal** log likelihood and
AIC/BIC/SABIC/ICL-style summaries.  Gaussian random effects are analytically
integrated, while fitted global parameters are held at posterior means/expected
precisions.  These are useful descriptive/model-comparison summaries but are
not a full Bayesian marginal likelihood and should not be assumed numerically
identical to criteria from another mixed-model implementation.

Historical WAIC remains available as an optional diagnostic through
`summarize_traj_model --compute_waic`. It is off by default and is not used to
select structured restarts.

## Legacy behavior

Historical mean-field random-effect inference is retained for backward
compatibility and regression testing, but it is not the recommended
random-effect inference path.  The new structured ELBO/convergence machinery is
implemented for the corrected structured Gaussian path.

# bayes_traj structured random-effect core patch

This patch promotes the experimental structured Gaussian random-effect
factorization into `MultDPRegression` while preserving historical behavior by
default.

## New core options

- `ranef_factorization='mean_field'` (default): historical bayes_traj behavior.
- `ranef_factorization='structured'`: uses `q(z_i) q(b_i | z_i)`.
- `ranef_cov_mode='fixed'` (default): use the supplied covariance unchanged.
- `ranef_cov_mode='estimate'`: update a common random-effect covariance per
  Gaussian outcome using the responsibility-weighted posterior second moment.
- Optional structured convergence controls use the structured ELBO.

Structured mode currently supports Gaussian targets only.

## Important behavioral change in structured mode

For new grouped data, trajectory probabilities are computed after analytically
conditioning the subject random effects within every candidate trajectory.
Random effects are no longer forced to zero for new/test longitudinal subjects.

## CLI flags

    --ranef_factorization structured
    --ranef_cov_mode fixed|estimate
    --ranef_cov_min_eig 1e-8
    --structured_tol_elbo_rel 1e-7
    --structured_tol_r 1e-5
    --structured_tol_w 1e-5
    --structured_tol_lambda 1e-5
    --structured_tol_ranef_cov 1e-5
    --structured_min_iters 5
    --structured_r_damping 1.0

If `--structured_tol_elbo_rel` is omitted, the requested `--iters` remains the
stopping rule.

## Validation performed

- 8 focused core tests pass.
- Historical `get_R_matrix` output on the frozen R01 model is bit-for-bit
  identical between the original source and this patch when default mean-field
  mode is used.
- On synthetic Gaussian data, the structured core implementation matches the
  previously validated standalone prototype to numerical precision after two
  iterations:
  - max responsibility difference: 4.44e-16
  - max coefficient-mean difference: 3.33e-16
  - max expected-precision difference: 1.07e-14
  - ELBO identical.
- With estimated random-effect covariance, the synthetic ELBO is monotone over
  the tested iterations and the covariance remains positive definite.
- On the frozen R01 model without refitting global parameters, the integrated
  core structured update reproduces the prior diagnostic for subject
  201706125378:
  - stored 15%-scale D: FEV1 RE slope +0.000661/y, FVC -0.002396/y
  - full-R-scale D: FEV1 +0.009939/y, FVC +0.000154/y

## Files

- `mult_dp_regression.py`: replacement core implementation.
- `bayes_traj_main.py`: replacement CLI with structured options.
- `test_mult_dp_regression_structured.py`: focused regression/analytic tests.
- `mult_dp_regression_structured.patch`: unified diff against the supplied
  project source.

Review/diff before replacing production files.

## Provenance API compatibility fix

All `bayes_traj_main.py` provenance writes now use the installed-project API:

    write_provenance_data(output_path, generator_args=op)

The unsupported `module_name='bayes_traj'` keyword has been removed. No
TypeError fallback is used, so real provenance failures are not hidden.

## v3 bug fix

Fixed verbose structured fitting with `ranef_cov_mode='estimate'`.

Because the legacy module imports `numpy.max` into the global namespace,
`max(generator)` returned the generator object instead of reducing it. The
covariance-change diagnostic now explicitly uses `builtins.max`. A regression
test exercises structured + estimated covariance + verbose output.


## v4: DP occupancy and repeat selection

Structured mode now restores the intended truncated-DP semantics:

- `-k` is a truncation ceiling, not the requested final number of trajectories.
- All truncated components remain eligible throughout structured VI; the
  historical `prob_thresh` hard-zeroing and irreversible `sig_trajs_` pruning
  are not used in structured mode.
- `num_init_trajs` is only an initialization hint. It does not constrain final
  posterior occupancy.
- If a structured prior contains fewer trajectory slots than `-k`, its
  trajectory-indexed initialization arrays are padded to the requested
  truncation ceiling. Existing prior-informed components are retained; added
  slots receive neutral stick initialization and randomizable trajectory
  parameters.
- Multiple structured repeats are selected by the highest final ELBO. WAIC2 is
  optional diagnostic output (`--structured_compute_waic`) and does not choose
  the winning structured repeat. Historical mean-field mode retains WAIC2.
- A per-repeat summary CSV is written by default next to `--out_model` (or can
  be set explicitly with `--repeat_summary`).
- Posterior occupancy is reported without feeding a threshold back into VI:
  expected occupied K, MAP occupied K, expected membership by component,
  probability each component is occupied, posterior mean stick weights, and
  truncation-tail diagnostics.

### Estimated random-effect covariance safeguard

`ranef_cov_mode=estimate` now defaults to a 10-iteration covariance warm-up
(`--ranef_cov_warmup_iters`). Each subsequent covariance block is checked
against the structured ELBO before and after re-optimizing the local random
effects; a covariance candidate that lowers that block objective is rejected.
This is a numerical safeguard, not a prior preventing large covariance values.

### Provenance

Provenance calls are executed with the package git checkout as the temporary
working directory, so saving outputs outside the repository does not cause
GitPython to search from the output directory.


## v5: group-order alignment bug fix

The first K=15 smoke test exposed a substantive historical indexing bug.

`group_first_index_` is a boolean dataframe-row mask. Boolean indexing returns
representative subjects in dataframe row order. `u_mu_`, `G_r_r_`, and
`N_to_G_index_map_`, however, follow pandas GroupBy iteration order. Those
orders are not guaranteed to match and do not match for the pooled
CARDIA/COPDGene data.

This affected calculations that paired `R_[group_first_index_]` with
G-dimensional state, including:
- historical mean-field random-effect updates,
- random-effect recentering,
- structured covariance estimation,
- structured ELBO calculation/convergence diagnostics,
- WAIC trajectory sampling when group probabilities were mapped back to rows.

v5 retains `group_first_index_` for backward compatibility, adds an ordered
`group_first_row_by_group_`, and centralizes aligned extraction through
`_get_group_responsibilities()`.

For the uploaded K=15 fixed-D model, the fitted posterior remains diagnostically
useful because the main structured local/global updates already used
`N_to_G_index_map_`. The recorded v4 ELBO history is not valid. At the uploaded
final state:
- v4-reported ELBO was about -59,272.76;
- corrected G-order ELBO is about -22,568.89;
- a fresh q(z) coordinate update increases the corrected ELBO as required.

This bug also changes interpretation of the earlier R comparison. Holding the
frozen model globals fixed and merely correcting group alignment in the
historical mean-field RE update raises mapped random-slope correlation with
mpjlcmm to about 0.75 for both FEV1 and FVC at the stored 15%-scale covariance.
The remaining magnitude shrinkage is still substantial. Thus some of the
apparent structured-factorization benefit was actually correction of this
group-order bug.


## v6: staged continuation and explicit convergence diagnostics

Structured fits now distinguish objective convergence from parameter-change
convergence:

- `objective_converged_`: final relative ELBO increment is below
  `structured_tol_elbo_rel` (or `None` when no ELBO tolerance is requested).
- `parameter_converged_`: responsibility, coefficient, residual-precision, and
  random-effect-covariance changes all meet their requested tolerances.
- `converged_`: both conditions are met. This remains the strict/backward
  compatible joint flag.

Every structured repeat-summary row now records the final ELBO increment,
relative increment, dR, dW, dLambda, dD, MAP stability, negative-ELBO-step
count, and the three convergence statuses.

A saved structured model can be continued without reinitialization using
`--resume_model`. `--iters` is then interpreted as *additional* iterations,
history is appended, and iteration numbering continues. `--resume_reset_history`
is available when a fresh history segment is desired.

For the planned staged covariance experiment, continue the selected fixed-D
model with `--ranef_cov_mode estimate --ranef_cov_warmup_iters 0`. This lets D
start moving from an already-stable fixed-covariance solution rather than from a
randomly initialized global model.

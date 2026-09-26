# Introduction

**bayes_traj** is a software package written in Python that provides
routines for performing Bayesian trajectory
modeling of longitudinal data. Multiple, longitudinally observed target
variables -- continuous, binary, or a combination -- can be modeled
simultaneously. Per-trajectory random effects can also be modeled for
continuous target variables. For Gaussian longitudinal random-effect models,
the recommended implementation uses structured variational inference
`q(z_i)q(b_i|z_i)`, posterior DP occupancy rather than hard component pruning,
and an ELBO-based convergence/quality-control workflow. This package also provides command-line tools
that facilitate spefication of Bayesian priors, enable visualization
of trajectory modeling results, and compute summary and model
fit statistics. 

# Installation

In order to install the package, type the folowing in the terminal:

    $ pip install bayes_traj

# Overview

**bayes_traj** provides several command-line tools: 

* `generate_prior` -- used to speficy Bayesian priors for use the trajectory
  modeling
* `viz_data_prior_draws` -- provides visualization of random draws from the
  prior
* `bayes_traj_main` -- performs Bayesian trajectory modeling using a prior file
* `viz_model_trajs` -- provides visualization of trajectories fit using
  `bayes_traj_main`
* `summarize_traj_model` -- prints subject-level assignment diagnostics,
  posterior occupancy/convergence information, and plug-in Gaussian model fit
  statistics given a model file produced by `bayes_traj_main`. Historical WAIC
  is available only when requested with `--compute_waic`.
* `assign_trajectory` -- applies a fitted model to new subjects. Structured
  models infer new-subject trajectory probabilities and local random effects
  with trained global parameters frozen by default.

Each of these tools can be run with the -h flag for additional usage information.

For additional documentation, see https://acil-bwh.github.io/bayes_traj/index.html

# Tests

To run all unit tests, type the following in the package root directory:

    $ pytest


# Contribute

Please read our [contribution guidelines](./CONTRIBUTING.md).


## Structured random-effect hardening

The current development implementation adds structured Gaussian random-effect
inference, staged estimation of the population random-effect covariance,
posterior DP occupancy diagnostics, robust ELBO-based repeat selection, and
strict/adaptive external-cohort inference. See
[`HARDENING_V7.md`](./HARDENING_V7.md) for behavior, compatibility notes, and
current limitations.

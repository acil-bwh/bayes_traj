# Welcome to **bayes_traj** documentation!

**bayes_traj** is a Python package developed to perform Bayesian trajectory
analysis, and it has several features that distinguish it from other
trajectory approaches:

* It is fully Bayesian, so it supports incorporation of prior knowledge into
the modeling process.
* It can simultaneously model multiple continuous and binary target
variables as a functions of predictor variables.
* It uses Bayesian nonparametrics to identify the number of trajectories best
supported by the data.
* Inference is performed using coordinate ascent variational inference, so
it is fast and scales well to large data sets. For Gaussian longitudinal
random-effect models, structured `q(z_i)q(b_i|z_i)` inference is recommended.
* Independantly estimates residual variance for each trajectory and
each target variable.
* Can be used without random effects (group-based trajectory modeling, or
GBTM), or with random effects (growth mixture modeling, or GMM). Structured
Gaussian fits can keep the population random-effect covariance fixed or estimate
it with a staged fixed-D then released-D workflow.
* Provides a suite of tools to facilitate prior specification, model
visualization, and summary statistic computation. 


### Installation

To get started using bayes_traj, simply install it using pip by running the
following in your terminal:
```bash
pip install bayes_traj
```

**bayes_traj** provides several command-line tools: 

* `generate_prior` -- used to speficy Bayesian priors for use the trajectory
  modeling
* `viz_data_prior_draws` -- provides visualization of random draws from the
  prior
* `bayes_traj_main` -- performs Bayesian trajectory modeling using a prior file
* `viz_model_trajs` -- provides visualization of trajectories fit using
  `bayes_traj_main`
* `summarize_traj_model` -- prints subject-level assignment, posterior
  occupancy/convergence, and fit statistics for a model produced by
  `bayes_traj_main`
* `assign_trajectory` -- applies a fitted model to new data; structured models
  infer new-subject assignments and random effects with global trajectory
  parameters frozen by default.

Each of these tools can be run with the -h flag for additional usage information.


### Documentation Contents

* [Tutorial](bayes_traj_tutorial.md) -- provides a walk-through of basic usage
* [Formulation](formulation.md) -- states the Bayesian model assumptions and
formulation
* [Inference](inference.md) -- details the variational inference update equations
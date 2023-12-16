# bayes_traj

Algorithms and tools for Bayesian trajectory modeling

```sh
pip install provenance_tools
pip install bayes_traj
```

## Overview of code

* `mult_dp_regression.py` is the current non-pyro model
  "multivariate Dirichlet process regression"
* `generate_prior.py` creates a prior file.
* `viz_data_prior_draws.py` prior predictive checks.
* `bayes_traj_main.py --out_model PICKLE_FILE` saves a trajectory model
  configured by other tools.
  This inputs a prior file.
* `viz_model_trajs.py` is posterior predictive check,
  visualizes the learned clusters.
* `sumarize_traj_model.py` prints additional info.
* `assign_trajectory.py` reads in the model + data.csv and outputs a
  data.csv file with one appended column that is the assigned trajectory.

Other files are legacy.

## Running tests

```sh
pytest
```
or when debugging
```sh
pytest -vsx --pdb
```

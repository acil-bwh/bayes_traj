# Introduction
This tutorial demonstrates the basic procedure for performing Bayesian
trajectory modeling using the *bayes_traj* python package. No
knowledge of the python programming language is required.

# Data Preliminaries
### General comments about input data
* Tools expect data to be in csv (comma-separated value) format
* If an intercept term will be used as a predictor, the data set should contain
a column of 1s
* There should be a subject identifier column (not strictly necessary, e.g. if
your data is cross-sectional)
* bayes_traj uses '^' to indicate a predictor is being raised to a power
and '\*' to indicate an interaction between two predictors. 

### Demonstration Using Synthetically Generated Data
We begin by describing a synthetically generated data set that mimics biomarkers
(y1 and y2) that decline with age. There are five trajectory subgroups in this
data set: three with modest rates of decline but with varying intercepts, and
two with accelerated rates of decline (also with varying intercepts). There are
three visits for each individual in our data set, simulating a longitudinal
study. The visits are spread 5 years apart. Subject "enrollment" is 45 - 80
years old. 

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>intercept</th>
      <th>age</th>
      <th>age^2</th>
      <th>id</th>
      <th>data_names</th>
      <th>traj_gt</th>
      <th>y1</th>
      <th>y2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>68.193104</td>
      <td>4650.299429</td>
      <td>1</td>
      <td>1_0</td>
      <td>1</td>
      <td>4.874896</td>
      <td>5.007974</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>73.193104</td>
      <td>5357.230469</td>
      <td>1</td>
      <td>1_1</td>
      <td>1</td>
      <td>4.972271</td>
      <td>4.903405</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>78.193104</td>
      <td>6114.161509</td>
      <td>1</td>
      <td>1_2</td>
      <td>1</td>
      <td>4.708326</td>
      <td>4.821643</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
      <td>54.138515</td>
      <td>2930.978773</td>
      <td>2</td>
      <td>2_0</td>
      <td>1</td>
      <td>5.302883</td>
      <td>5.193132</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.0</td>
      <td>59.138515</td>
      <td>3497.363920</td>
      <td>2</td>
      <td>2_1</td>
      <td>1</td>
      <td>5.206097</td>
      <td>5.129398</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1.0</td>
      <td>64.138515</td>
      <td>4113.749067</td>
      <td>2</td>
      <td>2_2</td>
      <td>1</td>
      <td>4.978147</td>
      <td>5.065314</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1.0</td>
      <td>58.282453</td>
      <td>3396.844384</td>
      <td>3</td>
      <td>3_0</td>
      <td>1</td>
      <td>5.184020</td>
      <td>5.059987</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1.0</td>
      <td>63.282453</td>
      <td>4004.668919</td>
      <td>3</td>
      <td>3_1</td>
      <td>1</td>
      <td>5.035844</td>
      <td>4.983862</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1.0</td>
      <td>68.282453</td>
      <td>4662.493453</td>
      <td>3</td>
      <td>3_2</td>
      <td>1</td>
      <td>5.073619</td>
      <td>4.895124</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1.0</td>
      <td>59.795287</td>
      <td>3575.476391</td>
      <td>4</td>
      <td>4_0</td>
      <td>1</td>
      <td>5.058325</td>
      <td>5.045555</td>
    </tr>
  </tbody>
</table>
</div>


Visualizing the data:
    
![png](bayes_traj_tutorial_files/bayes_traj_tutorial_6_0.png)
    
This is a highly idealized data set. Working with such a data set initially has
two purposes: 1) it makes clear what we are trying to achieve with trajectory
analysis (namely, to identify subgroups and their characteristic progression
patterns), and 2) it provides a "sanity check" that ensures the Bayesian
trajectory tools produce the results we expect. We will explore more challenging
data later in the tutorial.

# Generating Priors
Before we can perform Bayesian trajectory fitting to our data, we must first
generate priors for the parameters in our model. The *bayes_traj* package
provides two utilities for generating priors: *generate_prior* and
*viz_data_prior_draws*. We can inspect tool usage by running each with the -h flag:

```python
> generate_prior -h
```

    usage: generate_prior [-h] [--preds PREDS] [--targets TARGETS]
                          [--out_file OUT_FILE]
                          [--tar_resid TAR_RESID [TAR_RESID ...]]
                          [--coef COEF [COEF ...]]
                          [--coef_std COEF_STD [COEF_STD ...]] [--in_data IN_DATA]
                          [--num_trajs NUM_TRAJS] [--model MODEL]
                          [--model_trajs MODEL_TRAJS] [--groupby <string>]
                          [--alpha <class 'float'>]
    
    Generates a pickled file containing Bayesian trajectory prior information
    
    options:
      -h, --help            show this help message and exit
      --preds PREDS         Comma-separated list of predictor names
      --targets TARGETS     Comma-separated list of target names
      --out_file OUT_FILE   Output (pickle) file that will contain the prior
      --tar_resid TAR_RESID [TAR_RESID ...]
                            Use this flag to specify the residual precision mean
                            and variance for the corresponding target value.
                            Specify as a comma-separated tuple:
                            target_name,mean,var. Note that precision is the
                            inverse of the variance. Only applies to continuous
                            targets
      --coef COEF [COEF ...]
                            Coefficient prior for a specified target and
                            predictor. Specify as a comma-separated tuple:
                            target_name,predictor_name,mean,std
      --coef_std COEF_STD [COEF_STD ...]
                            Coefficient prior standard deviation for a specified
                            target and predictor. Specify as a comma-separated
                            tuple: target_name,predictor_name,std
      --in_data IN_DATA     If a data file is specified, it will be read in and
                            used to set reasonable prior values using regression.
                            It is assumed that the file contains data columns with
                            names corresponding to the predictor and target names
                            specified on the command line.
      --num_trajs NUM_TRAJS
                            Estimate of the number of trajectories expected in the
                            data set. Can be specified as a single value or as a
                            dash-separated range, such as 4-6. If a single value
                            is specified, a range will be assumed to be 1 to
                            (2*num_trajs-1)
      --model MODEL         Pickled bayes_traj model that has been fit to data and
                            from which information will be extracted to produce an
                            updated prior file
      --model_trajs MODEL_TRAJS
                            Comma-separated list of integers indicating which
                            trajectories to use from the specified model. If a
                            model is not specified, the values specified with this
                            flag will be ignored. If a model is specified, and
                            specific trajectories are not specified with this
                            flag, then all trajectories will be used to inform the
                            prior
      --groupby <string>    Column name in input data file indicating those data
                            instances that must be in the same trajectory. This is
                            typically a subject identifier (e.g. in the case of a
                            longitudinal data set).
      --alpha <class 'float'>
                            Dirichlet process scaling parameter. Higher values
                            indicate belief that more trajectoreis are present.
                            Must be a positive real value if specified.


This utility can be used in a number of ways. If you have no prior knowledge
about how to set values for the prior, you may wish to start by running the
utility with very basic information, which can later be fine-tuned.

Let's run this utility with some basic information: the target variable we wish
to analyze (y1), the predictors we wish to use (intercert and age), and our data
file.

```python
> generate_prior --num_trajs 5 --preds intercept,age --targets y1 --in_data bayes_traj_tutorial_std-0.05_visits-3.csv  --out_file bayes_traj_tutorial_std-0.05_visits-3_prior_v1.p --groupby id
```

    Reading data...
    ---------- Prior Info ----------
    alpha: 5.92e-01
     
    y1 residual (precision mean, precision variance):             (3.75e+00, 8.35e-02)
    y1 intercept (mean, std): (5.10e+00, 5.03e-01)
    y1 age (mean, std): (-2.74e-02, 7.46e-03)


By default, this utility makes a crude estimate of prior parameters.

The print-out provides information about how the prior has been set. The first
value, 'alpha', captures our prior belief about how many trajectories are
likely to exist in our data set. This value is determined by number specified
using the --num_trajs flag. Higher alpha values indicate more trajectories are
likely to be in the data and vice versa (this value is always greater than
zero). Note that the actual number of trajectories will be determined during
the data fitting process; the specified number of trajectories only represents
an expectation.

Next is the prior for the residual precision (1/variance). Following this are
priors for the trajectory predictor coefficients.

Is this a good prior? Does it reflect our believe about trajectories that may be
in our data? This can be difficult to assess with a numerical description. For a
more visual assesment, we can use the 'viz_data_prior_draws' to produce random
draws from this prior and overlay these with our data. Let's first look at usage
by running with the -h flag:

```python
> viz_data_prior_draws -h
```

    usage: viz_data_prior_draws [-h] [--data_file DATA_FILE] [--prior PRIOR]
                                [--num_draws NUM_DRAWS] [--y_axis Y_AXIS]
                                [--y_label Y_LABEL] [--x_axis X_AXIS]
                                [--x_label X_LABEL] [--ylim YLIM] [--hide_resid]
                                [--fig_file FIG_FILE]
    
    Produces a scatter plot of the data contained in the input data file as well
    as plots of random draws from the prior. This is useful to inspect whether the
    prior appropriately captures prior belief.
    
    options:
      -h, --help            show this help message and exit
      --data_file DATA_FILE
                            Input data file
      --prior PRIOR         Input prior file
      --num_draws NUM_DRAWS
                            Number of random draws to take from prior
      --y_axis Y_AXIS       Name of the target variable that will be plotted on
                            the y-axis
      --y_label Y_LABEL     Label to display on y-axis. If none given, the
                            variable name specified with the y_axis flag will be
                            used.
      --x_axis X_AXIS       Name of the predictor variable that will be plotted on
                            the x-axis
      --x_label X_LABEL     Label to display on x-axis. If none given, the
                            variable name specified with the x_axis flag will be
                            used.
      --ylim YLIM           Comma-separated tuple to set the limits of display for
                            the y-axis
      --hide_resid          If set, shaded regions corresponding to residual
                            spread will not be displayed. This can be useful to
                            reduce visual clutter. Only relevant for continuous
                            target variables.
      --fig_file FIG_FILE   File name where figure will be saved



```python
> viz_data_prior_draws --data_file bayes_traj_tutorial_std-0.05_visits-3.csv --prior bayes_traj_tutorial_std-0.05_visits-3_prior_v1.p --num_draws 2 --x_axis age  --y_axis y1
```
    
![png](bayes_traj_tutorial_files/bayes_traj_tutorial_14_2.png)
    
Shown are two random draw from our prior over trajectories. The solid, colored
lines represent the mean trend of randomly selected trajectories, and the shaded
regions reflect the prior belief about the residual spread around each
trajectory.

There are some issues with this first-pass prior: the prior for residual
variances (shaded regions), is much too large. Also, the variability in
intercepts does not appear to be high enough to adequately represent our data.

Let's rerun generate_prior, but this time will specify a higher residual
precision (i.e., lower residual standard deviation value). Because we are
dealing with a synthetically generated data set, we know *a priori* what the
residual standard deviation is for each trajectory (we set it when we created
the data!). In practice, you will need to use trial-and-error to select a value
that best captures your belief.

We will use the --tar_resid flag to over-ride the default residual prior settings:

```python
> generate_prior --num_trajs 5 --preds intercept,age --targets y1 --in_data bayes_traj_tutorial_std-0.05_visits-3.csv --out_file bayes_traj_tutorial_std-0.05_visits-3_prior_v2.p --groupby id --tar_resid y1,400,1e-5
```

    Reading data...
    ---------- Prior Info ----------
    alpha: 5.92e-01
     
    y1 residual (precision mean, precision variance):             (4.00e+02, 1.00e-05)
    y1 intercept (mean, std): (5.10e+00, 5.03e-01)
    y1 age (mean, std): (-2.74e-02, 7.46e-03)


As before, let's visualize some random draws from this prior to see how things
look:

```python
> viz_data_prior_draws --data_file bayes_traj_tutorial_std-0.05_visits-3.csv --prior bayes_traj_tutorial_std-0.05_visits-3_prior_v2.p --num_draws 20 --x_axis age --y_axis y1 
```
    
![png](bayes_traj_tutorial_files/bayes_traj_tutorial_18_2.png)
    
Now our prior over the residual variance seems reasonable. We can further
improve the prior by overriding the settings for the intercept using
the --coef_std flag. Let's see how this works:

```python
> generate_prior --num_trajs 5 --preds intercept,age --targets y1 --in_data bayes_traj_tutorial_std-0.05_visits-3.csv --out_file bayes_traj_tutorial_std-0.05_visits-3_prior_v3.p --groupby id --tar_resid y1,400,1e-5 --coef_std y1,intercept,1 
```

    Reading data...
    ---------- Prior Info ----------
    alpha: 5.92e-01
     
    y1 residual (precision mean, precision variance):             (4.00e+02, 1.00e-05)
    y1 intercept (mean, std): (5.10e+00, 1.00e+00)
    y1 age (mean, std): (-2.74e-02, 7.46e-03)


Now our prior over the intercept has been adjustes from a Gaussian distribution
with a mean of 5.1 and a standard deviation of 0.074 to a mean of 5.1 and
standard deviation of 1. Let's again look at some draws from the prior:

```python
> viz_data_prior_draws --data_file bayes_traj_tutorial_std-0.05_visits-3.csv --prior bayes_traj_tutorial_std-0.05_visits-3_prior_v3.p --num_draws 100 --hide_resid --x_axis age --y_axis y1
```
    
![png](bayes_traj_tutorial_files/bayes_traj_tutorial_22_2.png)
    
The increased variance for the intercept coefficient now better captures what
we observe in our data. We could further tweak the prior by adjusting the
coefficient for 'age' (i.e. the slope), but the trajectory subgroups are so well
delineated in this idealized data set, the fitting routine should have no
problem with this prior.

Now that we have a reasonable prior, we can proceed with the actual Bayesian
trajectory fitting and analysis.

# Bayesian Trajectory Analysis: Continuous Target Variables
Now that we have generated a prior for our data set, we are ready to perform
Bayesian trajectory fitting. First we demonstrate trajectory analysis in the
case of continuous target variables. In this case, the algorithm assumes that
the residuals around each trajectory are normally distributed.

### *bayes_traj_main*
The fitting routine is invoked with the *bayes_traj_main* utility. Let's run it
with the -h flag to see what inputs are required:

```python
> bayes_traj_main -h
```

    usage: bayes_traj_main [-h] --in_csv <string> --targets <string>
                           [--groupby <string>] [--out_csv <string>] --prior
                           <string> [--prec_prior_weight <float>]
                           [--alpha <class 'float'>] [--out_model <string>]
                           [--iters <int>] [--repeats <int>] [-k <int>]
                           [--prob_thresh <float>] [--num_init_trajs <int>]
                           [--verbose] [--probs_weight <float>] [--weights_only]
                           [--use_pyro]
    
    Runs Bayesian trajectory analysis on the specified data file with the
    specified predictors and target variables
    
    options:
      -h, --help            show this help message and exit
      --in_csv <string>     Input csv file containing data on which to run
                            Bayesian trajectory analysis
      --targets <string>    Comma-separated list of target names. Must appear as
                            column names of the input data file.
      --groupby <string>    Column name in input data file indicating those data
                            instances that must be in the same trajectory. This is
                            typically a subject identifier (e.g. in the case of a
                            longitudinal data set).
      --out_csv <string>    If specified, an output csv file will be generated
                            that contains the contents of the input csv file, but
                            with additional columns indicating trajectory
                            assignment information for each data instance. There
                            will be a column called traj with an integer value
                            indicating the most probable trajectory assignment.
                            There will also be columns prefixed with traj_ and
                            then a trajectory-identifying integer. The values of
                            these columns indicate the probability that the data
                            instance belongs to each of the corresponding
                            trajectories.
      --prior <string>      Input pickle file containing prior settings
      --prec_prior_weight <float>
                            Positive, floating point value indicating how much
                            weight to put on the prior over the residual
                            precisions. Values greater than 1 give more weight to
                            the prior. Values less than one give less weight to
                            the prior.
      --alpha <class 'float'>
                            If specified, over-rides the value in the prior file
      --out_model <string>  Pickle file name. If specified, the model object will
                            be written to this file.
      --iters <int>         Number of inference iterations
      --repeats <int>       Number of repeats to attempt. If a value greater than
                            1 is specified, the WAIC2 fit criterion will be
                            computed at the end of each repeat. If, for a given
                            repeat, the WAIC2 score is lower than the lowest score
                            seen at that point, the model will be saved to file.
      -k <int>              Number of columns in the truncated assignment matrix
      --prob_thresh <float>
                            If during data fitting the probability of a data
                            instance belonging to a given trajectory drops below
                            this threshold, then the probabality of that data
                            instance belonging to the trajectory will be set to 0
      --num_init_trajs <int>
                            If specified, the initialization procedure will
                            attempt to ensure that the number of initial
                            trajectories in the fitting routine equals the
                            specified number.
      --verbose             Display per-trajectory counts during optimization
      --probs_weight <float>
                            Value between 0 and 1 that controls how much weight to
                            assign to traj_probs, the marginal probability of
                            observing each trajectory. This value is only
                            meaningful if traj_probs has been set in the input
                            prior file. Otherwise, it has no effect. Higher values
                            place more weight on the model-derived probabilities
                            and reflect a stronger belief in those assignment
                            probabilities.
      --weights_only        Setting this flag will force the fitting routine to
                            only optimize the trajectory weights. The assumption
                            is that the specified prior file contains previously
                            modeled trajectory information, and that those
                            trajectories should be used for the current fit. This
                            option can be useful if a model learned from one
                            cohort is applied to another cohort, where it is
                            possible that the relative proportions of different
                            trajectory subgroups differs. By using this flag, the
                            proportions of previously determined trajectory
                            subgroups will be determined for the current data set.
      --use_pyro            Use Pyro for inference


We will run the fitting routine by specifying our data set, the prior we
generated, and the targets and predictors we are interested in analyzing. Also,
it is important to use the *groupby* flag to indicate the column name in your
data set that contains the subject identifier information.

```python
> bayes_traj_main --in_csv bayes_traj_tutorial_std-0.05_visits-3.csv --prior bayes_traj_tutorial_std-0.05_visits-3_prior_v3.p  --targets y1 --groupby id --iters 150 --verbose --out_model bayes_traj_tutorial_std-0.05_visits-3_model_v1.p  
```

    Reading prior...
    Reading data...
    Fitting...
    Initializing parameters...
    iter 1, [4164.1  135.6 4398.   294.8    3.8    2.9    0.7    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 2, [   0.   284.5 1870.1 3122.6 3722.4    0.4    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 3, [   0.  1072.1 1909.2 3593.4 2418.7    6.6    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 4, [   0.  1294.6 2305.4 1982.3 1800.  1617.7    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 5, [   0.  1411.9 2188.1 1800.  1800.  1800.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 6, [   0.  1445.1 2154.9 1800.  1800.  1800.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 7, [   0.  1454.6 2145.4 1800.  1800.  1800.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 8, [   0.  1459.4 2140.6 1800.  1800.  1800.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 9, [   0.  1463.6 2136.4 1800.  1800.  1800.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 10, [   0. 1468. 2132. 1800. 1800. 1800.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.]
    iter 11, [   0. 1473. 2127. 1800. 1800. 1800.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.]
    iter 12, [   0.  1478.7 2121.3 1800.  1800.  1800.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 13, [   0. 1485. 2115. 1800. 1800. 1800.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.]
    iter 14, [   0.  1491.8 2108.2 1800.  1800.  1800.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 15, [   0.  1498.9 2101.1 1800.  1800.  1800.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 16, [   0.  1505.9 2094.1 1800.  1800.  1800.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 17, [   0.  1512.8 2087.2 1800.  1800.  1800.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 18, [   0.  1519.6 2080.4 1800.  1800.  1800.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 19, [   0.  1526.5 2073.5 1800.  1800.  1800.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 20, [   0.  1533.5 2066.5 1800.  1800.  1800.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 21, [   0.  1540.6 2059.4 1800.  1800.  1800.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 22, [   0.  1547.8 2052.2 1800.  1800.  1800.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 23, [   0.  1554.8 2045.2 1800.  1800.  1800.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 24, [   0.  1561.3 2038.7 1800.  1800.  1800.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 25, [   0.  1567.3 2032.7 1800.  1800.  1800.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 26, [   0.  1572.8 2027.2 1800.  1800.  1800.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 27, [   0.  1577.9 2022.1 1800.  1800.  1800.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 28, [   0.  1582.7 2017.3 1800.  1800.  1800.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 29, [   0.  1587.4 2012.6 1800.  1800.  1800.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 30, [   0.  1592.3 2007.7 1800.  1800.  1800.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 31, [   0.  1597.2 2002.8 1800.  1800.  1800.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 32, [   0.  1602.3 1997.7 1800.  1800.  1800.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 33, [   0.  1607.6 1992.4 1800.  1800.  1800.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 34, [   0.  1613.1 1986.9 1800.  1800.  1800.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 35, [   0.  1618.8 1981.2 1800.  1800.  1800.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 36, [   0.  1624.6 1975.4 1800.  1800.  1800.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 37, [   0.  1630.5 1969.5 1800.  1800.  1800.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 38, [   0.  1636.4 1963.6 1800.  1800.  1800.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 39, [   0.  1642.2 1957.8 1800.  1800.  1800.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 40, [   0.  1647.7 1952.3 1800.  1800.  1800.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 41, [   0.  1652.9 1947.1 1800.  1800.  1800.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 42, [   0.  1657.8 1942.2 1800.  1800.  1800.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 43, [   0.  1662.5 1937.5 1800.  1800.  1800.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 44, [   0.  1666.8 1933.2 1800.  1800.  1800.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 45, [   0. 1671. 1929. 1800. 1800. 1800.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.]
    iter 46, [   0. 1675. 1925. 1800. 1800. 1800.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.]
    iter 47, [   0.  1678.8 1921.2 1800.  1800.  1800.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 48, [   0.  1682.6 1917.4 1800.  1800.  1800.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 49, [   0.  1686.2 1913.8 1800.  1800.  1800.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 50, [   0.  1689.8 1910.2 1800.  1800.  1800.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 51, [   0.  1693.4 1906.6 1800.  1800.  1800.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 52, [   0. 1697. 1903. 1800. 1800. 1800.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.]
    iter 53, [   0.  1700.5 1899.5 1800.  1800.  1800.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 54, [   0.  1704.1 1895.9 1800.  1800.  1800.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 55, [   0.  1707.6 1892.4 1800.  1800.  1800.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 56, [   0.  1711.2 1888.8 1800.  1800.  1800.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 57, [   0.  1714.9 1885.1 1800.  1800.  1800.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 58, [   0.  1718.5 1881.5 1800.  1800.  1800.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 59, [   0.  1722.1 1877.9 1800.  1800.  1800.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 60, [   0.  1725.8 1874.2 1800.  1800.  1800.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 61, [   0.  1729.4 1870.6 1800.  1800.  1800.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 62, [   0. 1733. 1867. 1800. 1800. 1800.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.]
    iter 63, [   0.  1736.5 1863.5 1800.  1800.  1800.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 64, [   0.  1739.9 1860.1 1800.  1800.  1800.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 65, [   0.  1743.2 1856.8 1800.  1800.  1800.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 66, [   0.  1746.4 1853.6 1800.  1800.  1800.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 67, [   0.  1749.4 1850.6 1800.  1800.  1800.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 68, [   0.  1752.2 1847.8 1800.  1800.  1800.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 69, [   0.  1754.9 1845.1 1800.  1800.  1800.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 70, [   0.  1757.4 1842.6 1800.  1800.  1800.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 71, [   0.  1759.8 1840.2 1800.  1800.  1800.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 72, [   0. 1762. 1838. 1800. 1800. 1800.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.]
    iter 73, [   0.  1764.1 1835.9 1800.  1800.  1800.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 74, [   0.  1766.1 1833.9 1800.  1800.  1800.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 75, [   0.  1767.9 1832.1 1800.  1800.  1800.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 76, [   0.  1769.7 1830.3 1800.  1800.  1800.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 77, [   0.  1771.3 1828.7 1800.  1800.  1800.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 78, [   0.  1772.9 1827.1 1800.  1800.  1800.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 79, [   0.  1774.3 1825.7 1800.  1800.  1800.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 80, [   0.  1775.7 1824.3 1800.  1800.  1800.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 81, [   0.  1777.1 1822.9 1800.  1800.  1800.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 82, [   0.  1778.3 1821.7 1800.  1800.  1800.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 83, [   0.  1779.5 1820.5 1800.  1800.  1800.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 84, [   0.  1780.7 1819.3 1800.  1800.  1800.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 85, [   0.  1781.8 1818.2 1800.  1800.  1800.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 86, [   0.  1782.8 1817.2 1800.  1800.  1800.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 87, [   0.  1783.8 1816.2 1800.  1800.  1800.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 88, [   0.  1784.8 1815.2 1800.  1800.  1800.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 89, [   0.  1785.7 1814.3 1800.  1800.  1800.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 90, [   0.  1786.6 1813.4 1800.  1800.  1800.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 91, [   0.  1787.4 1812.6 1800.  1800.  1800.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 92, [   0.  1788.2 1811.8 1800.  1800.  1800.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 93, [   0. 1789. 1811. 1800. 1800. 1800.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.]
    iter 94, [   0.  1789.7 1810.3 1800.  1800.  1800.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 95, [   0.  1790.4 1809.6 1800.  1800.  1800.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 96, [   0.  1791.1 1808.9 1800.  1800.  1800.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 97, [   0.  1791.8 1808.2 1800.  1800.  1800.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 98, [   0.  1792.4 1807.6 1800.  1800.  1800.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 99, [   0. 1793. 1807. 1800. 1800. 1800.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.]
    iter 100, [   0.  1793.6 1806.4 1800.  1800.  1800.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 101, [   0.  1794.1 1805.9 1800.  1800.  1800.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 102, [   0.  1794.7 1805.3 1800.  1800.  1800.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 103, [   0.  1795.2 1804.8 1800.  1800.  1800.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 104, [   0.  1795.7 1804.3 1800.  1800.  1800.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 105, [   0.  1796.2 1803.8 1800.  1800.  1800.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 106, [   0.  1796.6 1803.4 1800.  1800.  1800.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 107, [   0.  1797.1 1802.9 1800.  1800.  1800.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 108, [   0.  1797.5 1802.5 1800.  1800.  1800.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 109, [   0.  1797.9 1802.1 1800.  1800.  1800.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 110, [   0.  1798.3 1801.7 1800.  1800.  1800.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 111, [   0.  1798.6 1801.4 1800.  1800.  1800.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 112, [   0. 1799. 1801. 1800. 1800. 1800.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.]
    iter 113, [   0.  1799.3 1800.7 1800.  1800.  1800.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 114, [   0.  1799.7 1800.3 1800.  1800.  1800.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 115, [   0. 1800. 1800. 1800. 1800. 1800.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.]
    iter 116, [   0.  1800.3 1799.7 1800.  1800.  1800.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 117, [   0.  1800.6 1799.4 1800.  1800.  1800.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 118, [   0.  1800.9 1799.1 1800.  1800.  1800.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 119, [   0.  1801.1 1798.9 1800.  1800.  1800.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 120, [   0.  1801.4 1798.6 1800.  1800.  1800.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 121, [   0.  1801.6 1798.4 1800.  1800.  1800.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 122, [   0.  1801.9 1798.1 1800.  1800.  1800.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 123, [   0.  1802.1 1797.9 1800.  1800.  1800.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 124, [   0.  1802.3 1797.7 1800.  1800.  1800.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 125, [   0.  1802.5 1797.5 1800.  1800.  1800.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 126, [   0.  1802.7 1797.3 1800.  1800.  1800.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 127, [   0.  1802.9 1797.1 1800.  1800.  1800.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 128, [   0.  1803.1 1796.9 1800.  1800.  1800.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 129, [   0.  1803.3 1796.7 1800.  1800.  1800.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 130, [   0.  1803.5 1796.5 1800.  1800.  1800.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 131, [   0.  1803.7 1796.3 1800.  1800.  1800.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 132, [   0.  1803.8 1796.2 1800.  1800.  1800.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 133, [   0. 1804. 1796. 1800. 1800. 1800.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.]
    iter 134, [   0.  1804.1 1795.9 1800.  1800.  1800.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 135, [   0.  1804.3 1795.7 1800.  1800.  1800.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 136, [   0.  1804.4 1795.6 1800.  1800.  1800.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 137, [   0.  1804.5 1795.5 1800.  1800.  1800.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 138, [   0.  1804.7 1795.3 1800.  1800.  1800.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 139, [   0.  1804.8 1795.2 1800.  1800.  1800.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 140, [   0.  1804.9 1795.1 1800.  1800.  1800.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 141, [   0. 1805. 1795. 1800. 1800. 1800.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.]
    iter 142, [   0.  1805.1 1794.9 1800.  1800.  1800.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 143, [   0.  1805.2 1794.8 1800.  1800.  1800.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 144, [   0.  1805.3 1794.7 1800.  1800.  1800.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 145, [   0.  1805.4 1794.6 1800.  1800.  1800.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 146, [   0.  1805.5 1794.5 1800.  1800.  1800.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 147, [   0.  1805.6 1794.4 1800.  1800.  1800.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 148, [   0.  1805.7 1794.3 1800.  1800.  1800.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 149, [   0.  1805.8 1794.2 1800.  1800.  1800.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 150, [   0.  1805.9 1794.1 1800.  1800.  1800.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    Saving model...
    Saving model provenance info...
    DONE.


### *viz_model_trajs*
Now that we have saved the model to file, we can use another utility,
viz_model_trajs, to visually inspect the results. First run with the -h
flag to see the inputs:

```python
> viz_model_trajs -h
```

    usage: viz_model_trajs [-h] --model MODEL --y_axis Y_AXIS [--y_label Y_LABEL]
                           --x_axis X_AXIS [--x_label X_LABEL] [--trajs TRAJS]
                           [--min_traj_prob MIN_TRAJ_PROB]
                           [--max_traj_prob MAX_TRAJ_PROB] [--fig_file FIG_FILE]
                           [--traj_map TRAJ_MAP] [--xlim XLIM] [--ylim YLIM]
                           [--hs] [--htd] [--traj_markers TRAJ_MARKERS]
                           [--traj_colors TRAJ_COLORS] [--fill_alpha FILL_ALPHA]
    
    options:
      -h, --help            show this help message and exit
      --model MODEL         Model containing trajectories to visualize
      --y_axis Y_AXIS       Name of the target variable that will be plotted on
                            the y-axis
      --y_label Y_LABEL     Label to display on y-axis. If none given, the
                            variable name specified with the y_axis flag will be
                            used.
      --x_axis X_AXIS       Name of the predictor variable that will be plotted on
                            the x-axis
      --x_label X_LABEL     Label to display on x-axis. If none given, the
                            variable name specified with the x_axis flag will be
                            used.
      --trajs TRAJS         Comma-separated list of trajectories to plot. If none
                            specified, all trajectories will be plotted.
      --min_traj_prob MIN_TRAJ_PROB
                            The probability of a given trajectory must be at least
                            this value in order to be rendered. Value should be
                            between 0 and 1 inclusive.
      --max_traj_prob MAX_TRAJ_PROB
                            The probability of a given trajectory can not be
                            larger than this value in order to be rendered. Value
                            should be between 0 and 1 inclusive.
      --fig_file FIG_FILE   If specified, will save the figure to file.
      --traj_map TRAJ_MAP   The default trajectory numbering scheme is somewhat
                            arbitrary. Use this flag to provide a mapping between
                            the defualt trajectory numbers and a desired numbering
                            scheme. Provide as a comma-separated list of
                            hyphenated mappings. E.g.: 3-1,18-2,7-3 would indicate
                            a mapping from 3 to 1, from 18 to 2, and from 7 to 3.
                            Only the default trajectories in the mapping will be
                            plotted. If this flag is specified, it will override
                            --trajs
      --xlim XLIM           Comma-separated tuple to set the limits of display for
                            the x-axis
      --ylim YLIM           Comma-separated tuple to set the limits of display for
                            the y-axis
      --hs                  This flag will hide the data scatter plot
      --htd                 This flag will hide trajectory legend details (can
                            reduce clutter)
      --traj_markers TRAJ_MARKERS
                            Comma-separated list of markers to use for each
                            trajectory. The number of markers should match the
                            number of trajectories to renders. See matplotlib
                            documentation for marker options
      --traj_colors TRAJ_COLORS
                            Comma-separated list of colors to use for each
                            trajectory. The number of colors should match the
                            number of trajectories to renders. See matplotlib
                            documentation for color options
      --fill_alpha FILL_ALPHA
                            Value between 0 and 1 that controls opacity of each
                            trajectorys fill region (which indicates +\- 2
                            residual standard deviations about the mean)


```python
> viz_model_trajs --model bayes_traj_tutorial_std-0.05_visits-3_model_v1.p --x_axis age --y_axis y1 
```

![png](bayes_traj_tutorial_files/bayes_traj_tutorial_30_2.png)
    
Shown are the average trends (solid lines) identified by the algorithm together
with shaded regions indicating the estimated residual precision values. Data points are
color-coded based on which trajectory groub they most probably belong to.

# Trajectory Modeling with Noisy, Continuous Data
Now that we have generated a trajectory model that we are happy with, we may
wish to use this model to inform trajectory analysis in another data set. In this
section we will analyze a much noisier data set than above. We will begin by
plotting this data set.
    
![png](bayes_traj_tutorial_files/bayes_traj_tutorial_33_0.png)
    


This is also a synthetically generated data set, and -- except for the noise
level -- has the same characteristics as the data set we worked with above
(e.g. five trajectories, three "visits" per "individual" and so on). As before
we start with a prior.

### *generate_prior*: new data
There is no reason why we couldn't use our previously generated prior
straight-away to perform trajectory analysis in this data set. However, the
data fitting we performed above has presumably refined our knowledge about the
trajectories we are likely to encounter in this new data set. As such, we can
provide the previously fit model as an input to *generate_prior* so that it can
inform the generation of a new prior for our current data set (effectively, we
are using the *posterior* of our previously fit model to inform the *prior*
for our current data).

Let's see how this works:

```python
> generate_prior --preds intercept,age --targets y1 --out_file bayes_traj_tutorial_std-0.5_visits-3_prior_v1.p --model bayes_traj_tutorial_std-0.05_visits-3_model_v1.p 
```

    Reading model...
    ---------- Prior Info ----------
    alpha: 5.92e-01
     
    y1 residual (precision mean, precision variance):             (4.00e+02, 7.29e-13)
    y1 intercept (mean, std): (4.80e+00, 4.76e-01)
    y1 age (mean, std): (-2.30e-02, 4.77e-03)


As before, let's visualize some random draws from this prior to see how the
look with respect to our new data:

```python
> viz_data_prior_draws --data_file bayes_traj_tutorial_std-0.5_visits-3.csv --prior bayes_traj_tutorial_std-0.5_visits-3_prior_v1.p --num_draws 20 --x_axis age --y_axis y1
```
    
![png](bayes_traj_tutorial_files/bayes_traj_tutorial_38_2.png)
    

There are a few points to note:

* We did not explicitly define settings for the intercept or residual priors.
These were gleamed from the input model.
* The spread in intercept and slope appear reasonable.
* The residual precision (1/variance) is apparently too high for this data.

Given that the characteristics of our new data may be significantly different
than the characteristics of the data on which the previous model was fit, we
always have the option to specifically indicate what the varios prior settings
should be. For example, here we have good reason to believe that the residual
precision should be lower. As such, we can re-generate the prior and
specifically set the residual precsion:

```python
> generate_prior --preds intercept,age --targets y1 --out_file bayes_traj_tutorial_std-0.5_visits-3_prior_v2.p --model bayes_traj_tutorial_std-0.05_visits-3_model_v1.p --groupby id --tar_resid y1,4,0.01 
```

    Reading model...
    ---------- Prior Info ----------
    alpha: 5.92e-01
     
    y1 residual (precision mean, precision variance):             (4.00e+00, 1.00e-02)
    y1 intercept (mean, std): (4.80e+00, 4.76e-01)
    y1 age (mean, std): (-2.30e-02, 4.77e-03)

Again, visualize random draws from this prior to evaluate these settings:

```python
> viz_data_prior_draws --data_file bayes_traj_tutorial_std-0.5_visits-3.csv --prior bayes_traj_tutorial_std-0.5_visits-3_prior_v2.p --num_draws 2 --x_axis age --y_axis y1
```

![png](bayes_traj_tutorial_files/bayes_traj_tutorial_42_2.png)
    
Looks reasonable. Now let's see how well the trajectory fitting routine works
on this data set with this prior...

### *bayes_traj_main*: new data
Bayesian trajectory fitting proceeds exactly as before with the exception
that we now set the --probs_weight flag. Since the prior was generated from a
previously fit model, the prior file contains the trajectory shapes as well as
the trajectory proportions from that data fit. When we use this prior on a new
data set, we can tell bayes_traj_main how much weight to give to those
previously determined trajectories on initialization; the fitting routine will
then refine trajectory shapes and proportions using the data provided.

```python
> bayes_traj_main --in_csv bayes_traj_tutorial_std-0.5_visits-3.csv --prior bayes_traj_tutorial_std-0.5_visits-3_prior_v2.p  --targets y1 --groupby id --iters 150 --verbose --out_model bayes_traj_tutorial_std-0.5_visits-3_model_v1.p --probs_weight 1
```

    Reading prior...
    Using K=30 (from prior)
    Reading data...
    Fitting...
    Initializing parameters...
    iter 1, [   0.  1763.9 1859.7 1764.7 1858.2 1753.5    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 2, [   0.  1674.4 1997.4 1689.5 1984.7 1654.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 3, [   0.  1519.9 2203.1 1633.2 2165.7 1478.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 4, [   0.  1250.8 2450.4 1741.6 2388.2 1168.9    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 5, [   0.  1015.6 2592.6 1876.7 2514.2 1000.9    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 6, [   0.   938.1 2614.3 1902.8 2474.2 1070.7    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 7, [   0.   970.9 2596.4 1894.2 2377.1 1161.4    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 8, [   0.   983.4 2401.7 1894.3 2575.5 1145.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 9, [   0.   997.1 2187.  1885.  2895.9 1035.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 10, [   0.  1030.9 2046.2 1864.  3104.7  954.2    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 11, [   0.  1069.4 1966.4 1843.5 3214.2  906.4    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 12, [   0.  1104.3 1923.6 1828.1 3268.5  875.5    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 13, [   0.  1134.7 1901.1 1817.6 3293.8  852.7    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 14, [   0.  1161.7 1889.3 1810.7 3304.3  834.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 15, [   0.  1186.3 1882.8 1806.2 3307.3  817.4    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 16, [   0.  1209.6 1879.  1803.2 3306.4  801.9    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 17, [   0.  1231.9 1876.5 1801.2 3303.3  787.1    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 18, [   0.  1253.8 1874.7 1799.8 3299.2  772.6    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 19, [   0.  1275.4 1873.1 1798.9 3294.2  758.4    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 20, [   0.  1296.8 1871.8 1798.2 3288.7  744.5    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 21, [   0.  1318.2 1870.5 1797.9 3282.7  730.7    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 22, [   0.  1339.6 1869.2 1797.6 3276.3  717.2    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 23, [   0.  1361.  1868.  1797.6 3269.5  704.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 24, [   0.  1382.3 1866.7 1797.6 3262.2  691.2    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 25, [   0.  1403.6 1865.5 1797.8 3254.3  678.8    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 26, [   0.  1424.9 1864.2 1798.1 3245.8  667.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 27, [   0.  1446.1 1862.9 1798.5 3236.7  655.8    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 28, [   0.  1467.1 1861.6 1799.  3226.8  645.4    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 29, [   0.  1488.  1860.3 1799.6 3216.2  635.9    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 30, [   0.  1508.7 1859.  1800.3 3204.7  627.4    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 31, [   0.  1529.  1857.6 1801.1 3192.2  620.1    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 32, [   0.  1548.9 1856.2 1801.9 3178.7  614.2    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 33, [   0.  1568.3 1854.8 1802.9 3164.1  609.7    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 34, [   0.  1587.1 1853.4 1804.  3148.5  607.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 35, [   0.  1605.2 1851.9 1805.2 3131.7  606.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 36, [   0.  1622.4 1850.5 1806.4 3113.7  607.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 37, [   0.  1638.8 1849.  1807.7 3094.6  610.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 38, [   0.  1654.1 1847.5 1809.  3074.4  615.1    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 39, [   0.  1668.3 1846.  1810.3 3053.3  622.2    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 40, [   0.  1681.3 1844.5 1811.6 3031.2  631.4    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 41, [   0.  1693.3 1843.  1812.9 3008.3  642.5    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 42, [   0.  1704.1 1841.5 1814.2 2984.7  655.5    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 43, [   0.  1713.8 1840.1 1815.4 2960.5  670.2    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 44, [   0.  1722.4 1838.7 1816.5 2935.9  686.5    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 45, [   0.  1730.1 1837.3 1817.6 2911.   704.1    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 46, [   0.  1736.8 1835.9 1818.6 2885.9  722.8    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 47, [   0.  1742.8 1834.6 1819.4 2860.7  742.6    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 48, [   0.  1748.  1833.3 1820.2 2835.5  763.1    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 49, [   0.  1752.5 1832.  1820.9 2810.4  784.2    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 50, [   0.  1756.5 1830.7 1821.5 2785.4  805.8    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 51, [   0.  1760.  1829.5 1822.1 2760.8  827.7    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 52, [   0.  1763.  1828.3 1822.5 2736.4  849.7    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 53, [   0.  1765.7 1827.1 1822.9 2712.4  871.8    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 54, [   0.  1768.1 1825.9 1823.3 2688.8  893.9    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 55, [   0.  1770.2 1824.8 1823.6 2665.6  915.8    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 56, [   0.  1772.1 1823.7 1823.8 2642.9  937.5    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 57, [   0.  1773.8 1822.6 1824.  2620.7  958.9    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 58, [   0.  1775.3 1821.6 1824.2 2598.9  980.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 59, [   0.  1776.6 1820.5 1824.4 2577.8 1000.7    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 60, [   0.  1777.9 1819.5 1824.5 2557.1 1021.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 61, [   0.  1779.  1818.5 1824.6 2537.  1040.8    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 62, [   0.  1780.  1817.6 1824.7 2517.5 1060.2    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 63, [   0.  1781.  1816.6 1824.8 2498.4 1079.1    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 64, [   0.  1781.8 1815.7 1824.9 2480.  1097.6    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 65, [   0.  1782.7 1814.8 1825.  2462.  1115.5    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 66, [   0.  1783.4 1814.  1825.  2444.6 1132.9    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 67, [   0.  1784.2 1813.1 1825.1 2427.7 1149.9    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 68, [   0.  1784.9 1812.3 1825.1 2411.4 1166.3    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 69, [   0.  1785.5 1811.5 1825.2 2395.5 1182.3    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 70, [   0.  1786.1 1810.8 1825.2 2380.1 1197.8    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 71, [   0.  1786.7 1810.1 1825.3 2365.2 1212.8    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 72, [   0.  1787.3 1809.3 1825.3 2350.8 1227.3    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 73, [   0.  1787.9 1808.7 1825.3 2336.8 1241.4    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 74, [   0.  1788.4 1808.  1825.4 2323.3 1255.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 75, [   0.  1788.9 1807.3 1825.4 2310.1 1268.2    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 76, [   0.  1789.4 1806.7 1825.4 2297.4 1281.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 77, [   0.  1789.8 1806.1 1825.5 2285.1 1293.4    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 78, [   0.  1790.3 1805.5 1825.5 2273.2 1305.5    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 79, [   0.  1790.7 1805.  1825.5 2261.7 1317.1    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 80, [   0.  1791.1 1804.4 1825.5 2250.5 1328.4    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 81, [   0.  1791.5 1803.9 1825.6 2239.7 1339.3    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 82, [   0.  1791.9 1803.4 1825.6 2229.2 1349.9    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 83, [   0.  1792.3 1802.9 1825.6 2219.1 1360.1    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 84, [   0.  1792.6 1802.4 1825.7 2209.3 1370.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 85, [   0.  1793.  1802.  1825.7 2199.8 1379.6    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 86, [   0.  1793.3 1801.5 1825.7 2190.6 1388.9    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 87, [   0.  1793.7 1801.1 1825.7 2181.6 1397.9    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 88, [   0.  1794.  1800.7 1825.8 2173.  1406.6    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 89, [   0.  1794.3 1800.3 1825.8 2164.6 1415.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 90, [   0.  1794.6 1799.9 1825.8 2156.5 1423.2    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 91, [   0.  1794.9 1799.5 1825.8 2148.7 1431.2    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 92, [   0.  1795.1 1799.1 1825.8 2141.  1438.8    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 93, [   0.  1795.4 1798.8 1825.9 2133.7 1446.3    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 94, [   0.  1795.7 1798.4 1825.9 2126.5 1453.5    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 95, [   0.  1795.9 1798.1 1825.9 2119.6 1460.5    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 96, [   0.  1796.1 1797.8 1825.9 2112.8 1467.3    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 97, [   0.  1796.4 1797.5 1825.9 2106.3 1473.9    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 98, [   0.  1796.6 1797.2 1826.  2100.  1480.3    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 99, [   0.  1796.8 1796.9 1826.  2093.8 1486.5    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 100, [   0.  1797.  1796.6 1826.  2087.8 1492.6    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 101, [   0.  1797.2 1796.3 1826.  2082.  1498.4    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 102, [   0.  1797.4 1796.1 1826.  2076.4 1504.1    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 103, [   0.  1797.6 1795.8 1826.1 2071.  1509.6    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 104, [   0.  1797.8 1795.6 1826.1 2065.6 1515.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 105, [   0.  1797.9 1795.3 1826.1 2060.5 1520.2    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 106, [   0.  1798.1 1795.1 1826.1 2055.5 1525.2    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 107, [   0.  1798.3 1794.8 1826.1 2050.6 1530.2    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 108, [   0.  1798.4 1794.6 1826.1 2045.9 1534.9    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 109, [   0.  1798.6 1794.4 1826.1 2041.3 1539.6    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 110, [   0.  1798.7 1794.2 1826.2 2036.8 1544.1    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 111, [   0.  1798.8 1794.  1826.2 2032.5 1548.5    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 112, [   0.  1799.  1793.8 1826.2 2028.2 1552.8    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 113, [   0.  1799.1 1793.6 1826.2 2024.1 1557.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 114, [   0.  1799.2 1793.4 1826.2 2020.1 1561.1    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 115, [   0.  1799.4 1793.2 1826.2 2016.2 1565.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 116, [   0.  1799.5 1793.1 1826.2 2012.4 1568.9    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 117, [   0.  1799.6 1792.9 1826.2 2008.7 1572.6    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 118, [   0.  1799.7 1792.7 1826.3 2005.  1576.3    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 119, [   0.  1799.8 1792.6 1826.3 2001.5 1579.8    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 120, [   0.  1799.9 1792.4 1826.3 1998.1 1583.3    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 121, [   0.  1800.  1792.3 1826.3 1994.8 1586.7    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 122, [   0.  1800.1 1792.1 1826.3 1991.5 1590.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 123, [   0.  1800.2 1792.  1826.3 1988.3 1593.2    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 124, [   0.  1800.3 1791.8 1826.3 1985.2 1596.4    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 125, [   0.  1800.4 1791.7 1826.3 1982.2 1599.4    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 126, [   0.  1800.5 1791.5 1826.3 1979.2 1602.4    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 127, [   0.  1800.5 1791.4 1826.3 1976.3 1605.4    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 128, [   0.  1800.6 1791.3 1826.4 1973.5 1608.2    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 129, [   0.  1800.7 1791.2 1826.4 1970.8 1611.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 130, [   0.  1800.8 1791.  1826.4 1968.1 1613.7    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 131, [   0.  1800.8 1790.9 1826.4 1965.5 1616.4    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 132, [   0.  1800.9 1790.8 1826.4 1962.9 1619.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 133, [   0.  1801.  1790.7 1826.4 1960.4 1621.5    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 134, [   0.  1801.  1790.6 1826.4 1958.  1624.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 135, [   0.  1801.1 1790.5 1826.4 1955.6 1626.4    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 136, [   0.  1801.2 1790.4 1826.4 1953.2 1628.8    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 137, [   0.  1801.2 1790.3 1826.4 1951.  1631.1    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 138, [   0.  1801.3 1790.2 1826.4 1948.7 1633.4    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 139, [   0.  1801.3 1790.1 1826.4 1946.5 1635.6    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 140, [   0.  1801.4 1790.  1826.4 1944.4 1637.8    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 141, [   0.  1801.4 1789.9 1826.4 1942.3 1639.9    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 142, [   0.  1801.5 1789.8 1826.4 1940.3 1642.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 143, [   0.  1801.5 1789.7 1826.4 1938.2 1644.1    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 144, [   0.  1801.6 1789.6 1826.4 1936.3 1646.1    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 145, [   0.  1801.6 1789.5 1826.5 1934.4 1648.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 146, [   0.  1801.7 1789.5 1826.5 1932.5 1649.9    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 147, [   0.  1801.7 1789.4 1826.5 1930.6 1651.8    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 148, [   0.  1801.8 1789.3 1826.5 1928.8 1653.7    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 149, [   0.  1801.8 1789.2 1826.5 1927.  1655.5    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 150, [   0.  1801.8 1789.1 1826.5 1925.3 1657.2    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    Saving model...
    Saving model provenance info...
    DONE.


As before, let's visualize the trajectories:

```python
> viz_model_trajs --model bayes_traj_tutorial_std-0.5_visits-3_model_v1.p --x_axis age --y_axis y1
```
    
![png](bayes_traj_tutorial_files/bayes_traj_tutorial_47_2.png)
    
This is a challenging data set. Although this data set was created with 5
different trajectories, the noise level is high, making it challenging to
identify them. Here we benefited from a prior informed by a previously fit
model. 

### New data, multiple dimensions
Instead of performing trajectory analysis on a single target variable (e.g. y1),
we can identify trajectory subgroups by considering progression patterns in
multiple dimensions, here y1 *and* y2. 

Let's begin by plotting both y1 and y2 vs age for our new data set:
    
![png](bayes_traj_tutorial_files/bayes_traj_tutorial_50_0.png)
    
As before, we begin by generating a prior.

### *generate_prior*: new data, multiple dimensions
We will use our previously generated model as before. Recall that we
specifically set the prior over the y1 residual precision. Given that y2
visually appears to have similar data characteristics, a reasonable place to
start is to specify the same prior for the y2 residuals.

```python
> generate_prior --in_data bayes_traj_tutorial_std-0.5_visits-3.csv --preds intercept,age --targets y1,y2 --groupby id --out_file bayes_traj_tutorial_std-0.5_visits-3_prior_v3.p --tar_resid y1,4,0.01 --tar_resid y2,4,0.01
```

    Reading data...
    ---------- Prior Info ----------
    alpha: 3.31e-01
     
    y1 residual (precision mean, precision variance):             (4.00e+00, 1.00e-02)
    y1 intercept (mean, std): (4.92e+00, 5.49e-01)
    y1 age (mean, std): (-2.47e-02, 8.20e-03)
     
    y2 residual (precision mean, precision variance):             (4.00e+00, 1.00e-02)
    y2 intercept (mean, std): (4.84e+00, 3.93e-01)
    y2 age (mean, std): (-1.56e-02, 5.76e-03)


Again, visualize random draws from our prior overlayed on our data. Now that we
are considering multiple output dimensions, we should look at draws for both
y1 and y2:

```python
> viz_data_prior_draws --data_file bayes_traj_tutorial_std-0.5_visits-3.csv --prior bayes_traj_tutorial_std-0.5_visits-3_prior_v3.p --num_draws 100 --hide_resid --x_axis age --y_axis y1
```
    
![png](bayes_traj_tutorial_files/bayes_traj_tutorial_54_2.png)
    
```python
> viz_data_prior_draws --data_file bayes_traj_tutorial_std-0.5_visits-3.csv --prior bayes_traj_tutorial_std-0.5_visits-3_prior_v3.p --num_draws 100 --x_axis age --y_axis y2 --hide_resid 
```
    
![png](bayes_traj_tutorial_files/bayes_traj_tutorial_55_2.png)    

### *bayes_traj_main*: new data, multiple dimensions
Trajectory fitting proceeds as before. We specify the newly generated prior and
also indicate that we want trajectories defined with respect to y1 *and* y2. 

```python
> bayes_traj_main --in_csv bayes_traj_tutorial_std-0.5_visits-3.csv --prior bayes_traj_tutorial_std-0.5_visits-3_prior_v3.p  --targets y1,y2 --groupby id --iters 150 --verbose --out_model bayes_traj_tutorial_std-0.5_visits-3_model_v2.p --alpha 2
```

    Reading prior...
    Reading data...
    Fitting...
    Initializing parameters...
    iter 1, [7733.4  815.7  203.9   12.8   75.4   92.9   65.9    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 2, [7653.5  919.6  197.1    8.5   70.    93.9   57.5    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 3, [7458.4 1108.   211.8    5.7   72.7   93.1   50.3    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 4, [7096.5 1423.3  251.6    4.    80.8   96.3   47.5    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 5, [6641.9 1721.3  350.3    2.3   94.2  122.    67.9    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 6, [6019.8 1934.5  599.5    0.1  152.   168.2  125.9    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 7, [4801.8 2087.6  844.3    0.   820.8  231.6  213.9    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 8, [3949.5 1954.4  758.8    0.  1540.7  373.9  422.8    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 9, [3538.5 1946.5  720.1    0.  1679.9  535.3  579.7    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 10, [3389.1 1937.9  669.1    0.  1734.3  646.   623.6    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 11, [3281.5 1921.1  643.4    0.  1757.5  747.8  648.6    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 12, [3141.4 1892.7  637.6    0.  1767.2  875.6  685.5    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 13, [2903.6 1853.2  651.     0.  1770.3 1065.2  756.7    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 14, [2514.6 1821.3  677.6    0.  1770.6 1350.8  865.2    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 15, [2070.5 1810.8  697.2    0.  1771.  1669.4  981.1    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 16, [1849.6 1815.2  695.1    0.  1773.3 1800.1 1066.6    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 17, [1793.1 1819.9  678.3    0.  1776.3 1810.9 1121.5    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 18, [1780.5 1821.   655.9    0.  1778.6 1804.2 1159.7    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 19, [1777.6 1820.6  632.2    0.  1779.9 1800.  1189.8    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 20, [1776.9 1820.   608.6    0.  1780.6 1798.3 1215.6    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 21, [1776.7 1819.6  585.8    0.  1781.  1797.8 1239.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 22, [1776.7 1819.5  563.9    0.  1781.3 1797.9 1260.7    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 23, [1776.7 1819.4  543.     0.  1781.5 1798.3 1281.1    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 24, [1776.7 1819.5  523.1    0.  1781.8 1798.7 1300.3    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 25, [1776.7 1819.5  504.1    0.  1782.  1799.2 1318.4    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 26, [1776.8 1819.5  486.1    0.  1782.3 1799.7 1335.7    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 27, [1776.8 1819.5  468.9    0.  1782.5 1800.3 1352.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 28, [1776.8 1819.4  452.6    0.  1782.8 1800.9 1367.5    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 29, [1776.9 1819.3  437.1    0.  1783.1 1801.5 1382.2    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 30, [1776.9 1819.2  422.3    0.  1783.4 1802.1 1396.1    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 31, [1777.  1819.   408.2    0.  1783.8 1802.7 1409.4    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 32, [1777.  1818.9  394.8    0.  1784.1 1803.3 1422.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 33, [1777.  1818.7  382.     0.  1784.4 1803.8 1434.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 34, [1777.1 1818.6  369.8    0.  1784.7 1804.4 1445.5    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 35, [1777.1 1818.4  358.2    0.  1785.  1805.  1456.4    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 36, [1777.2 1818.2  347.     0.  1785.3 1805.5 1466.8    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 37, [1777.2 1818.1  336.4    0.  1785.6 1806.1 1476.7    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 38, [1777.2 1817.9  326.2    0.  1785.9 1806.6 1486.2    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 39, [1777.3 1817.8  316.4    0.  1786.1 1807.1 1495.3    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 40, [1777.3 1817.6  307.1    0.  1786.4 1807.6 1504.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 41, [1777.4 1817.5  298.1    0.  1786.7 1808.1 1512.3    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 42, [1777.4 1817.3  289.4    0.  1787.  1808.6 1520.3    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 43, [1777.5 1817.2  281.1    0.  1787.3 1809.  1528.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 44, [1777.5 1817.   273.1    0.  1787.5 1809.5 1535.4    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 45, [1777.5 1816.9  265.4    0.  1787.8 1809.9 1542.5    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 46, [1777.6 1816.8  257.9    0.  1788.  1810.4 1549.3    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 47, [1777.6 1816.6  250.7    0.  1788.3 1810.8 1556.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 48, [1777.7 1816.5  243.8    0.  1788.5 1811.2 1562.4    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 49, [1777.7 1816.4  237.     0.  1788.7 1811.6 1568.5    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 50, [1777.7 1816.3  230.5    0.  1789.  1812.  1574.5    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 51, [1777.8 1816.2  224.1    0.  1789.2 1812.4 1580.3    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 52, [1777.8 1816.2  217.9    0.  1789.4 1812.8 1585.9    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 53, [1777.8 1816.1  211.9    0.  1789.6 1813.2 1591.4    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 54, [1777.9 1816.   206.     0.  1789.9 1813.5 1596.7    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 55, [1777.9 1815.9  200.3    0.  1790.1 1813.9 1601.9    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 56, [1777.9 1815.9  194.7    0.  1790.3 1814.3 1606.9    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 57, [1778.  1815.8  189.3    0.  1790.5 1814.6 1611.8    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 58, [1778.  1815.8  183.9    0.  1790.7 1814.9 1616.6    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 59, [1778.  1815.8  178.7    0.  1790.9 1815.3 1621.3    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 60, [1778.1 1815.7  173.6    0.  1791.1 1815.6 1625.9    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 61, [1778.1 1815.7  168.6    0.  1791.3 1815.9 1630.4    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 62, [1778.1 1815.7  163.6    0.  1791.4 1816.2 1634.9    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 63, [1778.1 1815.7  158.8    0.  1791.6 1816.5 1639.2    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 64, [1778.2 1815.7  154.     0.  1791.8 1816.9 1643.5    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 65, [1778.2 1815.7  149.3    0.  1792.  1817.1 1647.7    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 66, [1778.2 1815.7  144.6    0.  1792.2 1817.4 1651.9    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 67, [1778.2 1815.7  140.     0.  1792.3 1817.7 1656.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 68, [1778.3 1815.8  135.4    0.  1792.5 1818.  1660.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 69, [1778.3 1815.8  131.     0.  1792.7 1818.3 1664.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 70, [1778.3 1815.8  126.5    0.  1792.9 1818.5 1668.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 71, [1778.3 1815.9  122.1    0.  1793.  1818.8 1671.9    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 72, [1778.4 1815.9  117.7    0.  1793.2 1819.1 1675.7    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 73, [1778.4 1816.   113.4    0.  1793.3 1819.3 1679.6    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 74, [1778.4 1816.1  109.1    0.  1793.5 1819.6 1683.4    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 75, [1778.4 1816.1  104.8    0.  1793.7 1819.8 1687.1    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 76, [1778.4 1816.2  100.6    0.  1793.8 1820.1 1690.9    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 77, [1778.5 1816.3   96.4    0.  1794.  1820.3 1694.6    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 78, [1778.5 1816.4   92.2    0.  1794.1 1820.5 1698.3    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 79, [1778.5 1816.5   88.     0.  1794.3 1820.8 1702.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 80, [1778.5 1816.6   83.8    0.  1794.4 1821.  1705.6    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 81, [1778.5 1816.7   79.7    0.  1794.6 1821.2 1709.3    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 82, [1778.6 1816.8   75.5    0.  1794.8 1821.4 1712.9    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 83, [1778.6 1816.9   71.4    0.  1794.9 1821.6 1716.6    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 84, [1778.6 1817.    67.2    0.  1795.1 1821.9 1720.2    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 85, [1778.6 1817.2   63.1    0.  1795.2 1822.1 1723.8    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 86, [1778.6 1817.3   58.9    0.  1795.4 1822.3 1727.5    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 87, [1778.7 1817.5   54.8    0.  1795.5 1822.5 1731.1    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 88, [1778.7 1817.6   50.7    0.  1795.7 1822.7 1734.7    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 89, [1778.7 1817.8   46.6    0.  1795.8 1822.9 1738.3    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 90, [1778.7 1818.    42.4    0.  1796.  1823.1 1741.9    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 91, [1778.7 1818.1   38.3    0.  1796.1 1823.3 1745.4    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 92, [1778.7 1818.3   34.2    0.  1796.3 1823.4 1749.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 93, [1778.8 1818.5   30.1    0.  1796.4 1823.6 1752.6    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 94, [1778.8 1818.7   25.9    0.  1796.6 1823.8 1756.2    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 95, [1778.8 1818.9   21.9    0.  1796.7 1824.  1759.7    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 96, [1778.8 1819.    17.8    0.  1796.9 1824.2 1763.3    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 97, [1778.8 1819.2   13.9    0.  1797.  1824.4 1766.8    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 98, [1778.8 1819.3   10.1    0.  1797.1 1824.5 1770.2    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 99, [1778.8 1819.3    6.4    0.  1797.2 1824.7 1773.5    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 100, [1778.9 1819.3    3.     0.  1797.2 1824.8 1776.7    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 101, [1778.9 1819.3    0.2    0.  1797.3 1825.  1779.4    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 102, [1778.9 1818.9    0.     0.  1797.2 1825.1 1779.9    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 103, [1778.9 1818.6    0.     0.  1797.2 1825.3 1780.1    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 104, [1778.9 1818.3    0.     0.  1797.2 1825.4 1780.1    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 105, [1778.9 1818.2    0.     0.  1797.3 1825.5 1780.1    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 106, [1778.9 1818.     0.     0.  1797.3 1825.7 1780.1    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 107, [1778.9 1817.9    0.     0.  1797.3 1825.8 1780.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 108, [1779.  1817.8    0.     0.  1797.3 1826.  1780.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 109, [1779.  1817.6    0.     0.  1797.4 1826.1 1779.9    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 110, [1779.  1817.5    0.     0.  1797.4 1826.2 1779.9    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 111, [1779.  1817.4    0.     0.  1797.4 1826.4 1779.9    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 112, [1779.  1817.3    0.     0.  1797.4 1826.5 1779.8    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 113, [1779.  1817.2    0.     0.  1797.4 1826.6 1779.8    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 114, [1779.  1817.1    0.     0.  1797.5 1826.7 1779.7    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 115, [1779.  1817.     0.     0.  1797.5 1826.8 1779.7    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 116, [1779.  1816.9    0.     0.  1797.5 1826.9 1779.6    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 117, [1779.  1816.8    0.     0.  1797.5 1827.1 1779.6    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 118, [1779.1 1816.7    0.     0.  1797.6 1827.2 1779.5    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 119, [1779.1 1816.6    0.     0.  1797.6 1827.3 1779.5    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 120, [1779.1 1816.5    0.     0.  1797.6 1827.4 1779.5    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 121, [1779.1 1816.4    0.     0.  1797.6 1827.5 1779.4    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 122, [1779.1 1816.3    0.     0.  1797.6 1827.6 1779.4    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 123, [1779.1 1816.2    0.     0.  1797.6 1827.7 1779.3    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 124, [1779.1 1816.2    0.     0.  1797.6 1827.8 1779.3    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 125, [1779.1 1816.1    0.     0.  1797.7 1827.9 1779.3    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 126, [1779.1 1816.     0.     0.  1797.7 1828.  1779.2    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 127, [1779.1 1815.9    0.     0.  1797.7 1828.1 1779.2    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 128, [1779.1 1815.8    0.     0.  1797.7 1828.1 1779.2    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 129, [1779.1 1815.8    0.     0.  1797.7 1828.2 1779.1    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 130, [1779.2 1815.7    0.     0.  1797.7 1828.3 1779.1    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 131, [1779.2 1815.6    0.     0.  1797.7 1828.4 1779.1    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 132, [1779.2 1815.5    0.     0.  1797.8 1828.5 1779.1    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 133, [1779.2 1815.5    0.     0.  1797.8 1828.6 1779.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 134, [1779.2 1815.4    0.     0.  1797.8 1828.6 1779.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 135, [1779.2 1815.3    0.     0.  1797.8 1828.7 1779.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 136, [1779.2 1815.3    0.     0.  1797.8 1828.8 1779.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 137, [1779.2 1815.2    0.     0.  1797.8 1828.9 1778.9    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 138, [1779.2 1815.1    0.     0.  1797.8 1828.9 1778.9    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 139, [1779.2 1815.1    0.     0.  1797.8 1829.  1778.9    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 140, [1779.2 1815.     0.     0.  1797.8 1829.1 1778.9    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 141, [1779.2 1815.     0.     0.  1797.8 1829.1 1778.8    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 142, [1779.2 1814.9    0.     0.  1797.8 1829.2 1778.8    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 143, [1779.2 1814.8    0.     0.  1797.9 1829.3 1778.8    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 144, [1779.2 1814.8    0.     0.  1797.9 1829.3 1778.8    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 145, [1779.3 1814.7    0.     0.  1797.9 1829.4 1778.8    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 146, [1779.3 1814.7    0.     0.  1797.9 1829.5 1778.7    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 147, [1779.3 1814.6    0.     0.  1797.9 1829.5 1778.7    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 148, [1779.3 1814.6    0.     0.  1797.9 1829.6 1778.7    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 149, [1779.3 1814.5    0.     0.  1797.9 1829.6 1778.7    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 150, [1779.3 1814.5    0.     0.  1797.9 1829.7 1778.7    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    Saving model...
    Saving model provenance info...
    DONE.


Plot the trajectories for both y1 and y2 to assess the fit:

```python
> viz_model_trajs --model bayes_traj_tutorial_std-0.5_visits-3_model_v2.p --x_axis age --y_axis y1 
```
    
![png](bayes_traj_tutorial_files/bayes_traj_tutorial_59_2.png)

```python
> viz_model_trajs --model bayes_traj_tutorial_std-0.5_visits-3_model_v2.p --x_axis age --y_axis y2 
```
    
![png](bayes_traj_tutorial_files/bayes_traj_tutorial_60_2.png)    

The addition of y2 improves the ability of the trajectory algorithm to recover
the underlying population structure. 

Note that the fitting routine begins with a *random initialization*. For this
particular data set and prior, multiple invocations of the fitting routine may
result in different trajectory fit results. In practice, it is advised to
generate multiple models for a given prior. Comparing these models both
qualitatively and quantitatively can help identify spurious and stable
trajectory subgroups.

# Model Comparison

At this point, we may be wondering if we can produce a better fit using a
different predictor set. For example, what if we include age^2 as an
additional predictor?

The basic steps should now be familiar. Start by generating a prior.

### *generate_prior*: new data, multiple dimensions, different predictors

```python
> generate_prior --in_data bayes_traj_tutorial_std-0.5_visits-3.csv --preds intercept,age,age^2 --targets y1,y2 --out_file bayes_traj_tutorial_std-0.5_visits-3_prior_v4.p  --tar_resid y1,4,0.01 --tar_resid y2,4,0.01
```

    Reading data...
    ---------- Prior Info ----------
    alpha: 2.91e-01
     
    y1 residual (precision mean, precision variance):             (4.00e+00, 1.00e-02)
    y1 intercept (mean, std): (4.86e+00, 3.21e+00)
    y1 age (mean, std): (-2.30e-02, 9.78e-02)
    y1 age^2 (mean, std): (-1.25e-05, 7.31e-04)
     
    y2 residual (precision mean, precision variance):             (4.00e+00, 1.00e-02)
    y2 intercept (mean, std): (4.91e+00, 2.29e+00)
    y2 age (mean, std): (-1.77e-02, 6.90e-02)
    y2 age^2 (mean, std): (1.59e-05, 5.10e-04)

```python
> viz_data_prior_draws --data_file bayes_traj_tutorial_std-0.5_visits-3.csv --prior bayes_traj_tutorial_std-0.5_visits-3_prior_v4.p --num_draws 10 --x_axis age --y_axis y1 
```
    
![png](bayes_traj_tutorial_files/bayes_traj_tutorial_65_2.png)

```python
> viz_data_prior_draws --data_file bayes_traj_tutorial_std-0.5_visits-3.csv --prior bayes_traj_tutorial_std-0.5_visits-3_prior_v4.p --num_draws 5 --x_axis age --y_axis y2
```
    
![png](bayes_traj_tutorial_files/bayes_traj_tutorial_66_2.png)    

We could further refine it as described above, but in the interest of
demonstrating model comparison, it suffices.

### *bayes_traj_main*: new data, multiple dimensions, different predictors

```python
> bayes_traj_main --in_csv bayes_traj_tutorial_std-0.5_visits-3.csv --prior bayes_traj_tutorial_std-0.5_visits-3_prior_v4.p --targets y1,y2 --groupby id --iters 150 --verbose --out_model bayes_traj_tutorial_std-0.5_visits-3_model_v3.p --alpha .6
```

    Reading prior...
    Reading data...
    Fitting...
    Initializing parameters...
    iter 1, [3986.6 1161.7 2267.7  853.3  730.7    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 2, [3892.1  903.1 2533.9  848.8  822.1    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 3, [3936.8  709.6 2565.1  886.   902.5    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 4, [3859.2  639.2 2536.6 1021.7  943.3    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 5, [3496.2  686.3 2670.9 1200.2  946.5    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 6, [2729.9  747.7 3193.7 1340.7  988.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 7, [2320.9  744.8 3478.9 1475.4  980.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 8, [1993.   863.3 3365.9 1729.5 1048.3    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 9, [1733.   958.2 3246.  1902.7 1160.1    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 10, [1576.3 1004.  3166.7 2010.9 1242.1    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 11, [1504.4 1025.9 3112.7 2071.4 1285.7    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 12, [1483.5 1043.7 3071.  2099.5 1302.3    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 13, [1485.8 1062.1 3033.4 2112.7 1306.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 14, [1498.7 1081.  2995.9 2120.2 1304.3    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 15, [1516.9 1100.2 2956.2 2125.8 1301.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 16, [1537.7 1120.1 2912.2 2131.6 1298.4    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 17, [1559.5 1141.5 2861.8 2138.5 1298.7    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 18, [1581.3 1165.6 2802.1 2147.5 1303.4    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 19, [1602.8 1193.5 2730.2 2159.2 1314.2    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 20, [1623.6 1226.3 2642.8 2175.  1332.4    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 21, [1643.8 1265.8 2535.7 2195.8 1359.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 22, [1663.5 1315.7 2402.3 2222.8 1395.7    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 23, [1683.  1376.  2238.2 2256.4 1446.4    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 24, [1702.6 1435.6 2052.6 2292.8 1516.3    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 25, [1722.6 1468.2 1884.2 2322.3 1602.7    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 26, [1743.  1462.7 1767.6 2336.8 1689.9    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 27, [1762.8 1430.9 1700.9 2339.5 1765.9    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 28, [1780.1 1389.6 1665.3 2338.8 1826.2    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 29, [1794.3 1347.1 1646.6 2340.9 1871.1    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 30, [1805.5 1308.6 1637.  2348.  1900.9    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 31, [1814.2 1277.1 1632.4 2359.1 1917.2    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 32, [1820.7 1253.9 1630.5 2371.8 1923.2    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 33, [1825.1 1238.5 1630.  2384.  1922.4    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 34, [1828.  1229.6 1630.4 2394.  1917.9    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 35, [1829.9 1226.2 1631.4 2400.8 1911.8    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 36, [1831.1 1226.9 1632.7 2404.3 1905.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 37, [1831.9 1230.9 1634.2 2404.8 1898.2    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 38, [1832.6 1237.3 1635.8 2402.8 1891.5    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 39, [1833.1 1245.6 1637.5 2398.7 1885.1    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 40, [1833.5 1255.2 1639.1 2393.2 1879.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 41, [1833.9 1265.7 1640.8 2386.4 1873.2    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 42, [1834.2 1277.  1642.4 2378.7 1867.7    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 43, [1834.5 1288.7 1644.  2370.4 1862.5    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 44, [1834.8 1300.7 1645.5 2361.5 1857.5    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 45, [1835.  1313.  1647.  2352.2 1852.7    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 46, [1835.3 1325.4 1648.5 2342.5 1848.3    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 47, [1835.6 1337.8 1650.  2332.4 1844.2    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 48, [1835.9 1350.1 1651.5 2322.2 1840.4    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 49, [1836.1 1362.4 1652.9 2311.7 1836.8    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 50, [1836.4 1374.6 1654.4 2301.  1833.6    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 51, [1836.7 1386.7 1655.8 2290.1 1830.7    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 52, [1837.  1398.6 1657.3 2279.  1828.1    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 53, [1837.3 1410.4 1658.7 2267.9 1825.7    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 54, [1837.7 1422.  1660.2 2256.5 1823.6    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 55, [1838.  1433.4 1661.7 2245.1 1821.8    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 56, [1838.4 1444.7 1663.2 2233.6 1820.1    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 57, [1838.8 1455.8 1664.8 2222.1 1818.6    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 58, [1839.2 1466.8 1666.4 2210.5 1817.1    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 59, [1839.7 1477.7 1668.  2198.8 1815.8    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 60, [1840.2 1488.5 1669.7 2187.2 1814.5    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 61, [1840.7 1499.2 1671.4 2175.4 1813.2    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 62, [1841.2 1510.  1673.2 2163.6 1812.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 63, [1841.9 1520.8 1675.  2151.7 1810.7    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 64, [1842.5 1531.7 1676.8 2139.6 1809.4    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 65, [1843.2 1542.7 1678.7 2127.3 1808.1    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 66, [1843.9 1553.9 1680.7 2114.7 1806.8    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 67, [1844.7 1565.2 1682.6 2101.8 1805.6    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 68, [1845.6 1576.8 1684.6 2088.3 1804.7    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 69, [1846.5 1588.8 1686.7 2074.1 1803.9    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 70, [1847.4 1601.  1688.8 2059.2 1803.6    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 71, [1848.4 1613.6 1690.9 2043.4 1803.7    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 72, [1849.3 1626.5 1693.1 2026.7 1804.4    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 73, [1850.3 1639.9 1695.2 2008.9 1805.7    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 74, [1851.3 1653.6 1697.4 1990.1 1807.7    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 75, [1852.2 1667.6 1699.6 1970.3 1810.3    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 76, [1853.  1681.8 1701.8 1949.9 1813.5    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 77, [1853.5 1696.2 1704.  1929.1 1817.2    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 78, [1853.8 1710.6 1706.2 1908.  1821.4    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 79, [1853.6 1724.9 1708.3 1887.2 1826.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 80, [1852.9 1738.9 1710.5 1867.2 1830.5    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 81, [1851.4 1752.4 1712.6 1849.  1834.6    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 82, [1849.1 1765.  1714.8 1833.5 1837.6    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 83, [1846.1 1776.3 1717.  1821.3 1839.2    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 84, [1842.6 1786.  1719.3 1812.8 1839.3    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 85, [1838.7 1793.9 1721.7 1807.6 1838.1    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 86, [1834.7 1800.  1724.2 1804.9 1836.2    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 87, [1830.8 1804.3 1726.8 1804.3 1833.9    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 88, [1827.1 1807.1 1729.4 1804.8 1831.6    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 89, [1823.6 1808.6 1732.1 1806.1 1829.6    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 90, [1820.4 1809.  1734.8 1807.7 1828.1    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 91, [1817.5 1808.7 1737.4 1809.4 1826.9    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 92, [1814.9 1807.9 1740.  1811.1 1826.2    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 93, [1812.4 1806.6 1742.5 1812.7 1825.8    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 94, [1810.2 1805.1 1744.9 1814.1 1825.7    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 95, [1808.1 1803.5 1747.1 1815.4 1825.9    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 96, [1806.2 1801.8 1749.3 1816.5 1826.2    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 97, [1804.5 1800.  1751.4 1817.5 1826.6    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 98, [1802.8 1798.3 1753.4 1818.3 1827.2    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 99, [1801.3 1796.7 1755.3 1819.  1827.7    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 100, [1800.  1795.1 1757.1 1819.6 1828.3    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 101, [1798.7 1793.6 1758.7 1820.1 1828.9    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 102, [1797.5 1792.2 1760.3 1820.5 1829.4    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 103, [1796.4 1790.9 1761.9 1820.9 1830.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 104, [1795.3 1789.7 1763.3 1821.2 1830.5    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 105, [1794.3 1788.5 1764.7 1821.5 1831.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 106, [1793.4 1787.5 1766.  1821.7 1831.4    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 107, [1792.6 1786.5 1767.3 1821.9 1831.8    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 108, [1791.8 1785.6 1768.4 1822.  1832.2    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 109, [1791.1 1784.7 1769.6 1822.1 1832.5    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 110, [1790.4 1784.  1770.6 1822.2 1832.8    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 111, [1789.7 1783.2 1771.6 1822.3 1833.1    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 112, [1789.1 1782.6 1772.6 1822.3 1833.4    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 113, [1788.6 1781.9 1773.5 1822.4 1833.6    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 114, [1788.1 1781.4 1774.4 1822.4 1833.8    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 115, [1787.6 1780.8 1775.2 1822.4 1834.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 116, [1787.1 1780.3 1776.  1822.4 1834.2    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 117, [1786.7 1779.9 1776.7 1822.4 1834.3    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 118, [1786.3 1779.4 1777.4 1822.4 1834.4    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 119, [1785.9 1779.1 1778.1 1822.4 1834.6    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 120, [1785.6 1778.7 1778.7 1822.3 1834.7    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 121, [1785.2 1778.3 1779.3 1822.3 1834.7    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 122, [1784.9 1778.  1779.9 1822.3 1834.8    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 123, [1784.6 1777.7 1780.5 1822.3 1834.9    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 124, [1784.4 1777.4 1781.  1822.2 1835.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 125, [1784.1 1777.2 1781.5 1822.2 1835.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 126, [1783.9 1776.9 1782.  1822.2 1835.1    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 127, [1783.7 1776.7 1782.4 1822.1 1835.1    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 128, [1783.4 1776.5 1782.8 1822.1 1835.1    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 129, [1783.2 1776.3 1783.3 1822.  1835.2    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 130, [1783.1 1776.1 1783.6 1822.  1835.2    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 131, [1782.9 1776.  1784.  1821.9 1835.2    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 132, [1782.7 1775.8 1784.4 1821.9 1835.2    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 133, [1782.6 1775.7 1784.7 1821.8 1835.3    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 134, [1782.4 1775.5 1785.  1821.8 1835.3    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 135, [1782.3 1775.4 1785.3 1821.7 1835.3    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 136, [1782.2 1775.3 1785.6 1821.6 1835.3    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 137, [1782.  1775.2 1785.9 1821.6 1835.3    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 138, [1781.9 1775.1 1786.2 1821.5 1835.3    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 139, [1781.8 1775.  1786.4 1821.5 1835.3    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 140, [1781.7 1774.9 1786.7 1821.4 1835.3    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 141, [1781.6 1774.8 1786.9 1821.4 1835.3    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 142, [1781.5 1774.8 1787.1 1821.3 1835.3    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 143, [1781.4 1774.7 1787.3 1821.3 1835.3    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 144, [1781.3 1774.7 1787.5 1821.2 1835.3    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 145, [1781.3 1774.6 1787.7 1821.1 1835.3    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 146, [1781.2 1774.6 1787.9 1821.1 1835.3    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 147, [1781.1 1774.5 1788.1 1821.  1835.3    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 148, [1781.1 1774.5 1788.2 1821.  1835.3    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 149, [1781.  1774.4 1788.4 1820.9 1835.3    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 150, [1780.9 1774.4 1788.5 1820.9 1835.3    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    Saving model...
    Saving model provenance info...
    DONE.

```python
> viz_model_trajs --model bayes_traj_tutorial_std-0.5_visits-3_model_v3.p --x_axis age --y_axis y1 
```
    
![png](bayes_traj_tutorial_files/bayes_traj_tutorial_69_2.png)

```python
> viz_model_trajs --model bayes_traj_tutorial_std-0.5_visits-3_model_v3.p --x_axis age --y_axis y2
```
    
![png](bayes_traj_tutorial_files/bayes_traj_tutorial_70_2.png)

The addition of the age^2 term enables representation of nonlinearity, which
can be seen in these results; in this case, the nonlinearity is an overfit to
the date (as we know the underlying trends in this simulated data set are
linear).

### *summarize_traj_model*
The *summarize_traj_model* utility allows us to inspect trajectory models
quantitatively. Run with the -h flag to see usage information:

```python
> summarize_traj_model -h
```

    usage: summarize_traj_model [-h] --model MODEL [--trajs TRAJS]
                                [--min_traj_prob MIN_TRAJ_PROB] [--hide_ic]
    
    options:
      -h, --help            show this help message and exit
      --model MODEL         Bayesian trajectory model to summarize
      --trajs TRAJS         Comma-separated list of integers indicating
                            trajectories for which to print results. If none
                            specified, results for all trajectories will be
                            printed
      --min_traj_prob MIN_TRAJ_PROB
                            The probability of a given trajectory must be at least
                            this value in order for results to be printed for that
                            trajectory. Value should be between 0 and 1 inclusive.
      --hide_ic             Use this flag to hide compuation and display of
                            information criterai (BIC and WAIC2), which can take
                            several moments to compute.


Lets use this utility to inspect the model that used *intercept*, *age*, and
*age^2* as predictors:


```python
!summarize_traj_model --model bayes_traj_tutorial_std-0.5_visits-3_model_v3.p
```

                                     Summary                                  
    ==========================================================================
    Num. Trajs:         5
    Trajectories:       0,1,2,3,4                               
    No. Observations:   9000           
    No. Groups:         3000           
    WAIC2:              27772     
    BIC1:               -13030    
    BIC2:               -13000    
    
                             Summary for Trajectory 0                         
    ==========================================================================
    No. Observations:                             1779
    No. Groups:                                    593
    % of Sample:                                  19.8
    Odds Correct Classification:                 509.3
    Ave. Post. Prob. of Assignment:               0.99
    
                      Residual STD       Precision Mean      Precision Var    
    --------------------------------------------------------------------------
    y1                    0.51                3.86               0.0060       
    y2                    0.50                4.05               0.0066       
    
                          coef                STD           [95% Cred. Int.]  
    --------------------------------------------------------------------------
    intercept (y1)       0.070               0.012           0.046    0.093   
    age (y1)             0.163               0.000           0.163    0.164   
    age^2 (y1)           -0.001              0.000           -0.001  -0.001   
    intercept (y2)       4.245               0.012           4.221    4.268   
    age (y2)             0.037               0.000           0.037    0.037   
    age^2 (y2)           -0.000              0.000           -0.000  -0.000   
    
    
                             Summary for Trajectory 1                         
    ==========================================================================
    No. Observations:                             1788
    No. Groups:                                    596
    % of Sample:                                  19.9
    Odds Correct Classification:                  61.2
    Ave. Post. Prob. of Assignment:               0.94
    
                      Residual STD       Precision Mean      Precision Var    
    --------------------------------------------------------------------------
    y1                    0.51                3.85               0.0060       
    y2                    0.50                4.05               0.0066       
    
                          coef                STD           [95% Cred. Int.]  
    --------------------------------------------------------------------------
    intercept (y1)       -1.477              0.012           -1.501  -1.454   
    age (y1)             0.146               0.000           0.146    0.147   
    age^2 (y1)           -0.001              0.000           -0.001  -0.001   
    intercept (y2)       3.668               0.012           3.644    3.691   
    age (y2)             -0.008              0.000           -0.008  -0.008   
    age^2 (y2)           -0.000              0.000           -0.000  -0.000   
    
    
                             Summary for Trajectory 2                         
    ==========================================================================
    No. Observations:                             1785
    No. Groups:                                    595
    % of Sample:                                  19.8
    Odds Correct Classification:                 178.0
    Ave. Post. Prob. of Assignment:               0.98
    
                      Residual STD       Precision Mean      Precision Var    
    --------------------------------------------------------------------------
    y1                    0.50                4.00               0.0064       
    y2                    0.50                4.00               0.0064       
    
                          coef                STD           [95% Cred. Int.]  
    --------------------------------------------------------------------------
    intercept (y1)       5.887               0.012           5.863    5.910   
    age (y1)             -0.093              0.000           -0.093  -0.092   
    age^2 (y1)           0.000               0.000           0.000    0.000   
    intercept (y2)       2.505               0.012           2.481    2.528   
    age (y2)             0.031               0.000           0.030    0.031   
    age^2 (y2)           -0.000              0.000           -0.000  -0.000   
    
    
                             Summary for Trajectory 3                         
    ==========================================================================
    No. Observations:                             1815
    No. Groups:                                    605
    % of Sample:                                  20.2
    Odds Correct Classification:                  65.6
    Ave. Post. Prob. of Assignment:               0.94
    
                      Residual STD       Precision Mean      Precision Var    
    --------------------------------------------------------------------------
    y1                    0.50                4.03               0.0065       
    y2                    0.51                3.89               0.0060       
    
                          coef                STD           [95% Cred. Int.]  
    --------------------------------------------------------------------------
    intercept (y1)       3.722               0.012           3.698    3.746   
    age (y1)             0.005               0.000           0.004    0.005   
    age^2 (y1)           -0.000              0.000           -0.000  -0.000   
    intercept (y2)       2.540               0.012           2.517    2.564   
    age (y2)             0.055               0.000           0.055    0.055   
    age^2 (y2)           -0.000              0.000           -0.000  -0.000   
    
    
                             Summary for Trajectory 4                         
    ==========================================================================
    No. Observations:                             1833
    No. Groups:                                    611
    % of Sample:                                  20.4
    Odds Correct Classification:                 146.0
    Ave. Post. Prob. of Assignment:               0.97
    
                      Residual STD       Precision Mean      Precision Var    
    --------------------------------------------------------------------------
    y1                    0.50                4.02               0.0064       
    y2                    0.50                3.98               0.0063       
    
                          coef                STD           [95% Cred. Int.]  
    --------------------------------------------------------------------------
    intercept (y1)       5.297               0.012           5.273    5.320   
    age (y1)             -0.026              0.000           -0.027  -0.026   
    age^2 (y1)           0.000               0.000           0.000    0.000   
    intercept (y2)       6.311               0.012           6.287    6.334   
    age (y2)             -0.056              0.000           -0.056  -0.056   
    age^2 (y2)           0.000               0.000           0.000    0.000   
    


The print-out shows information for the model as a whole in the 'Summary'
section at the top. Included here are three information criterion measures:
BIC1, BIC2, and WAIC2. These measures reward goodness of fit while penalizing
model complexity. The penalty terms is a function of *N*; BIC1 takes *N* to be
the number of observations, while BIC2 takes *N* to be the number of groups.
Generally, a higher BIC value is preferred. WAIC2 is an alternative information
criterion measure which has been recommended in the Bayesian context; lower
WAIC2 scores are preferred.

Below the overall summary is per-trajectory information. This includes posterior
estimates for residual precisions and predictor coefficients. Note that STD is
not the same as *standard error*, and 95% Cred. Int. (credible interval) is not
the same as a *confidence interval*. Rather, these are quantities that describe
the Bayesian posterior distribution over these parameters. (Side note: the
trajectory fitting routine uses a technique called *variational inference*,
which is fast and scales well, but is known to *underestimate* posterior
variances. Posterior standard deviations and credible intervals should be
interpreted with this in mind). 

Also shown for each trajectory are the *odds of correct classification* and the
*average posterior probability of assignment*. A rule of thumb for the odds of
correct classification is that it should be greater than 5. For the average
posterior probability of assignment, values of .7 or greater for each trajectory
are recommended as rules of thumb.

Now let's inspect the model generated using only *intercept* and *age*:

```python
> summarize_traj_model --model bayes_traj_tutorial_std-0.5_visits-3_model_v2.p
```

                                     Summary                                  
    ==========================================================================
    Num. Trajs:         5
    Trajectories:       0,1,4,5,6                               
    No. Observations:   9000           
    No. Groups:         3000           
    WAIC2:              27228     
    BIC1:               -12765    
    BIC2:               -12744    
    
                             Summary for Trajectory 0                         
    ==========================================================================
    No. Observations:                             1779
    No. Groups:                                    593
    % of Sample:                                  19.8
    Odds Correct Classification:                 674.7
    Ave. Post. Prob. of Assignment:               0.99
    
                      Residual STD       Precision Mean      Precision Var    
    --------------------------------------------------------------------------
    y1                    0.50                4.03               0.0065       
    y2                    0.50                4.07               0.0066       
    
                          coef                STD           [95% Cred. Int.]  
    --------------------------------------------------------------------------
    intercept (y1)       5.963               0.012           5.940    5.987   
    age (y1)             -0.014              0.000           -0.014  -0.014   
    intercept (y2)       5.914               0.012           5.891    5.938   
    age (y2)             -0.014              0.000           -0.014  -0.013   
    
    
                             Summary for Trajectory 1                         
    ==========================================================================
    No. Observations:                             1839
    No. Groups:                                    613
    % of Sample:                                  20.4
    Odds Correct Classification:                  61.8
    Ave. Post. Prob. of Assignment:               0.94
    
                      Residual STD       Precision Mean      Precision Var    
    --------------------------------------------------------------------------
    y1                    0.50                3.96               0.0062       
    y2                    0.50                4.02               0.0065       
    
                          coef                STD           [95% Cred. Int.]  
    --------------------------------------------------------------------------
    intercept (y1)       4.001               0.012           3.977    4.024   
    age (y1)             -0.015              0.000           -0.015  -0.015   
    intercept (y2)       3.939               0.012           3.916    3.963   
    age (y2)             -0.014              0.000           -0.014  -0.013   
    
    
                             Summary for Trajectory 4                         
    ==========================================================================
    No. Observations:                             1794
    No. Groups:                                    598
    % of Sample:                                  19.9
    Odds Correct Classification:                 211.1
    Ave. Post. Prob. of Assignment:               0.98
    
                      Residual STD       Precision Mean      Precision Var    
    --------------------------------------------------------------------------
    y1                    0.50                4.02               0.0065       
    y2                    0.50                4.01               0.0064       
    
                          coef                STD           [95% Cred. Int.]  
    --------------------------------------------------------------------------
    intercept (y1)       3.916               0.012           3.893    3.940   
    age (y1)             -0.034              0.000           -0.034  -0.033   
    intercept (y2)       4.059               0.012           4.036    4.083   
    age (y2)             -0.016              0.000           -0.016  -0.015   
    
    
                             Summary for Trajectory 5                         
    ==========================================================================
    No. Observations:                             1827
    No. Groups:                                    609
    % of Sample:                                  20.3
    Odds Correct Classification:                 167.4
    Ave. Post. Prob. of Assignment:               0.98
    
                      Residual STD       Precision Mean      Precision Var    
    --------------------------------------------------------------------------
    y1                    0.50                4.03               0.0065       
    y2                    0.50                3.99               0.0063       
    
                          coef                STD           [95% Cred. Int.]  
    --------------------------------------------------------------------------
    intercept (y1)       4.994               0.012           4.971    5.017   
    age (y1)             -0.015              0.000           -0.015  -0.015   
    intercept (y2)       4.964               0.012           4.940    4.987   
    age (y2)             -0.014              0.000           -0.015  -0.014   
    
    
                             Summary for Trajectory 6                         
    ==========================================================================
    No. Observations:                             1761
    No. Groups:                                    587
    % of Sample:                                  19.6
    Odds Correct Classification:                  74.2
    Ave. Post. Prob. of Assignment:               0.95
    
                      Residual STD       Precision Mean      Precision Var    
    --------------------------------------------------------------------------
    y1                    0.50                4.03               0.0065       
    y2                    0.50                3.94               0.0062       
    
                          coef                STD           [95% Cred. Int.]  
    --------------------------------------------------------------------------
    intercept (y1)       5.020               0.012           4.996    5.044   
    age (y1)             -0.035              0.000           -0.035  -0.035   
    intercept (y2)       4.861               0.012           4.837    4.885   
    age (y2)             -0.013              0.000           -0.014  -0.013   
    


As expected, the more parsimonious model (using only *intercept* and *age* as
predictors) results in better information criteria scores. 

# Bayesian Trajectory Analysis: Binary Target Variables

We now turn our attention to the case of binary target variables. In this case,
the algorithm models the data as a mixture of logistic regressors.

As before, we begin by printing the first few rows of a synthetically generated
data set that containts two binary target variables (*y1* and *y2*) and a time
variable, *x*. In this data set, there are four distinct trajectories, with 200
individuals in each trajectory and 10 time points per individual. 'sid' is the
subject identifier column, 'traj_gt' indicates the ground-truth trajectory
assignment, and 'intercept' is a column of 1s. The four trajectories were
generated to have the following probability patterns with respect to the time
variable, *x*:

* Trajectory 1 -- y1: decreasing, y2: decreasing, shifted left
* Trajectory 2 -- y1: decreasing, y2: decreasing, shifted right
* Trajectory 3 -- y1: increasing, y2: decreasing, shifted left
* Trajectory 4 -- y1: increasing, y2: decreasing, shifted right

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sid</th>
      <th>mu1</th>
      <th>mu2</th>
      <th>y1</th>
      <th>y2</th>
      <th>intercept</th>
      <th>x</th>
      <th>traj_gt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>0.994590</td>
      <td>0.999554</td>
      <td>1</td>
      <td>1</td>
      <td>1.0</td>
      <td>-5.214084</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>0.976913</td>
      <td>0.998064</td>
      <td>1</td>
      <td>1</td>
      <td>1.0</td>
      <td>-3.745119</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>0.939964</td>
      <td>0.994785</td>
      <td>1</td>
      <td>1</td>
      <td>1.0</td>
      <td>-2.750904</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>0.789713</td>
      <td>0.978610</td>
      <td>1</td>
      <td>1</td>
      <td>1.0</td>
      <td>-1.323194</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>0.699091</td>
      <td>0.965874</td>
      <td>1</td>
      <td>1</td>
      <td>1.0</td>
      <td>-0.842974</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.0</td>
      <td>0.362511</td>
      <td>0.873859</td>
      <td>1</td>
      <td>1</td>
      <td>1.0</td>
      <td>0.564483</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.0</td>
      <td>0.153625</td>
      <td>0.688593</td>
      <td>0</td>
      <td>0</td>
      <td>1.0</td>
      <td>1.706451</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.0</td>
      <td>0.061672</td>
      <td>0.444659</td>
      <td>0</td>
      <td>1</td>
      <td>1.0</td>
      <td>2.722275</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.0</td>
      <td>0.021875</td>
      <td>0.214112</td>
      <td>0</td>
      <td>0</td>
      <td>1.0</td>
      <td>3.800312</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.0</td>
      <td>0.005367</td>
      <td>0.061681</td>
      <td>0</td>
      <td>0</td>
      <td>1.0</td>
      <td>5.222111</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1.0</td>
      <td>0.993601</td>
      <td>0.999472</td>
      <td>1</td>
      <td>1</td>
      <td>1.0</td>
      <td>-5.045176</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1.0</td>
      <td>0.978141</td>
      <td>0.998169</td>
      <td>1</td>
      <td>1</td>
      <td>1.0</td>
      <td>-3.801039</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>1.0</td>
      <td>0.935340</td>
      <td>0.994357</td>
      <td>1</td>
      <td>1</td>
      <td>1.0</td>
      <td>-2.671766</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>1.0</td>
      <td>0.834289</td>
      <td>0.983957</td>
      <td>1</td>
      <td>0</td>
      <td>1.0</td>
      <td>-1.616334</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>1.0</td>
      <td>0.577248</td>
      <td>0.943293</td>
      <td>0</td>
      <td>1</td>
      <td>1.0</td>
      <td>-0.311488</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>

### Prior Generation

The procedure for generating priors in this case proceeds as above. Here we
will specifically set the priors over the predictor coefficients to be
zero-centered Gaussian distributions with unit variance:

```python
> generate_prior --num_trajs 4 --preds intercept,x --targets y1,y2 --in_data binary_data_4.csv  --coef y1,intercept,0,1 --coef y1,x,0,1 --coef y2,intercept,0,1 --coef y2,x,0,1 --out_file binary_data_4_prior.p --groupby sid
```

    Reading data...
    Optimization terminated successfully.
             Current function value: 0.693140
             Iterations 3
    Optimization terminated successfully.
             Current function value: 0.466741
             Iterations 6
    ---------- Prior Info ----------
    alpha: 5.51e-01
     
    y1 intercept (mean, std): (0.00e+00, 1.00e+00)
    y1 x (mean, std): (0.00e+00, 1.00e+00)
     
    y2 intercept (mean, std): (0.00e+00, 1.00e+00)
    y2 x (mean, std): (0.00e+00, 1.00e+00)


As before, let's visualize draws from this prior:


```python
> viz_data_prior_draws --data_file binary_data_4.csv --prior binary_data_4_prior.p --num_draws 20 --x_axis x  --y_axis y1
```

![png](bayes_traj_tutorial_files/bayes_traj_tutorial_85_2.png)
    
### Model Fitting and Visualization

Model fitting proceeds as in the continuous case. Here we will fit using both
target variables.


```python
> bayes_traj_main --in_csv binary_data_4.csv --prior binary_data_4_prior.p --targets y1,y2 --groupby sid --out_model binary_data_4_model.p --verbose --iters 50 --alpha .55
```

    Reading prior...
    Reading data...
    Fitting...
    Initializing parameters...
    iter 1, [6560.7 1215.2  117.9  106.2    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 2, [5700.8 2099.2  122.8   77.2    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 3, [4643.9 3260.5   60.5   35.1    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 4, [4188.2 3676.3   44.6   90.9    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 5, [3480.1 3552.   216.9  751.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 6, [2133.  2335.3 1259.6 2272.1    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 7, [1866.6 1512.1 2064.7 2556.7    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 8, [1906.2 1713.3 2093.3 2287.2    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 9, [1928.2 1841.2 2071.8 2158.9    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 10, [1938.5 1903.7 2061.5 2096.3    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 11, [1951.  1925.8 2049.  2074.2    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 12, [1952.7 1938.8 2047.3 2061.2    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 13, [1952.8 1953.7 2047.2 2046.3    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 14, [1957.5 1951.6 2042.5 2048.4    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 15, [1962.1 1963.5 2037.9 2036.5    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 16, [1957.1 1961.2 2042.9 2038.8    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 17, [1959.4 1962.8 2040.6 2037.2    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 18, [1963.2 1971.8 2036.8 2028.2    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 19, [1959.2 1969.7 2040.8 2030.3    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 20, [1964.6 1973.8 2035.4 2026.2    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 21, [1965.4 1976.9 2034.6 2023.1    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 22, [1961.6 1980.7 2038.4 2019.3    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 23, [1963.3 1984.9 2036.7 2015.1    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 24, [1962.1 1984.5 2037.9 2015.5    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 25, [1962.1 1984.2 2037.9 2015.8    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 26, [1959.3 1990.3 2040.7 2009.7    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 27, [1964.4 1984.7 2035.6 2015.3    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 28, [1960.7 1991.4 2039.3 2008.6    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 29, [1966.6 1993.4 2033.4 2006.6    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 30, [1965.7 1993.  2034.3 2007.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 31, [1964.7 1993.7 2035.3 2006.3    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 32, [1959.2 1999.4 2040.8 2000.6    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 33, [1967.2 2002.3 2032.8 1997.7    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 34, [1966.4 1999.9 2033.6 2000.1    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 35, [1964.3 1993.1 2035.7 2006.9    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 36, [1963.8 1996.6 2036.2 2003.4    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 37, [1967.7 1995.8 2032.3 2004.2    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 38, [1964.9 1997.7 2035.1 2002.3    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 39, [1964.3 2002.2 2035.7 1997.8    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 40, [1967.4 1993.8 2032.6 2006.2    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 41, [1964.6 1998.9 2035.4 2001.1    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 42, [1965.8 1996.8 2034.2 2003.2    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 43, [1961.2 1993.5 2038.8 2006.5    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 44, [1967.  1998.8 2033.  2001.2    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 45, [1961.  1998.4 2039.  2001.6    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 46, [1963.5 1999.  2036.5 2001.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 47, [1963.4 1995.4 2036.6 2004.6    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 48, [1967.7 1993.6 2032.3 2006.4    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 49, [1962.3 2000.8 2037.7 1999.2    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    iter 50, [1962.8 1992.9 2037.2 2007.1    0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0.     0. ]
    Saving model...
    Saving model provenance info...
    DONE.


Finally, let's visualize the results and inspect the *y1* and *y2* trends:

```python
> viz_model_trajs --model binary_data_4_model.p --x_axis x --y_axis y1 --traj_markers o,s,^,d
```
    
![png](bayes_traj_tutorial_files/bayes_traj_tutorial_90_2.png)    

```python
> viz_model_trajs --model binary_data_4_model.p --x_axis x --y_axis y2 --traj_markers o,s,^,d
```
    
![png](bayes_traj_tutorial_files/bayes_traj_tutorial_91_2.png)
    
The detected trajectories capture the underlying trends. Note that the
trajectory numbers assigned by the algorithm are arbitrary.
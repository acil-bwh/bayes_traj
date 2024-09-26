---
title: 'bayes_traj: A Python package for Bayesian trajectory analysis'
tags:
  - Python
  - trajectories
  - Bayesian
authors:
  - name: James C. Ross
    orcid: 0000-0002-2338-3644
    affiliation: 1
affiliations:
 - name: Department of Radiology, Brigham and Women’s Hospital, Harvard Medical School, Boston, MA, USA
   index: 1

date: 26 Septembe 2024
bibliography: paper.bib

---

# Statement of need 

Trajectory analysis broadly refers to the application of methods to
longitudinal data to identify distinct patterns of change and assign study
subjects to their most likely trajectory group.
Although trajectory analysis has been applied in mutiple
domains, the motivation for developing **bayes_traj** has been to improve our
understanding of
heterogeneity in the context of chronic obstructive pulmonary disease (COPD), a
leading cause of death worldwide. Research has shown that there are multiple
patterns of lung function development and decline, with some patterns associated
with greater risk of developing COPD [@lange2015lung]. Researchers have applied techniques
of trajectory analysis to longitudinal measures of lung function to delineate
distinct patterns of progression for further analysis [@agusti2019lung].
Existing trajectory approaches
are predominantly frequentist in nature and use maximum likelihood to identify point
estimates of unknown parameters. These approaches do not permit incorporation of prior informaiton.
Challenges arise when study cohorts lack sufficient longitudinal data characteristics
to adequately power frequentist-based trajectory algorithms. Furthermore, there
is a growing recognition that COPD is better conceived of as a multi-faceted syndrome,
requiring consideration of other disease facets (such as clinical presentation
and structural assesment from medical images) [@lowe2019copdgene]. This
motivates development and
application of scalable approaches that can simultaneously model distinct progression
patterns across multiple health measures, especially in data-limited scenarios.

# Summary

**bayes_traj** is a Python package developed to perform Bayesian trajectory analysis.
Although our primary motivation is to improve understanding of COPD heterogeneity,
the package makes no domain-specific assumptions and is generally applicable.

**bayes_traj** has several distinguishing features:
* It can simultaneously model multiple continuous and binary target
variables as a functions of predictor variables.
* Given an estimate of the number of trajectories, it uses
Bayesian nonparametrics (Dirichlet Process mixture modeling) to automatically
identify the number of groups in a data set. 
* It makes the assumption that target variables are conditionally independent
given trajectory assignments, enabling the algorithm to scale well to multiple
targets.
* Bayesian approximate inference is performed using coordinate ascent variational
inference, which is fast as scales well to large data sets.
* Independantly estimates residual variance posteriors for each trajectory and
each target variable
* Allows specification of random effects for continuous target variables using
unstructured covariance matrices
* Provides a suite of tools to facilitate prior specification, model
visualization, and summary statistic computation. 

These features make **bayes_traj** a great fit for investigating COPD
heterogeniety, and we have used it in several publications. In an early
implementation, we used it to identify disease subtypes using five measures
of emphysema computed from medical images [@ross2016bayesian]. Later we used
it to identify distinct lung
function trajectories in one cohort and to then probabilistically assign
individuals in another cohort to their most likely trajectory for further
analysis [@ross2018longitudinal]. Recently, we applied **bayes_traj** to
multiple measures of lung function in a cohort of middle-aged and
older adults, using an informative prior to capture known information
about lung function in early adulthood [@ross2024dysanapsis].

# State of the field

Van der Nest et al. provide an excellent overview of the various approaches
and implementations for trajectory analysis [@van2020overview]. The methods
covered in that paper are all frequentist in nature. To our knowledge, the
only other Bayesian approach for modeling trajectories is Zang and Max 
[@zang2022bayesian]; they implement a form of group-based trajectory modeling
(GBTM) and use MCMC for approximate Bayesian inference.

# Acknowledgements

Continued development of **bayes_traj** is supported by the US National Heart,
Lung, and Blood Institute (1R01-HL164380-01).
**bayes_traj** would not be possible without numerous other open-source
Python packages, especially numpy [@harris2020array], scipy [@2020SciPy-NMeth],
matplotlib [@Hunter_2007], PyTorch [@NEURIPS2019_9015], and
pandas [@reback2020pandas]. Special thanks also to Tinting Zhao for developing
variational update equations for inference over binary target variables.

# References
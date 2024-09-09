Here we describe our Bayesian formulation, beginning with a brief review of
Dirichlet process mixtures, a key element of our model.

# Dirichlet Process Mixtures
[Ferguson (1973)](https://www.jstor.org/stable/pdf/2958008.pdf?casa_token=tzi5CgijJnAAAAAA:P1JvaSlGmHtc9-8jJSib5nzt3lO_GgOBi5iiZth2bg-sbyMotPV7flSvBsIxiLTmYXdXAJSVA6qWIgLAbdvWtTOJiw9VNjGOUWy9vTzIhVgNRjCl6zA)
first introduced the Dirichlet process (DP) as a
measure on measures. It is parameterized by a base measure,
\\(\mathit{G}\_0\\), and a positive scaling parameter \\( \alpha \\):
$$
  \mathit{G} \vert \\{ \mathit{G}\_0, \alpha \\} \sim \text{DP}\left( \mathit{G}\_0, \alpha \right)
$$
The notion of a Dirichlet process mixture (DPM) arises if we treat the
\\( k^{th} \\) draw from \\( \mathit{G} \\) as a parameter of the distribution over some
observation ([Antoniak (1974)](https://www.jstor.org/stable/pdf/2958336.pdf?casa_token=JDOSeMgW_a4AAAAA:sYAU_MBeUgA113mrQnVM2SNbRjqMGvzBHMT8PUDGH-LQlliLQRlWbXKtD5Gl4ycRfCiTH4ABRPJnaY-UV-sA7HhwJzLT39QrVKWLRVPr2NeZRnLX-O4)).
DPMs can be interpreted as mixture models with an
infinite number of mixture components.

More recently,
[Blei and Jordan (2006)](https://www.cs.princeton.edu/courses/archive/fall07/cos597C/readings/BleiJordan2005.pdf)
described a variational inference
algorithm for DPMs using the stick-breaking construction introduced in
[Sethuraman (1991)](https://groups.seas.harvard.edu/courses/cs281/papers/sethuraman-1994.pdf).
The stick-breaking construction represents \\( G \\) as
$$
  \pi\_{k}( \mathbf{v} ) = \mathbf{v}\_{k}\prod^{k-1}_{j=1}\left( 1-\mathbf{v}\_{j}\right)
$$

$$
\mathit{G} = \sum\_{i=1}^{\infty}\pi\_{i}\left( \mathbf{v}\right)\delta\_{\eta\_{i}^{\ast}} 
$$
where \\( \delta\_{\eta\_{i}^{\ast}}   \\) is the Kronecker delta, and the
\\( \mathbf{v}\_{i} \\) are distributed according to a beta distribution:
\\( \mathbf{v}\_{i} \sim \text{Beta}\left( 1, \alpha \right) \\), and 
\\( \eta_{i}^{\ast} \sim \mathit{G}\_0 \\). We use a DPM in our model to
automatically identify the number of trajectories that best explain our data.


# Model Formulation
We consider a collection of multiple longitudinally observed target variables,
which can be continuous, binary, or a combination.
We let \\(y_{g,i,d}\\) represent the observation for individual \\(g\\)
\\( (g=1,\dots,G) \\) at the \\( i^{th} \\) occasion \\( (i=1,\dots,n_g) \\) for
target variable \\(d\\). 
Similarly, \\(x_{g,i,m}\\) represents predictor \\(m\\) \\( (m=1,\dots,M)\\) for
individual \\(g\\) on occasion \\(i\\).
Here, \\(G\\) is the total number of individuals in the data sample, \\(n_g\\)
is the number of observations per individual, and \\(M\\) is the number of
predictors. The likelihood in our formulation factorizes into two terms:
$$
p(\mathbf{Y}\mid \mathbf{Z}, \mathbf{W}, \boldsymbol{\lambda}, \mathbf{U}) =
p(\mathbf{Y}_c\mid \mathbf{Z}, \mathbf{W}_c, \boldsymbol{\lambda}, \mathbf{U})
p(\mathbf{Y}_b\mid \mathbf{Z}, \mathbf{W}_b)
$$
where we distinguish between the collection of \\(D_c\\) continuous target
variables, \\(Y_c\\), and the collection of \\(D_b\\) binary target variables,
\\(Y_b\\). The likelihood factors are given by:

$$
p(\mathbf{Y}_c\mid \mathbf{Z}, \mathbf{W}_c, \boldsymbol{\lambda}, \mathbf{U})=
\prod\_{k=1}^{\infty} \prod\_{g=1}^{G}\prod\_{d_c=1}^{D_c}\left[
\prod\_{i=1}^{n_g}\mathcal{N}\left(y\_{g,i,d_c} \mid
(\mathbf{w}\_{\cdot, d_c, k} + \mathbf{u}\_{g, d_c, k})^{T} \mathbf{x}\_{g,i,\cdot},
\lambda\_{d_c,k}^{-1}\right)\right]^{z\_{g, k}}
$$
and
$$
p(\mathbf{Y}_b\mid \mathbf{Z}, \mathbf{W}_b)=\prod\_{k=1}^{\infty}\prod\_{g=1}^{G} \prod\_{d_b=1}^{D_b} 
\left[\prod\_{i=1}^{n_g}
\frac{\exp(\mathbf{w}\_{\cdot, d_b, k}^{T} \mathbf{x}\_{g,i,\cdot})^{y\_{g,i,d_b}}}
{1+\exp(\mathbf{w}\_{\cdot, d_b, k}^{T} \mathbf{x}\_{g,i,\cdot})}
\right]^{z\_{g,k}}.
\label{YbLike}
$$

We formulate our model as a DPMM, which can be interpreted as a mixture model
with a potentially infinite number of mixture components
([Antoniak (1974)](https://www.jstor.org/stable/pdf/2958336.pdf?casa_token=JDOSeMgW_a4AAAAA:sYAU_MBeUgA113mrQnVM2SNbRjqMGvzBHMT8PUDGH-LQlliLQRlWbXKtD5Gl4ycRfCiTH4ABRPJnaY-UV-sA7HhwJzLT39QrVKWLRVPr2NeZRnLX-O4)).
The \\(G \times \infty\\) binary indicator matrix, \\(\mathbf{Z}\\), represents
the association between subjects and the potentially infinite number of latent
regression functions (trajectories), and \\(k\\) represents the group membership for each individual.
In the case of \\(\mathbf{Y}_c\\) this formulation can be see as mixture of linear
regressors, and in the case of \\(\mathbf{Y}_b\\) it can be seen as an infinite
mixture of logistic regressors.  

\\( \mathbf{W}\_c \\) represents the \\(M \times D_c \times \infty\\) matrix of
predictor coefficients for the linear regressors, and  \\( \mathbf{W}\_b \\)
represents the \\(M \times D_b \times \infty\\) matrix of predictor coefficients for
the logistic regressors. We put Gaussian priors over both \\( \mathbf{W}\_c\\) and
\\(\mathbf{W}\_b\\), where \\( \boldsymbol{\mu}\_0 \\) and \\( \boldsymbol{\lambda}\_0 \\)
capture practioner believe about coeffient values:

$$
p\left(\mathbf{W}\_c \right)=
\prod\_{m=1}^{M}
\prod\_{d_c=1}^{D_c}
\prod\_{k=1}^{\infty}
\mathcal{N}\left(w\_{m, d_c, k} \mid
\mu\_{0\_{c_{m, d_c}}},
\lambda\_{0\_{c_{m, d_c}}}^{-1}\right)
$$

and
$$
p\left(\mathbf{W}\_b \right)=
\prod_\{m=1}^{M}
\prod_\{d_b=1}^{D_b}
\prod_\{k=1}^{\infty}
\mathcal{N}\left(w_{m, d_b, k}
\mid \mu\_{0\_{b_m, d_b}},
\lambda\_{0\_{b_m, d_b}}^{-1}\right)
\label{Wbprior}
$$

\\( \mathbf{U} \\) represents the \\(G \times D_c \times \infty\\) matrix of
(optional) random effects for the continuous target variables:

$$
p\left(\mathbf{U}\right)=
\prod_\{g=1}^{G}
\prod_\{d=1}^{D_c}
\prod_\{k=1}^{\infty}
\mathcal{N}\left(\mathbf{u}_\{g, d_c, k}
\mid \mathbf{0}, \mathbf{\Sigma_0} \right)
$$

(Note that the dimension length of random effect vectors is generally less than
\\( \mathbf{M} \\), the number of predictors. Elements of \\( \mathbf{u}\_{g, d\_c, k} \\)
corresponding to predictors with no random effects are set to 0.) Here we assume that
the random effects have mean \\( \mathbf{0} \\), and the unstructured covariance
matrix \\( \boldsymbol{\Sigma}_0\\) captures prior believe about predictor
variability within a trajectory subgroup.


We learn the residual precisions, \\( \boldsymbol{\lambda} \\), for each of the
\\(D_c \times \infty\\) linear regressors, and place gamma priors over these terms:
$$
p(\boldsymbol{\lambda})=
\prod\_{k=1}^{\infty}
\prod\_{d_c=1}^{D_c}
\mathrm{Gam}\left(\lambda\_{d_c, k} \mid a\_{0_{d_c}}, b\_{0_{d_c}}\right).
$$

The nonparametric prior distribution over \\( \mathbf{Z} \\) is given by:
$$
p(\mathbf{Z} \mid \mathbf{v})=
\prod\_{g=1}^{G} \prod\_{k=1}^{\infty}\left(v_{k} \prod\_{j=1}^{k-1}\left(1-v_{j}\right)\right)^{z_{g, k}}
$$
This can be considered a \\( G\times \infty \\) multinomial distribution with
parameters drawn for a DP using the stick-breaking construction
(see [Blei 2006](http://www.cs.princeton.edu/courses/archive/fall11/cos597C/reading/BleiJordan2005.pdf)
and [Sethuraman 1994](https://www.jstor.org/stable/pdf/24305538.pdf?casa_token=Os5nKQ7AeAMAAAAA:BbFJcPpxB-O10ntvO7ii9ruY3oeIAeTiAoQZ2X5rF9BhnaXIAV2rLqcx78Vji-vqRtOCJAeB1kWw_kcV1NvhmHssQamAxLb87mz0o_9oilKInFy_K80) for details),
where the elements
of \\(\mathbf{v}\\) are drawn from a beta distribution with concentration parameter \\(\alpha\\):
$$
p(\mathbf{v})=
\prod\_{k=1}^{\infty}
\mathrm{Beta}\left(v_{k} \mid 1, \alpha\right).
\label{prior_v}
$$
The concentration parameter \\(\alpha\\) captures the practitioner’s prior
belief about whether there are fewer groups (low scale parameter value) or
more groups (larger scale parameter value).
A benefit of the non-parametric framework is that the number of components
that best describe the observed data is automatically determined conditioned on
this value.

With these terms defined, the joint density is given as:
$$
p\left(\mathbf{Y}\_c, 
\mathbf{Y}\_b, 
\mathbf{W}\_c, 
\mathbf{W}\_b, 
\boldsymbol{\lambda}, 
\mathbf{Z}, 
\mathbf{v},
\mathbf{U}, 
\mid 
\mathbf{X},
\boldsymbol{\mu}\_{c}, 
\boldsymbol{\lambda}\_c, 
\boldsymbol{\mu}\_b, 
\boldsymbol{\lambda}\_b, 
\mathbf{a}\_0, 
\mathbf{b}\_0, 
\alpha,
\mathbf{\Sigma}\_0, 
\right) = \\\
p(\mathbf{Y}\_c \mid \mathbf{Z}, \mathbf{W}\_c, \boldsymbol{\lambda}, \mathbf{U})
p(\mathbf{Y}\_b \mid \mathbf{Z}, \mathbf{W}\_b)
p\left(\mathbf{W}\_c \mid \boldsymbol{\mu}\_c, \boldsymbol{\lambda}\_c\right)
p\left(\mathbf{W}\_b \mid \boldsymbol{\mu}\_b, \boldsymbol{\lambda}\_{b}\right) \\\
p(\boldsymbol{\lambda} \mid \mathbf{a}\_0, \mathbf{b}\_0)
p(\mathbf{Z} \mid \mathbf{v}) 
p(\mathbf{v} \mid \alpha)
$$

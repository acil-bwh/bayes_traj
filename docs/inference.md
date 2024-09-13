
# Variational Inference
## Overview
Frequently with complex models, obtaining the exact posterior distribution can
be intractable. While Markov Chain Monte Carlo (MCMC) 
methods offer a systematic approach to sample from the posterior distribution,
they can be slow in high-dimensional parameter spaces
([Hastings (1970)](https://www.jstor.org/stable/pdf/2334940.pdf?casa_token=ZEcTsBtCLZkAAAAA:-f5zD4VUydXnjBQe5ErtLRajoF6QScZsOTatDPDjQiawsV_HAjdiNI6T4HmdnmuEbXq8oWXF6DXvQUKrZbNCgc3laLqrW0NWEtyhSS3LMmZa96Odlw)).
An alternative is variational inference, a form
of Bayesian approximate inference that tends to be fast and scales well to large
data sets ([Jordan (1999)](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=642dba1a71033b9587e4cbcb993a8016f012dc00)).
It typically makes a factorization assumption over the approximate
posterior distribution of interest, and it turns an inference problem into an
optimization problem by finding the
approximate posterior that minimizes the Kullback-Leibler divergence to the true
posterior distribution 
(see [Jordan (1999)](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=642dba1a71033b9587e4cbcb993a8016f012dc00),
[Jaakkola (2001)](http://people.csail.mit.edu/people/tommi/papers/Jaa-nips00-tutorial.pdf), and
[Blei (2017)](http://www.cs.columbia.edu/~blei/fogm/2018F/materials/BleiKucukelbirMcAuliffe2017.pdf)).

In the case of conjugate priors (a prior is conjugate when the posterior
distribution belongs to the same family of probability distributions as the
prior distribution given a specific likelihood function), there is a straightforward
procedure for deriving variational update eqautions. In the case of
non-conjugate priors, alternative approaches are needed.

For our model, the posterior distribution we wish to estimate is given by:
$$
p\left(\mathbf{W}\_c,
\mathbf{W}\_b,
\boldsymbol{\lambda},
\mathbf{Z},
\mathbf{v},
\mathbf{U}
\mid
\mathbf{Y}\_c,
\mathbf{Y}\_b,
\mathbf{X},
\boldsymbol{\mu}\_{0_c},
\boldsymbol{\lambda}\_{0_c},
\boldsymbol{\mu}\_{0_b},
\boldsymbol{\lambda}\_{0_b},
\mathbf{a}\_0,
\mathbf{b}\_0,
\alpha,
\boldsymbol{\Sigma}_0
\right)
$$
This posterior probability is approximated using variational inference. The
standard mean field variational inference approach is to assume a factorized
approximation of this distribution, in our case:
$$
p^\*(\mathbf{W}\_{c})
p^\*(\mathbf{W}\_{b})
p^\*(\boldsymbol{\lambda})
p^\*(\mathbf{Z})
p^\*(\mathbf{v})
p^\*(\mathbf{U})
$$
In order to derive the expression for each of these factors, the expectation
with respect to the other factors is considered. Derivation of the variational
distributions begins with the following expressions:
$$
\text{ln}p^\*(\mathbf{W}\_c) =
\mathbb{E}\_{\mathbf{W}\_b,
\mathbf{\lambda},
\mathbf{Z},
\mathbf{v},
\mathbf{U}
}\\{\text{ln}p(\mathbf{Y}\_c,
\mathbf{Y}\_b,
\mathbf{W}_c,
\mathbf{W}\_b,
\boldsymbol{\lambda},
\mathbf{Z},
\mathbf{v},
\mathbf{U}
) \\} +
\text{const}
$$

$$
\text{ln}p^\*(\mathbf{U}) =
\mathbb{E}\_{
\mathbf{W}\_c,
\mathbf{W}\_b,
\mathbf{\lambda},
\mathbf{Z},
\mathbf{v}
}\\{\text{ln}p(\mathbf{Y}\_c,
\mathbf{Y}\_b,
\mathbf{W}_c,
\mathbf{W}\_b,
\boldsymbol{\lambda},
\mathbf{Z},
\mathbf{v},
\mathbf{U}
) \\} +
\text{const}
$$

$$
\text{ln}p^\*(\boldsymbol{\lambda}) =
\mathbb{E}\_{
\mathbf{W}_c,
\mathbf{W}\_b,
\mathbf{Z},
\mathbf{v},
\mathbf{U}
}\\{ \text{ln}p(\mathbf{Y}\_c,
\mathbf{Y}\_b,
\mathbf{W}\_c,
\mathbf{W}\_b,
\boldsymbol{\lambda},
\mathbf{Z},
\mathbf{v},
\mathbf{U}
)\\} +
\text{const}
$$

$$
\text{ln}p^\*(\mathbf{v}) =
\mathbb{E}\_{
\mathbf{W}\_c,
\mathbf{W}\_b,
\boldsymbol{\lambda},
\mathbf{Z},
\mathbf{U}
}
\\{\text{ln}p(\mathbf{Y}\_c,
\mathbf{Y}\_b,
\mathbf{W}\_c,
\mathbf{W}\_b,
\mathbf{\lambda},
\mathbf{Z},
\mathbf{v},
\mathbf{U}
) \\} + \text{const}
$$

$$
\text{ln}p^\*(\mathbf{Z}) =
\mathbb{E}\_{
\mathbf{W}\_c,
\mathbf{W}\_b,
\boldsymbol{\lambda},
\mathbf{v},
\mathbf{U}
}
\\{\text{ln}p(\mathbf{Y}\_c,
\mathbf{Y}\_b,
\mathbf{W}\_c,
\mathbf{W}\_b,
\boldsymbol{\lambda},
\mathbf{Z},
\mathbf{v},
\mathbf{U}
)\\}
+ \text{const}
$$



A challenge arises, however, if priors in the model are not conditionally
conjugate – i.e. if factor posteriors are not in the same family as the
corresponding priors. This is the case with the Gaussian priors for the
coefficients of logistic regression \\(\mathbf{W}\_b\\), meaning that the
distribution \\(p^\*(\mathbf{W}\_b) \\) can not be assumed Gaussian unless
approximations are made to restore conjugacy. We address this challenge
by integrating the coordinate ascent variational inference algorithm with
the EM (Expectation-Maximization) algorithm to facilitate updates of
\\( p^\*(\mathbf{W}\_b) \\) (see below).


## Variational Distributions
Here we provide expressions for each of the variational distributions.

### Variational distribution \\( p^\*(\mathbf{W}_c) \\)
The variational distribution over the coefficients \\( \mathbf{W}_c \\) is given
by a multivariate Gaussian distribution that factorizes over the predictors,
targets, and subtype clusters:

$$
p^\*(\mathbf{W}_c)=
\prod\_{m=1}^{M}
\prod\_{d_c=1}^{D_c}
\prod\_{k=1}^{K}
\mathcal{N}
\left(
w\_{m, d_c, k} \mid
\mu\_{m, d_c, k},
{\lambda^{-1}\_{m, d_c, k}}
\right)
$$

$$
\lambda\_{m, d_c, k}=
\lambda\_{0\_{c\_{m, d_c}}}+
\frac{a\_{d_c, k}}{b\_{d_c, k}}
\sum\_{g=1}^{G}
\mathbb{E}\_{\mathbf{z}}\\{z\_{g, k}\\}
\sum\_{i=1}^{n_g}
x\_{i,m}^2
$$

$$
\mu_{m, d_c, k}=
{\lambda_{m, d_c, k}}^{-1}
\bigg[
\mu_{0_{c\_{m, d_c}}} {\lambda\_{0_{c_{m, d_c}}}} -
\frac{{a\_{d_c, k}}}{{b\_{d_c, k}}}
\sum\_{g=1}^{G}
\mathbb{E}\_{\mathbf{z}}\\{z\_{g, k}\\} \times
$$
$$
\sum\_{i=1}^{n_g}x\_{g,i,m}
\left(\mathbb{E}\_{\mathbf{w}}\\{\mathbf{w}\_{-, d_c, k}\\}^{T} \
\mathbf{x}\_{g,i,-} +
\mathbb{E}\\{ \mathbf{u}\_{g,d_c,k} \\}^T \mathbf{x}\_{g,i,.} -
y\_{g,i, d_c}\right)
\bigg] 
\label{Wc_ast}
$$

The \\(-\\) in \\( \mathbf{w}\_{-,d_c,k} \\) and \\( \mathbf{x}\_{g,i,-} \\)
indicates all but the \\(m^{th}\\) predictor.

### Variational distribution \\( p^{\*}(\mathbf{U}) \\)

The variational distribution for \\( p^{\*}(\mathbf{U}) \\) is given by

$$
p^{\*}(\mathbf{U}) =
\prod\_{g=1}^{G}
\prod\_{d_c=1}^{D_c}
\prod\_{k=1}^{K}
\mathcal{N}
\left(
\mathbf{u}\_{g, d_c, k} \mid
\boldsymbol{\mu}\_{g, d_c, k},
\boldsymbol{\Sigma}\_{g, d_c, k}
\right)
$$
where
$$
\boldsymbol{\Sigma}\_{g, d_c, k} =
\left[ \boldsymbol{\Sigma}_0^{-1} + 
\mathbb{E}\\{ z\_{g,k}\\}
\frac{a\_{d_c,k}}{b\_{d_c,k}}
\sum\_{i=1}^{n_g}
\mathbf{x}\_{g,i,d_c}\mathbf{x}\_{g,i,d_c}^T
\right]^{-1}
$$
and
$$
\boldsymbol{\mu}\_{g, d_c, k} =
\left[
\mathbb{E}\\{ z\_{g,k}\\}
\frac{a\_{d_c,k}}{b\_{d_c,k}}
\sum\_{i=1}^{n_g}
\left(
\mathbb{E}\\{  w\_{., d, k}  \\}^T
\mathbf{x}\_{g,i,d_c} - y\_{g, i, d_c}
\right)\mathbf{x}\_{g,i,d_c}^T
\right]
\boldsymbol{\Sigma}\_{g, d_c, k}
$$

### Variational distribution \\( p^{\*}(\boldsymbol{\lambda}) \\)

The variational distribution over \\( \mathbf{\lambda} \\) is given by a gamma
distribution with parameters \\( \mathbf{a} \\) and \\( \mathbf{b} \\):
$$
p^{\*}(\boldsymbol{\lambda})=
\prod\_{d_c=1}^{D_c}
\prod\_{k=1}^{K}
\operatorname{Gam}
\left(\lambda\_{d_c, k} \mid a\_{d_c, k}, b\_{d_c, k}
\right)
$$

$$
a_\{d_c, k}=
a\_{0\_{d_c}}+
\frac{1}{2}
\sum\_{g=1}^{G}n_g
\mathbb{E}\_{\mathbf{z}}\\{z\_{g, k}\\} 
$$

$$
b\_{d_c, k}=
b\_{0_{d_c}}+
\frac{1}{2}
\sum\_{g=1}^{G}
\sum\_{i=1}^{n_g}
\mathbb{E}\\{z\_{g, k}\\}
\bigg(\mathbb{E}\\{
\left({\mathbf{w}^{T}\_{\cdot, d_c, k}} \mathbf{x}\_{g,i, \cdot}
\right)^{2}\\}
-2 y\_{g,i,d_c}
\mathbb{E}\\{\mathbf{w}\_{\cdot,d_c, k}\\}^{T} \mathbf{x}_{g,i,\cdot} +
{y^{2}\_{g,i, d_c}}
$$
$$
-2y\_{g,i,d_c}\mathbb{E}\\{ \mathbf{u}\_{g,d_c,k}\\}^T\mathbf{x}\_{g,i,.} +
2\mathbb{E}\\{ \mathbf{w}\_{.,d_c,k}\\}^T\mathbf{x}\_{g,i,.}
\mathbb{E}\\{ \mathbf{u}\_{g,d_c,k}\\}^T\mathbf{x}\_{g,i,.} +
\mathbb{E}\\{ ( \mathbf{u}\_{g,d_c,k}^T\mathbf{x}\_{g,i,.} )^2 \\}
\bigg)
\label{lambda_ast}
$$

### Variational distribution \\( p^{\*}(\mathbf{v}) \\)

The variational distribution for \\( p^{\*}(\mathbf{v}) \\) is given by
$$
p^{\*}(\mathbf{v})=
\prod\_{k=1}^{K}
\text{Beta}
\left( 
v_k \mid 1 +
\sum\_{g=1}^{G}
\mathbb{E}\\{\mathbf{Z} \\}\_{g, k}, 
\alpha +
\sum\_{j=k+1}^{K}
\sum\_{g=1}^{G}
\mathbb{E}\\{\mathbf{Z} \\}\_{g, j}
\right)
\label{v_ast}
$$
where \\( K \\) is an integer (e.g. 20) chosen by the user for the truncated stick-breaking process.


### Variational distribution \\( p^{\*}(\mathbf{Z}) \\)

The variational distribution for \\( p^{\*}(\mathbf{Z}) \\) is given by

$$
p^{*}(\mathbf{Z})=
\prod_{g=1}^{G}\prod_{k=1}^{K}r_{g,k}^{z_{g,k}}
\label{pZstar}
$$

where

$$
r\_{g,k}=\frac{\rho\_{g,k}}{\sum\_{k=1}^{K}\rho\_{g,k}} \label{eq:rnk}
$$

and

$$
\ln \rho\_{g, k} = n_g\mathbb{E}\\{\ln v_k\\} +
n_g\sum\_{j=1}^{k-1}\mathbb{E}\\{\ln (1-v_j)\\} + \\
$$

$$
\frac{1}{2}\sum_{d_c=1}^{D_c}
\bigg[n_g\mathbb{E}\\{\ln \lambda\_{d_c, k}\\}-n_g\ln (2 \pi) -
$$
$$
\mathbb{E}\\{\lambda\_{d_c, k}\\}
\sum_{i=1}^{n_g}
\bigg(\mathbb{E}\\{\left({\mathbf{w}^{T}\_{\cdot, d_c, k}} \mathbf{x}\_{g,i,\cdot}\right)^2\\} -
2 y\_{g,i,d_c} \mathbb{E}\\{{\mathbf{w}\_{\cdot, d_c, k}}\\}^{T} \mathbf{x}\_{g,i,\cdot}+y^2\_{g,i, d_c} +
$$

$$
-2y\_{g,i,d_c}\mathbb{E}\\{ \mathbf{u}\_{g,d_c,k}\\}^T\mathbf{x}\_{g,i,.} +
2\mathbb{E}\\{ \mathbf{w}\_{.,d_c,k}\\}^T\mathbf{x}\_{g,i,.}
\mathbb{E}\\{ \mathbf{u}\_{g,d_c,k}\\}^T\mathbf{x}\_{g,i,.} +
\mathbb{E}\\{ ( \mathbf{u}\_{g,d_c,k}^T\mathbf{x}\_{g,i,.} )^2 \\}
\bigg)\bigg] +
$$
$$
\sum\_{d_b=1}^{D_b}
\sum\_{i=1}^{n_g}
\left[
y\_{g,i, d_b} \mathbb{E}\\{
    {\mathbf{w}\_{\cdot, d_b, k}}\\}^{T}{\mathbf{x}\_{g,i, \cdot}}
    -\mathbb{E}\\{\ln\left(1+\exp\left(\mathbf{x}\_{g,i, \cdot}\cdot{\mathbf{w}\_{\cdot, d_b, k}}^T\right)\right)\\}
    \right]
$$


## Update Strategy for \\( \mathbf{W}\_b \\) 
As mentioned above, the Gaussian priors over the coefficients
\\( \mathbf{W}\_b \\) are not conjugate concerning the likelihood factor
\\( p(\mathbf{Y}_b\mid \mathbf{Z}, \mathbf{W}_b) \\).
When the prior and likelihood are not conjugate, Bayesian inference becomes more
complex and computationally demanding since the posterior distribution cannot be
derived analytically. 
Our methodology applies a tangent quadratic lower bound to the logistic
likelihoods within the framework of variational inference for conditionally
conjugate exponential family models 
(see [Durante (2019)](https://www.jstor.org/stable/pdf/26874191.pdf?acceptTC=true&coverpage=false&addFooter=false&casa_token=5tSjE7iYI8wAAAAA:9UYtJvE_cGh2930jPcoCnSuWlK7-scKaCRwS1LwsRrF2_Uwq5qGsA-PcU4P_QAlJWwI8-M86kfGTfdCwFjYGZ39HW1SGCbGA_RYGk9iDog3yZgYJxQ)).
This approach restores conjugacy between the approximate bounds and the
Gaussian priors on \\( \mathbf{W}_b \\).

[Jaakkola and Jordan (2000)](http://www2.stat.duke.edu/homeweb/scs/Courses/Stat376/Papers/Variational/JaakkolaJordan2000.pdf)
introduced a straightforward variational approach based
on a family of tangent quadratic lower bounds of logistic log-likelihoods. They
derived an EM algorithm to iteratively refine the variational parameters of the
lower bound and the mean and covariance of the Gaussian distribution over the
predictor coefficients. However, this method was specifically designed for
simple logistic regression and did not extend to mixtures of logistic
regressors. To address this, we extend these concepts to Dirichlet Process
mixture models in our formulation. We can augment the likelihood function
\\( p(\mathbf{Y}_b\mid \mathbf{Z}, \mathbf{W}_b) \\):

$$
p(\mathbf{Y}\_b, \boldsymbol{\zeta} \mid \mathbf{Z}, \mathbf{W}\_b)=
\prod_\{k=1}^{\infty}\prod\_{g=1}^{G} \prod\_{d_b=1}^{D_b} 
\left[\prod\_{i=1}^{n\_g}
p\left( y\_{g,i,d_b}\vert \mathbf{w}\_{\cdot,d_b,k} \right)
p\left(\zeta\_{g,i,d_b,k}\vert \mathbf{w}\_{\cdot,d_b,k} \right)
\right]^{z_{g,k}},
$$

where \\(\boldsymbol{\zeta}\_{g,i,d_b,k}\\) are Polya-gamma densities
\\(\text{PG}(1,\mathbf{w}^{T}\_{\cdot,d_b,k}\mathbf{x}\_{g,i,\cdot}) \\)
as described in
[Durante (2019)](https://www.jstor.org/stable/pdf/26874191.pdf?acceptTC=true&coverpage=false&addFooter=false&casa_token=5tSjE7iYI8wAAAAA:9UYtJvE_cGh2930jPcoCnSuWlK7-scKaCRwS1LwsRrF2_Uwq5qGsA-PcU4P_QAlJWwI8-M86kfGTfdCwFjYGZ39HW1SGCbGA_RYGk9iDog3yZgYJxQ),
except in our case we consider a nonparametric mixture of \\(D\_b\\)
conditionally independent target variables. Importantly, the augmented
likelihood is within the exponential family of distributions, and the prior
over \\(\mathbf{W}\_b\\) is now conjugate. 

[Durante and Rigon (2019)](https://www.jstor.org/stable/pdf/26874191.pdf?acceptTC=true&coverpage=false&addFooter=false&casa_token=5tSjE7iYI8wAAAAA:9UYtJvE_cGh2930jPcoCnSuWlK7-scKaCRwS1LwsRrF2_Uwq5qGsA-PcU4P_QAlJWwI8-M86kfGTfdCwFjYGZ39HW1SGCbGA_RYGk9iDog3yZgYJxQ)
provide coordinate ascent variational inference
updates for the variational distributions \\(p^\*\left(\mathbf{W}\_b\right) \\)
and \\(p^\*\left(\boldsymbol{\zeta} \right)\\) (which in turn relate directly
the the EM algorithm proposed by
[Jaakkola and Jordan (2000)](http://www2.stat.duke.edu/homeweb/scs/Courses/Stat376/Papers/Variational/JaakkolaJordan2000.pdf)).
Extending these updates to our model gives the variational distribution over
\\(\mathbf{W}\_b\\) as:
$$
    p^\*(\mathbf{W}\_b) = \prod\_{d_b=1}^{D_b}\prod_{k=1}^{K}
    N(\mathbf{w}\_{\cdot,d_b,k} \vert 
    \boldsymbol{\mu}\_{d_b,k},\boldsymbol{\lambda}^{-1}\_{d_b,k})
$$
where
$$
\boldsymbol{\lambda}^{-1}\_{d_b,k} = \left(\boldsymbol{\Sigma}^{-1}\_{d_b} +
\mathbf{X}^{T}\mathbf{G}\_{k}\mathbf{X} \right)^{-1}
$$
and

$$
    \mathbf{G}\_{k} = \text{diag}\\{  
    0.5\left[\xi\_{1,d_b,k} \right]^{-1}\text{tanh}\left(0.5 \xi\_{1,d_b,k}\right)r\_{1,k}, \cdots,
    0.5\left[\xi\_{N,d_b,k} \right]^{-1}\text{tanh}\left(0.5 \xi\_{N,d_b,k}\right)r\_{N,k}
    \\}
$$

and

$$
    \boldsymbol{\mu}\_{d_b,k}=\boldsymbol{\lambda}^{-1}\_{d_b,k}\left[ 
    \mathbf{X}^{T}\text{diag}\\{r\_{\cdot,k}\\}
    \left( \mathbf{y}\_{\cdot,d_b}-0.5\mathbf{1}\_{N} \right) +
    \mathbf{\Sigma}\_{d_b}\boldsymbol{\mu}\_{d_b}
    \right]
$$

Note that in order to more easily express the variational distribution
parameters, we have introduced the index \\(n\\) to refer to individual data
points: \\((g=1, i=1) \mapsto n=1, (g=1, i=2) \mapsto n=2, \cdots,
(g=G, i=n_g) \mapsto n=N\\). The vector \\( \boldsymbol{\mu}\_{d_b} \\) is
the \\(M\\)-dimensional vector of mean prior values for the \\(d\_{b}^{th}\\)
dimension for the prior over \\( \mathbf{W}_b \\). Similarly,
\\(\mathbf{\Sigma}\_{d_b}\\) is the \\(M\times M\\) diagonal matrix of variance
(inverse precision) values for
the \\(d\_{b}^{th}\\) dimension. \\(r\_{n,k}\\) is
the probability that data instance \\(n\\) belongs to component \\(k\\)
(see above).

The variational distribution \\(p^\*\left( \zeta\_{n,d_b,k} \right)\\) is a
Polya-gamma distribution, \\(\text{PG}(1,\xi\_{n,d_b,k})\\) with
$$
    \xi\_{n,d_b,k} = \left[ 
    \mathbf{x}^{T}\_{n,\cdot}\boldsymbol{\lambda}^{-1}\_{d_b,k}
    \mathbf{x}\_{n,\cdot} +
    \left(\mathbf{x}^{T}\_{n,\cdot}\boldsymbol{\mu}_{d_b,k}\right)^{2}
    \right]^{\frac{1}{2}}.
$$

## Optimization
With these terms define, inference proceeds by iteratively updating the parameters for
the variational distributions
\\( p^\*(\mathbf{W}_c) \\),
\\( p^\*(\mathbf{W}_b) \\),
\\( p^{\*}(\mathbf{U}) \\),
\\( p^{\*}(\boldsymbol{\lambda}) \\),
\\( p^{\*}(\mathbf{v}) \\), and
\\( p^{\*}(\mathbf{Z}) \\) for a prespecified number of
iterations, set to ensure successive iterations produce a negligible change in the
variational parameters.

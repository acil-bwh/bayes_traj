from argparse import ArgumentParser
import matplotlib.pyplot as plt
import numpy as np
from numpy import abs, dot, mean, log, sum, exp, tile, max, sum, isnan, diag, \
     sqrt, pi, newaxis, outer, genfromtxt, where
from numpy.random import multivariate_normal, randn, gamma, binomial
from bayes_traj.utils import sample_cos
from scipy.optimize import minimize_scalar
from scipy.special import psi, gammaln, logsumexp
from scipy.stats import norm
import pandas as pd
import pdb, sys, pickle, time, warnings

class MultDPRegression:
    """Uses Dirichlet process mixture modeling to identify mixtures of
    regressors. In the below, 'D' signifies the dimension of the target
    variable, and we assume that each target is associated with a vector of
    predictors with dimension 'M'. 'N' signifies the number of data
    samples, and 'K' is an integer indicating the number of elements in the
    truncated Dirichlet process. We use 'DP' for 'Dirichlet Process'.

    Parameters
    ----------
    w_mu0 : array, shape ( M, D )
        The coefficient for each predictor is drawn from a normal distribution.
        This is the matrix of hyperparameters is the mean values of those
        normal distributions.

    w_var0 : array, shape ( M, D )
        The coefficient for each predictor is drawn from a normal distribution.
        This is the matrix of hyperparameters of the precision values of those
        normal distributions.

    lambda_a0 : array, shape ( D )
        For each target dimension 'd' (of 'D'), this is the first parameter of
        the Gamma prior over the precision for that target dimension.

    lambda_b0 : array, shape ( D )
        For each target dimension 'd' (of 'D'), this is the second parameter of
        the Gamma prior over the precision for that target dimension.

    prec_prior_weight : float
        Value > 0 by which to scale the Gamma priors of the residual 
        precisions. The higher the value, the more weight that is given to the
        prior.

    alpha : float
        Hyper parameter of the Beta destribution involved in the stick-breaking
        construction of the Dirichlet Process. Increasing 'alpha' tends to
        produce more clusters and vice-versa.

    K : int
        An integer indicating the number of elements in the truncated Dirichlet
        Process.

    prob_thresh : float, optional
        Once the probability of belonging to a component drops below this
        threshold, the corresponding latent variables will cease to be
        updated (saving computation time). Higher values indicate that
        components will more readily be discarded as possible components.
        While this will speed computation, setting values too high runs
        the risk of discarding actual components.

    Attributes
    ----------
    v_a_ : array, shape ( K )
        For each of the 'K' elements in the truncated DP, this is the first
        parameter of posterior Beta distribution describing the latent vector,
        'v', which is involved in the stick-breaking construction of the DP.

    v_b_ : array, shape ( K )
        For each of the 'K' elements in the truncated DP, this is the second
        parameter of posterior Beta distribution describing the latent vector,
        'v', which is involved in the stick-breaking construction of the DP.

    R_ : array, shape ( N, K )
        Each element of this matrix represents the posterior probability that
        instance 'n' belongs to cluster 'k'.

    w_mu_ : array, shape ( M, D, K )
        The posterior means of the Normal distributions describing each of the
        predictor coefficients, for each dimension of the proportion vector,
        for each of the 'K' components.

    w_var_ : array, shape ( M, D, K )
        The posterior variances of the Normal distributions describing each of
        the predictor coefficients, for each dimension of the proportion vector,
        for each of the 'K' components.

    lambda_a : array, shape ( D, K ), optional
        For component 'K' and target dimension 'D', this is the first parameter
        of the posterior Gamma distribution describing the precision of the
        target variable. Only relevant for continuous (Gaussian) target 
        variables.

    lambda_b : array, shape ( D, K ), optional
        For component 'K' and target dimension 'D', this is the second parameter
        of the posterior Gamma distribution describing the precision of the
        target variable. Only relevant for continuous (Gaussian) target 
        variables.
    """
    def __init__(self, w_mu0, w_var0, lambda_a0, lambda_b0, prec_prior_weight,
                 alpha, K=10, prob_thresh=0.001):
        self.w_mu0_ = w_mu0
        self.w_var0_ = w_var0
        self.lambda_a0_ = lambda_a0
        self.lambda_b0_ = lambda_b0
        self.prec_prior_weight_ = prec_prior_weight
        self.K_ = K
        self.M_ = self.w_mu0_.shape[0]
        self.D_ = self.w_mu0_.shape[1]
        self.alpha_ = alpha
        self.prob_thresh_ = prob_thresh

        # Keeps track of what data type each dimension is
        self.target_type_ = {}
        
        # For recording the lower-bound terms
        self.lower_bounds_ = []

        # X_ and Y_ will become defined when 'fit' is called
        self.X_ = None
        self.Y_ = None
        self.target_names_ = None
        self.predictor_names_ = None

        self.sig_trajs_ = np.ones(self.K_, dtype=bool)

        assert self.w_mu0_.shape[0] == self.w_var0_.shape[0] and \
          self.w_mu0_.shape[1] == self.w_var0_.shape[1], \
          "Shape mismatch for w_mu0_ and w_var0_"

        assert self.lambda_a0_.shape[0] == self.lambda_b0_.shape[0], \
          "Shape mismatch for lambda_a0_ and lambda_b0_"

        assert self.w_mu0_.shape[1] == self.lambda_a0_.shape[0], \
          "Target dimension mismatch"

    def fit(self, target_names, predictor_names, df, groupby=None, iters=100,
            R=None, traj_probs=None, traj_probs_weight=None, v_a=None,
            v_b=None, w_mu=None, w_var=None, lambda_a=None, lambda_b=None,
            verbose=False):
        """Performs variational inference (coordinate ascent or SVI) given data
        and provided parameters.

        Parameters
        ----------
        target_names : list of strings
            Data frame column names of the target variables.

        predictor_names : list of strings
            Data frame column names of the predictors

        df : pandas dataframe
            Data frame containing predictor, target, and group information.

        groupby : str, optional
            Data frame column name used to group data instances. All data 
            instances within a group will be forced into the same trajectory.
            This is generally desired when multiple data instances correspond
            to the same individual. 

        iters : int, optional
            Number of variational inference iterations to run. 

        R : array, shape ( N, K ), optional
            Each element of this matrix represents the posterior probability
            that instance 'n' belongs to cluster 'k'. If specified, the
            algorithm will be initialized with this matrix, otherwise a default
            matrix (randomly generated) will be used. If a traj_probs_weightd 
            value is also specified, this matrix will be combined with a 
            randomly generated matrix in a weighted fashion.

        traj_probs : array, shape ( K ), optional
            A priori probabilitiey of each of the K trajectories. Each element 
            must be >=0 and <= 1, and all elements must sum to one.

        traj_probs_weight : float, optional
            Value between 0 and 1 inclusive that controls how R is combined with
            a randomly generated matrix: traj_probs_weightd*R + 
            (1-traj_probs_weightd)*R_random. If traj_probs_weightd is not 
            specified, it will be assumed equal to 1. If R is not specified, 
            traj_probs_weightd has no effect.

        v_a : array, shape ( K, 1 ), optional
            For each of the 'K' elements in the truncated DP, this is the first
            parameter of posterior Beta distribution describing the latent
            vector, 'v', which is involved in the stick-breaking construction
            of the DP. If specified, the algorithm will be initialized with this
            vector.

        v_b : array, shape ( K, 1 ), optional
            For each of the 'K' elements in the truncated DP, this is the second
            parameter of posterior Beta distribution describing the latent
            vector, 'v', which is involved in the stick-breaking construction
            of the DP. If specified, the algorithm will be initialized with this
            vector.

        w_mu : array, shape ( M, D, K ), optional
            The posterior means of the Normal distributions describing each of
            the predictor coefficients, for each dimension of the proportion
            vector, for each of the 'K' components. If specified, the algorithm
            will be initialized with this matrix.

        w_var : array, shape ( M, D, K ), optional
            The posterior variances of the Normal distributions describing each
            of the predictor coefficients, for each dimension of the proportion
            vector, for each of the 'K' components. If specified, the algorithm
            will be initialized with this matrix.

        lambda_a : array, shape ( D, K ), optional
            For component 'K' and target dimension 'D', this is the first
            parameter of the posterior Gamma distribution describing the
            precision of the target variable. If specified, the algorithm will
            be initialized with this matrix. Only relevant for continuous 
           (Gaussian) target variables.

        lambda_b : array, shape ( D, K ), optional
            For component 'K' and target dimension 'D', this is the second
            parameter of the posterior Gamma distribution describing the
            precision of the target variable. If specified, the algorithm will
            be initialized with this matrix. Only relevant for continuous 
           (Gaussian) target variables.

        verbose : bool, optional
            If true, a printout of the sum along rows of the R_ matrix will
            be provided during optimization. This sum indicates how many data
            instances are being assigned to each of the K possible
            trajectories.
        """
        if traj_probs_weight is not None:
            assert traj_probs_weight >= 0 and traj_probs_weight <=1, \
                "Invalid traj_probs_weightd value"
        
        self.X_ = df[list(predictor_names)].values
        self.Y_ = df[list(target_names)].values        

        self.gb_ = None
        self.df_ = df
        if groupby is not None:
            self.gb_ = df[[groupby]].groupby(groupby)
            
        assert len(set(target_names)) == len(target_names), \
            "Duplicate target name found"
        self.target_names_ = target_names

        assert len(set(predictor_names)) == len(predictor_names), \
            "Duplicate predictor name found"
        self.predictor_names_ = predictor_names

        assert self.w_mu0_.shape[0] == self.X_.shape[1], \
          "Dimension mismatch between mu_ and X_"
        assert self.X_.shape[0] == self.Y_.shape[0], \
          "X_ and Y_ do not have the same number of samples"        
        
        self.N_ = self.X_.shape[0]
        self.M_ = self.X_.shape[1]
        self.D_ = self.Y_.shape[1]
        self.lambda_a_ = lambda_a
        self.lambda_b_ = lambda_b
        self.w_mu_ = w_mu
        self.w_var_ = w_var
        self.v_a_ = v_a
        self.v_b_ = v_b
        self.R_ = R

        # w_covmat_ is used for binary target variables. The EM algorithm
        # that is used to estimate w_mu_ and w_var_ for binary targets
        # actually gives us a full covariance matrix which we take advantage
        # of when performing sampling based estimates elsewhere. Although
        # we only need/have w_covmat_ for binary targets, we allocate space
        # over all dimensions, D, to make implementation clearer (i.e. when
        # we iterate over D)
        self.w_covmat_ = np.nan*np.ones([self.M_, self.M_, self.D_, self.K_])
        
        self.num_binary_targets_ = 0
        for d in range(self.D_):
            if set(self.Y_[:, d]).issubset({1.0, 0.0}):
                self.target_type_[d] = 'binary'
                self.num_binary_targets_ += 1
            else:
                self.target_type_[d] = 'gaussian'
        print("Initializing paramters...")
        self.init_traj_params()

        # The prior over the residual precision can get overwhelmed by the
        # data -- so much so that residual precision posteriors can wind up
        # in regimes that have near-zero mass in the prior. Given this, we
        # scale the prior params (essentially lowering the variance of the
        # prior) by an amount proportional to the number of subjects in the
        # data set. Note that this step needs to be done AFTER
        # init_traj_params, which uses the original prior to randomly
        # initialize trajectory precisions.
        self.lambda_a0_ *= self.prec_prior_weight_*\
            (self.gb_.ngroups if self.gb_ is not None else self.N_)
        self.lambda_b0_ *= self.prec_prior_weight_*\
            (self.gb_.ngroups if self.gb_ is not None else self.N_)        

        if self.v_a_ is None:
            self.v_a_ = np.ones(self.K_)

        if self.v_b_ is None:
            self.v_b_ = self.alpha_*np.ones(self.K_)

        # Initialize the latent variables if needed
        if self.R_ is None:
            self.init_R_mat(traj_probs, traj_probs_weight)

        self.fit_coordinate_ascent(iters, verbose)

        
    def fit_coordinate_ascent(self, iters, verbose):
        """This function contains the iteratrion loop for mean-field 
        variational inference using coordinate ascent

        Parameters
        ----------
        iters : int, optional
            Number of variational inference iterations to run.

        verbose : bool, optional
            If true, a printout of the sum along rows of the R_ matrix will
            be provided during optimization. This sum indicates how many data
            instances are being assigned to each of the K possible
            trajectories.
        """
        inc = 0
        while inc < iters:
            inc += 1
            self.update_v()
            if self.num_binary_targets_ > 0:
                self.update_w_logistic(em_iters=1)
            if self.D_ - self.num_binary_targets_ > 0:
                self.update_w_gaussian()
                self.update_lambda()
            self.R_ = self.update_z(self.X_, self.Y_)
            self.sig_trajs_ = np.max(self.R_, 0) > self.prob_thresh_

            if verbose:
                print("iter {},  {}".format(inc, sum(self.R_, 0)))

                
    def update_v(self):
        """Updates the parameters of the Beta distributions for latent
        variable 'v' in the variational approximation.
        """
        self.v_a_ = 1.0 + np.sum(self.R_, 0)

        for k in np.arange(0, self.K_):
            self.v_b_[k] = self.alpha_ + np.sum(self.R_[:, k+1:])


    def update_z(self, X, Y):
        """
        """
        expec_ln_v = psi(self.v_a_) - psi(self.v_a_ + self.v_b_)
        expec_ln_1_minus_v = psi(self.v_b_) - psi(self.v_a_ + self.v_b_)

        tmp = np.array(expec_ln_v)
        for k in range(1, self.K_):
            tmp[k] += np.sum(expec_ln_1_minus_v[0:k])

        ln_rho = np.ones([self.N_, self.K_])*tmp[newaxis, :]

        if self.num_binary_targets_ > 0:
            num_samples = 20 # Arbitrary. Should be "big enough"
            mc_term = np.zeros([self.N_, self.K_])
        for d in range(0, self.D_):
            non_nan_ids = ~np.isnan(Y[:, d])
            if self.target_type_[d] == 'gaussian':                
                tmp = (dot(self.w_mu_[:, d, :].T, \
                    X[non_nan_ids, :].T)**2).T + \
                    np.sum((X[non_nan_ids, newaxis, :]**2)*\
                    (self.w_var_[:, d, :].T)[newaxis, :, :], 2)

                ln_rho[non_nan_ids, :] += \
                  0.5*(psi(self.lambda_a_[d, :]) - \
                    log(self.lambda_b_[d, :]) - \
                    log(2*np.pi) - \
                    (self.lambda_a_[d, :]/self.lambda_b_[d, :])*\
                    (tmp - \
                     2*Y[non_nan_ids, d, newaxis]*dot(X[non_nan_ids, :], \
                    self.w_mu_[:, d, :]) + \
                    Y[non_nan_ids, d, newaxis]**2))
            elif self.target_type_[d] == 'binary':
                for k in range(self.K_):
                    mc_term[:, k] = np.mean(np.log(1 + \
                        np.exp(np.dot(self.X_, \
                        np.random.multivariate_normal(self.w_mu_[:, d, k], \
                            self.w_covmat_[:, :, d, k], num_samples).T))), 1)

                ln_rho[non_nan_ids, :] += Y[non_nan_ids, d, newaxis]*\
                    dot(X[non_nan_ids, :], self.w_mu_[:, d, :]) - mc_term                

        # The values of 'ln_rho' will in general have large magnitude, causing
        # exponentiation to result in overflow. All we really care about is the
        # normalization of each row. We can use the identity exp(a) =
        # 10**(a*log10(e)) to put things in base ten, and then subtract from
        # each row the max value and also clipping the resulting row vector to
        # lie within -300, 300 to ensure that when we exponentiate we don't
        # have any overflow issues.
        rho_10 = ln_rho*np.log10(np.e)
        rho_10_shift = 10**((rho_10.T - max(rho_10, 1)).T + 300).clip(-300, 300)
        R = (rho_10_shift.T/sum(rho_10_shift, 1)).T

        # Within a group, all data instances must have the same probability of
        # belonging to each of the K trajectories
        if self.gb_ is not None:
            for g in self.gb_.groups.keys():
                tmp_ids = self.gb_.get_group(g).index.values
                tmp_vec = np.sum(np.log(R[tmp_ids, :] + \
                                        np.finfo(float).tiny), 0)
                vec = np.exp(tmp_vec + 700 - np.max(tmp_vec))
                vec = vec/np.sum(vec)

                # Now update the rows of 'normalized_mat'.
                R[tmp_ids, :] = vec                

        # Any instance that has miniscule probability of belonging to a
        # trajectory, set it's probability of belonging to that trajectory to 0
        R[R <= self.prob_thresh_] = 0

        return R
    
    def update_w_logistic(self, em_iters=1):
        """Uses an EM algorithm based on the approach in the reference to update
        the coefficient distributions corresponding to binary targets.

        Parameters
        ----------
        em_iters : int, optional
            The number of EM iterations to perform

        References
        ----------
        Durante D, Rigon T. Conditionally conjugate mean-field variational 
        Bayes for logistic models. Statistical science. 2019;34(3):472-85.
        """        
        d_bin = -1
        for d in range(self.D_):
            if self.target_type_[d] == 'binary':
                d_bin += 1
            
                for k in np.where(self.sig_trajs_)[0]:
                    non_nan_ids = ~np.isnan(self.Y_[:, d]) #& \
                       # (self.R_[:, k] > 0)

                    for i in range(em_iters):
                        # E-step
                        #print(self.w_mu_[:, d, k])
                        Z_bar = np.diag(0.5*self.R_[non_nan_ids, k]*\
                                        (1/self.xi_[non_nan_ids, d_bin, k])*\
                            np.tanh(0.5*self.xi_[non_nan_ids, d_bin, k]))
                        
                        sig_mat_0 = np.diag(self.w_var0_[:, d])
                        mu_0 = self.w_mu0_[:, d]

                        self.w_covmat_[:, :, d, k] = \
                            np.linalg.inv(np.linalg.inv(sig_mat_0) + \
                            np.dot(self.X_[non_nan_ids, :].T, \
                                   np.dot(Z_bar, self.X_[non_nan_ids, :])))
                        self.w_var_[:, d, k] = \
                            np.diag(self.w_covmat_[:, :, d, k])
                        #self.w_mu_[:, d, k] = \
                        #    np.dot(self.w_covmat_[:, :, d, k], \
                        #           np.dot(self.X_[non_nan_ids, :].T,
                        #        (self.Y_[non_nan_ids, d] - 0.5)) + \
                        #            np.dot(np.linalg.inv(sig_mat_0), mu_0))

                        # DEB:
                        self.w_mu_[:, d, k] = \
                            np.dot(self.w_covmat_[:, :, d, k], \
                                   np.dot(self.X_[non_nan_ids, :].T,
                                self.R_[non_nan_ids, k]*(self.Y_[non_nan_ids, d] - 0.5)) + \
                                    np.dot(np.linalg.inv(sig_mat_0), mu_0))
                        
                        
                        # M-step
                        self.xi_[non_nan_ids, d_bin, k] = \
                            np.sqrt(np.diag(np.dot(self.X_[non_nan_ids, :],
                                np.dot(self.w_covmat_[:, :, d, k], \
                                       self.X_[non_nan_ids, :].T))) + \
                                np.dot(self.X_[non_nan_ids, :], \
                                       self.w_mu_[:, d, k])**2)
                        
    def update_w_gaussian(self):
        """ Updates the variational distributions over predictor coefficients 
        corresponding to continuous (Gaussian) target variables. 
        """
        mu0_DIV_var0 = self.w_mu0_/self.w_var0_
        for m in range(0, self.M_):
            ids = np.ones(self.M_, dtype=bool)
            ids[m] = False
            for d in range(0, self.D_):
                if self.target_type_[d] == 'gaussian':
                    non_nan_ids = ~np.isnan(self.Y_[:, d])
    
                    tmp1 = (self.lambda_a_[d, self.sig_trajs_]/\
                            self.lambda_b_[d, self.sig_trajs_])*\
                            (np.sum(self.R_[:, self.sig_trajs_, newaxis]\
                                    [non_nan_ids, :, :]*\
                                    self.X_[non_nan_ids, newaxis, :]**2, 0).T)\
                                    [:, newaxis, :]
                
                    self.w_var_[:, :, self.sig_trajs_] = \
                        (tmp1 + (1.0/self.w_var0_)[:, :, newaxis])**-1
                
                    sum_term = sum(self.R_[non_nan_ids, :][:, self.sig_trajs_]*\
                                   self.X_[non_nan_ids, m, newaxis]*\
                                   (dot(self.X_[:, ids][non_nan_ids, :], \
                                        self.w_mu_[ids, d, :]\
                                        [:, self.sig_trajs_]) - \
                                    self.Y_[non_nan_ids, d][:, newaxis]), 0)
    
                    self.w_mu_[m, d, self.sig_trajs_] = \
                        self.w_var_[m, d, self.sig_trajs_]*\
                        (-(self.lambda_a_[d, self.sig_trajs_]/\
                           self.lambda_b_[d, self.sig_trajs_])*\
                         sum_term + mu0_DIV_var0[m, d])

                
    def update_lambda(self):
        """Updates the variational distribution over latent variable lambda.
        """    
        for d in range(self.D_):
            non_nan_ids = ~np.isnan(self.Y_[:, d])
    
            self.lambda_a_[d, self.sig_trajs_] = self.lambda_a0_[d, newaxis] + \
                0.5*sum(self.R_[:, self.sig_trajs_][non_nan_ids, :], 0)\
                [newaxis, :]
            
            tmp = (dot(self.w_mu_[:, d, self.sig_trajs_].T, \
                       self.X_[non_nan_ids, :].T)**2).T + \
                       np.sum((self.X_[non_nan_ids, newaxis, :]**2)*\
                              (self.w_var_[:, d, self.sig_trajs_].T)\
                              [newaxis, :, :], 2)
    
            self.lambda_b_[d, self.sig_trajs_] = \
                self.lambda_b0_[d, newaxis] + \
                0.5*sum(self.R_[non_nan_ids, :][:, self.sig_trajs_]*\
                        (tmp - 2*self.Y_[non_nan_ids, d, newaxis]*\
                         np.dot(self.X_[non_nan_ids, :], \
                                self.w_mu_[:, d, self.sig_trajs_]) + \
                         self.Y_[non_nan_ids, d, newaxis]**2), 0)

            
    def sample(self, index=None, x=None):
        """sample from the posterior distribution using the input data.

        Parameters
        ----------
        index : int, optional
            The data sample index at which to sample. If none specified, samples
            will be drawn for all of the 'N' sample points.

        x : array, shape ( M ), optional
            Only relevant if 'index' is also specified. If both 'index' and 'x'
            are specified, a sample will be drawn for the data point specified
            by 'index', using the predictor values specified by 'x'.  If 'x' is
            not specified, the original predictor values for the 'index' data
            point will be used to draw the sample. Here, 'M' is the dimension of
            the predictors.

        Returns
        -------
        y_rep : array, shape ( N, M ) or ( M )
            The sample(s) randomly drawn from the posterior distribution
        """
        if index is None:
            indices = range(0, self.N_)
            y_rep = np.zeros([self.N_, self.D_])
            X = self.X_
        else:
            if not (type(index) == int or type(index) == np.int64):
                raise ValueError('index must be an integer if specified')

            indices = range(index, index+1)
            y_rep = np.zeros([1, self.D_])

            if x is not None:
                if len(x.shape) != 1:
                    raise ValueError('x must be a vector')

                if x.shape[0] != self.M_:
                    raise ValueError('x has incorrect dimension')

        for n in indices:
            z = np.random.multinomial(1, self.R_[n, :]/np.sum(self.R_[n, :]))
            for d in range(0, self.D_):
                if self.target_type_[d] == 'gaussian':

                    # Draw a precision value from the gamma distribution. Note
                    # that numpy uses a slightly different parameterization
                    scale = 1./self.lambda_b_[d, z.astype(bool)]
                    shape = self.lambda_a_[d, z.astype(bool)]
                    var = 1./gamma(shape, scale, size=1)
                    
                    co = multivariate_normal(\
                        self.w_mu_[:, d, z.astype(bool)][:, 0],
                        diag(self.w_var_[:, d, z.astype(bool)][:, 0]), 1)

                    if x is not None:
                        mu = dot(co, x)
                    else:
                        mu = dot(co, self.X_[n, :])

                    if index is None:
                        y_rep[n, d] = sqrt(var)*randn(1) + mu
                    else:
                        y_rep[0, d] = sqrt(var)*randn(1) + mu                    
                else:
                    # Target assumed to be binary
                    co = multivariate_normal(\
                        self.w_mu_[:, d, z.astype(bool)][:, 0],
                        self.w_covmat_[:, :, d, z.astype(bool)][:, 0], 1)

                    if x is not None:
                        mu = dot(co, x)
                    else:
                        mu = dot(co, self.X_[n, :])

                    if index is None:
                        y_rep[n, d] = binomial(1, np.exp(mu)/\
                                               (1 + np.exp(mu)), 1)
                    else:
                        y_rep[0, d] = binomial(1, np.exp(mu)/\
                                               (1 + np.exp(mu)), 1)

        return y_rep

    def lppd(self, y, index, S=20, x=None):
        """Compute the log pointwise predictive density (lppd) at a specified
        point.

        # TODO: test binary implementation

        This function implements equation 7.5 of 'Bayesian Data Analysis, Third
        Edition' for a single point.

        Parameters
        ----------
        y : array, shape ( D )
            The vector of target values at which to compute the log pointwise
            predictive density.

        index : int, optional
            The data sample index at which to sample.

        S : int, optional
            The number of samples to draw in order to compute the lppd.

        x : array, shape ( M ), optional
            If specified, a sample will be drawn for the data point specified
            by 'index', using the predictor values specified by 'x'.  If 'x' is
            not specified, the original predictor values for the 'index' data
            point will be used to draw the sample. Here, 'M' is the dimension of
            the predictors.

        Returns
        -------
        log_dens : float
            The computed lppd.

        References
        ----------
        Gelman et al, 'Bayesian Data Analysis, 3rd Edition'
        """
        if not (type(index) == int or type(index) == np.int64):
            raise ValueError('index must be an integer')

        if x is not None:
            if len(x.shape) != 1:
                raise ValueError('x must be a vector')

            if x.shape[0] != self.M_:
                raise ValueError('x has incorrect dimension')

        accums = np.zeros([self.D_, S])
        for s in range(0, S):  
            z = np.random.multinomial(1, self.R_[index, :])
            for d in range(0, self.D_):
                if self.target_type_[d] == 'gaussian':
                    co = multivariate_normal(\
                        self.w_mu_[:, d, z.astype(bool)][:, 0],
                            diag(self.w_var_[:, d, z.astype(bool)][:, 0]), 1)
                    if x is not None:
                        mu = dot(co, x)
                    else:
                        mu = dot(co, self.X_[index, :])

                    # Draw a precision value from the gamma distribution. Note
                    # that numpy uses a slightly different parameterization
                    scale = 1./self.lambda_b_[d, z.astype(bool)]
                    shape = self.lambda_a_[d, z.astype(bool)]
                    var = 1./gamma(shape, scale, size=1)

                    accums[d, s] = np.clip((1/sqrt(var*2*np.pi))*\
                      exp((-(0.5/var)*(y[d] - mu)**2).astype('float64')),
                      1e-300, 1e300)                   
                else:
                    # Target assumed binary
                    co = multivariate_normal(\
                        self.w_mu_[:, d, z.astype(bool)][:, 0],
                            self.w_covmat_[:, :, d, z.astype(bool)], 1)
                    if x is not None:
                        mu = dot(co, x)
                    else:
                        mu = dot(co, self.X_[index, :])

                    accums[d, s] = \
                        np.clip((np.exp(mu)**y[d])/(1 + np.exp(mu)),
                                1e-300, 1e300)

        log_dens = np.sum(np.log(np.mean(accums, 1)))

        return log_dens

    def log_likelihood(self):
        """Compute the log-likelihood given expected values of the latent 
        variables.

        TODO: Test implementation of binary targets

        Returns
        -------
        log_likelihood : float
            The log-likelihood
        """
        tmp_k = np.zeros(self.N_)
        for k in range(self.K_):
            tmp_d = np.ones(self.N_)
            for d in range(self.D_):
                # Some of the target variables can be missing (NaNs). Exclude
                # these from the computation.
                ids = ~np.isnan(self.Y_[:, d])
                mu = np.dot(self.X_, self.w_mu_[:, d, k])
                
                if self.target_type_[d] == 'gaussian':
                    v = self.lambda_b_[d, k]/self.lambda_a_[d, k]
                    co = 1/sqrt(2.*np.pi*v)

                    tmp_d[ids] *= co*np.exp(-((self.Y_[ids, d]-\
                                               mu[ids])**2)/(2.*v))
                else:
                    # Target assumed to be binary
                    tmp_d[ids] *= (np.exp(mu)**self.Y_[ids, d])/(1 + np.exp(mu))
                    
            tmp_k += self.R_[:, k]*tmp_d
        log_likelihood = np.sum(np.log(tmp_k))
                
        return log_likelihood

    def bic(self):
        """Computes the Bayesian Information Criterion according to formula 4.1 
        in the reference. Assumes same number of parameters for each trajectory 
        and for each target dimension.
    
        Returns
        -------
        bic_obs[, bic_groups] : float or tuple
            The BIC values. It 'groupby' was specified during the fitting 
            routine, the groups will indicate the number of individuals in the 
            data set. In this case, two BIC values will be computed: one in 
            which N is taken to be the number of observations (underestimates 
            true BIC) and one in which N is taken to be the number of subjects 
            (overstates true BIC). If the number of subjects can not be 
            ascertained, a single BIC value (where N is taken to be the number 
            of observations) is returned.
    
        References
        ----------
        Nagin DS, Group-based modeling of development. Harvard University Press; 
        2005.
        """
        ll = self.log_likelihood()
        num_trajs = np.sum(np.sum(self.R_, 0) > 0.0)
    
        # The first term below tallies the number of predictors for each
        # trajectory and for each target variable. Here we assume the same
        # number of predictors for each trajectory and for each target variable.
        # The second term tallies the number of parameters derived from the
        # total number of trajectories (i.e. the trajectory weights). They must
        # sum to one, that's why we subtract by one (the last parameter value is
        # determined by the sum of the others).
        num_params = (num_trajs*self.M_*self.D_) + (num_trajs - 1.)
    
        # Per recommendation in the reference, two BICs are computed: one in
        # which N is taken to be the number of observations (overstates true
        # "N") and one in which N is taken to be the number of subjects
        # (understates true "N"). Note that when we compute the total number of
        # observations, we sum across each of the target variable dimensions,
        # and for each dimension, we only consider those instances with
        # non-NaN values.
        num_obs_tally = 0
        for d in range(self.D_):
            num_obs_tally += np.sum(~np.isnan(self.Y_[:, d]))
        
        bic_obs = ll - 0.5*num_params*np.log(num_obs_tally)
        
        if self.gb_ is not None:
            num_subjects = self.gb_.ngroups
            bic_groups = ll - 0.5*num_params*np.log(num_subjects)
    
            return (bic_obs, bic_groups)
        else:
            return bic_obs
    
    def compute_waic2(self, S=1000):
        """Computes the Watanabe-Akaike (aka widely available) information
        criterion, using the variance of individual terms in the log predictive
        density summed over the n data points.

        TODO: Test implementation of binary target accomodation

        Parameters
        ----------
        S : integer, optional
            The number of draws from the posterior to use when computing the
            required expectations.

        Returns
        -------
        waic2 : float
            The Watanable-Akaike information criterion.

        References
        ----------
        Gelman et al, 'Bayesian Data Analysis, 3rd Edition'
        """
        accum = np.zeros([self.N_, self.D_, S])
        for k in np.where(self.sig_trajs_)[0]:
            for d in range(0, self.D_):
                if self.target_type_[d] == 'gaussian':
                    co = multivariate_normal(self.w_mu_[:, d, k],
                        diag(self.w_var_[:, d, k]), S)
                    mu = dot(co, self.X_.T)

                    # Draw a precision value from the gamma distribution. Note
                    # that numpy uses a slightly different parameterization
                    scale = 1./self.lambda_b_[d, k]
                    shape = self.lambda_a_[d, k]
                    var = 1./gamma(shape, scale, size=1)

                    prob = ((1/sqrt(2*np.pi*var))[:, newaxis]*\
                        exp(-(1/(2*var))[:, newaxis]*\
                            (mu - self.Y_[:, d])**2)).T
                else:
                    # Target assumed to be binary
                    co = multivariate_normal(self.w_mu_[:, d, k],
                        self.w_covmat_[:, :, d, k], S)
                    mu = dot(co, self.X_.T)
                    prob = ((np.exp(mu)**self.Y_[:, d])/(1 + np.exp(mu))).T

                accum[:, d, :] += self.R_[:, k][:, newaxis]*prob

        lppd = np.nansum(log(np.nanmean(accum, axis=2)))
        mean_ln_accum = np.nanmean(log(accum), axis=2)
        p_waic2 = np.nansum((1./(S-1.))*(log(accum) - \
            mean_ln_accum[:, :, newaxis])**2)
        waic2 = -2.*(lppd - p_waic2)

        return waic2

    def expand_mat(self, mat, which_m, which_d, expand_vec):
        """Helper function for trajectory coefficient initialization. It takes 
        in a matrix and returns a larger matrix (along the k-dimension) with
        values at the specified predictor and target location set to the values
        in expand_vec.

        Parameters
        ----------
        mat : array, shape ( M, D, kk )
            Matrix of trajectory coefficients. kk is expected to vary from 
            function call to function call

        which_m : int
            Matrix index corresponding to the predictor coefficient that will
            be expanded with respect to.

        which_d : int
            Matrix index corresponding to the target dimension of the predictor
            coefficient that will be expanded with respect to.

        expand_vec : array, shape ( L )
            Array of coefficient values that will be used to create the expanded
            matrix.

        Returns
        -------
        expanded_mat : array, shape ( M, D, kk*L)
            Expanded trajectory coefficient matrix
        """
        M = mat.shape[0]
        D = mat.shape[1]
        K = mat.shape[2]
    
        expand_size = expand_vec.shape[0]
        
        new_mat = np.zeros([M, D, K*expand_size])
        for kk in range(K):
            for xx in range(expand_size):
                new_mat[:, :, kk*expand_size + xx] = mat[:, :, kk]
            new_mat[which_m, which_d, \
                    (kk*expand_size):(kk*expand_size)+expand_size] = expand_vec
            
        return new_mat
    
    def prune_coef_mat(self, w_mu):
        """
        TODO: Test implementation of binary target accomodation
        """
        tmp_K = w_mu.shape[2]
        R_tmp = np.ones([self.N_, tmp_K])

        for k in range(tmp_K):
            for d in range(self.D_):
                ids = ~np.isnan(self.Y_[:, d])

                mu_tmp = np.dot(self.X_, w_mu[:, d, k])

                if self.target_type_[d] == 'gaussian':
                    var = self.lambda_b0_/self.lambda_a0_
                    R_tmp[ids, k] *= (1/(np.sqrt(2*np.pi*var[d])))*\
                        np.exp(-((self.Y_[ids, d] - mu_tmp[ids])**2)/(2*var[d]))
                else:
                    # Target assumed binary
                    R_tmp[ids, k] *= (np.exp(mu_tmp[ids])**self.Y_[ids, d])/\
                        (1 + np.exp(mu_tmp[ids]))
                    
        zero_ids = np.sum(R_tmp, 1) == 0
        R_tmp[zero_ids] = np.ones([np.sum(zero_ids), tmp_K])/tmp_K
        R = np.array((R_tmp.T/np.sum(R_tmp, 1)).T)
            
        # Within a group, all data instances must have the same probability of
        # belonging to each of the K trajectories
        if self.gb_ is not None:
            for g in self.gb_.groups.keys():
                tmp_ids = self.gb_.get_group(g).index.values
                tmp_vec = np.sum(np.log(R[tmp_ids, :] + \
                                        np.finfo(float).tiny), 0)
                vec = np.exp(tmp_vec + 700 - np.max(tmp_vec))
                vec = vec/np.sum(vec)
    
                # Now update the rows of 'normalized_mat'.
                R[tmp_ids, :] = vec                

        # By using R_sel here, we are preferring sets of trajectories that
        # tend not to be redundant. E.g. consider a matrix where each
        # trajectory has equal probability vs the scenario where each
        # instance is assigned to some trajectory with high probability. It
        # may be the case that both of these matrices, when summed across rows
        # equal the same sum vector, but we prefer the latter case. If we
        # did not do this little trick, then as the number of predictors and
        # targets grew, the set of selected trajectories would tend toward
        # the average trajectory, with each data instance having approximately
        # the probability of belonging to each those -- approximately the same
        # -- trajectories.
        R_sel = np.zeros([self.N_, tmp_K])
        for i in range(self.N_):
            col = np.where(R[i, :] == np.max(R[i, :]))[0][0]
            R_sel[i, col] = R[i, col]
                
        top_indices = np.argsort(-np.sum(R_sel, 0))[0:self.K_]        
        
        return w_mu[:, :, top_indices]
        
    def init_traj_params(self):
        """Initializes trajectory parameters. 
        """
        # TODO: What is best way to initialize xi?
        if self.num_binary_targets_ > 0:
            self.xi_ = np.ones([self.N_, self.num_binary_targets_, self.K_])
        else:
            self.xi_ = None
        
        if self.w_var_ is None:
            self.w_var_ = np.zeros([self.M_, self.D_, self.K_])
            for k in range(self.K_):
                self.w_var_[:, :, k] = np.array(self.w_var0_)
            
        if self.lambda_a_ is None and self.lambda_b_ is None:
            if self.gb_ is not None:
                scale_factor = self.gb_.ngroups
            else:
                scale_factor = self.N_
            self.lambda_a_ = scale_factor*np.ones([self.D_, self.K_])
            self.lambda_b_ = scale_factor*np.ones([self.D_, self.K_])
            for d in range(self.D_):
                # Generate a random sample from the prior
                scale = 1./self.lambda_b0_[d]
                shape = self.lambda_a0_[d]                                
                self.lambda_a_[d, :] *= gamma(shape, scale, size=self.K_)

        w_mu_tmp = np.zeros([self.M_, self.D_, 1])
        w_mu_tmp[:, :, 0] = self.w_mu0_
                    
        num_param_levels = 5
        if num_param_levels**(self.M_*self.D_) < self.K_:
            num_param_levels = \
                int(np.ceil(10**(np.log10(self.K_)/(self.D_*self.M_))))

        # Permute predictor and target indices. This is to further sample the
        # space of possible trajectories (given multipler restarts) on
        # initializeation. Also randomly sample (low and high) the range over
        # which cos are chosen to further jitter the initialization
        for m in np.random.permutation(range(self.M_)):
            for d in np.random.permutation(range(self.D_)):
                low = np.random.uniform(1.7, 2.3)
                high = np.random.uniform(1.7, 2.3)
                cos = np.linspace(self.w_mu0_[m, d] - \
                                  low*np.sqrt(self.w_var0_[m, d]),
                                  self.w_mu0_[m, d] + \
                                  high*np.sqrt(self.w_var0_[m, d]),
                                  num_param_levels)        
                w_mu_tmp = self.expand_mat(w_mu_tmp, m, d, cos)
                if w_mu_tmp.shape[2] > self.K_:
                    w_mu_tmp = self.prune_coef_mat(w_mu_tmp)
    
        self.w_mu_ = w_mu_tmp
        
    def init_R_mat(self, traj_probs=None, traj_probs_weight=None):
        """Initializes 'R_', using the stick-breaking construction. Also
        enforces any longitudinal constraints.

        Parameters
        ----------
        traj_probs : array, shape ( K ), optional
            A priori probabilitiey of each of the K trajectories. Each element 
            must be >=0 and <= 1, and all elements must sum to one.

        traj_probs_weight : float, optional
            Value between 0 and 1 inclusive that controls how traj_probs are 
            combined with randomly generated trajectory probabilities using 
            stick-breaking: 
            traj_probs_weight*traj_probs + (1-traj_probs_weight)*random_probs.
        """
        # Draw a weight vector from the stick-breaking process
        #tmp = np.random.beta(1, self.alpha_, self.K_)
        #one_tmp = 1. - tmp
        #vec = np.array([np.prod(one_tmp[0:k])*tmp[k] for k in range(self.K_)])

        # Each trajectory gets equal weight
        vec = np.ones(self.K_)/self.K_

        if (traj_probs is None and traj_probs_weight is not None) or \
           (traj_probs_weight is None and traj_probs is not None):
            warnings.warn('Both traj_probs and traj_probs_weight \
            should be None or non-None')

        if traj_probs is not None and traj_probs_weight is not None:
            assert traj_probs_weight >= 0 and traj_probs_weight <= 1, \
                "Invalid traj_probs_weight"
            assert np.isclose(np.sum(traj_probs), 1), \
                "Invalid traj_probs"
            init_traj_probs = traj_probs_weight*traj_probs + \
                (1-traj_probs_weight)*vec
        else:
            init_traj_probs = vec

        if np.sum(init_traj_probs) < 0.95:
            warnings.warn("Initial trajectory probabilities sum to {}. \
            Alpha may be too high.".format(np.sum(init_traj_probs)))
        
        self.R_ = self.predict_proba_(self.X_, self.Y_, init_traj_probs)

    def predict_proba_(self, X, Y, traj_probs=None):
        """Compute the probability that each data instance belongs to each of
        the 'k' clusters. Note that 'X' and 'Y' can be "new" data; that is,
        data that was not necessarily used to train the model.

        TODO: Test implementation of binary data accomodation

        Returns
        -------
        X : array, shape ( N, M )
            Each row is an M-dimensional predictor vector for the nth data
            sample. Note that 'N' here does not necessarily correspond to the
            same number of data points ('N') that was used to train the model.

        Y : array, shape ( N, D )
            Each row is a D-dimensional target vector for the nth data sample.
            Note that 'N' here does not necessarily correspond to the same
            number of data points ('N') that was used to train the model.

        traj_probs : array, shape ( K ), optional
            A priori probabilitiey of each of the K trajectories. Each element 
            must be >=0 and <= 1, and all elements must sum to one. For general
            usage, this should be not set (non-None settings for internal 
            usage).

        Returns
        -------
        R : array, shape ( N, K )
            Each element of this matrix represents the probability that
            instance 'n' belongs to cluster 'k'.
        """
        N = X.shape[0]
        D = Y.shape[1]

        if traj_probs is None:
            traj_probs = np.sum(self.R_, 0)/np.sum(self.R_)
            
        R_tmp = np.ones([N, self.K_])
        for k in range(self.K_):
            for d in range(D):
                ids = ~np.isnan(Y[:, d])
                mu_tmp = np.dot(X, self.w_mu_[:, d, k])

                if self.target_type_[d] == 'gaussian':
                    var_tmp = self.lambda_b_[d, k]/self.lambda_a_[d, k]
                    R_tmp[ids, k] *= (1/(np.sqrt(2*np.pi*var_tmp)))*\
                        np.exp(-((Y[ids, d] - mu_tmp[ids])**2)/(2*var_tmp))
                else:
                    # Target assumed binary
                    R_tmp[ids, k] *= (np.exp(mu_tmp[ids])**Y[ids, d])/\
                        (1 + np.exp(mu_tmp[ids]))
                    
            R_tmp[:, k] *= traj_probs[k]

        zero_ids = np.sum(R_tmp, 1) == 0
        R_tmp[zero_ids] = np.ones([np.sum(zero_ids), self.K_])/self.K_
        R = np.array((R_tmp.T/np.sum(R_tmp, 1)).T)
        
        # Within a group, all data instances must have the same probability of
        # belonging to each of the K trajectories
        if self.gb_ is not None:
            for g in self.gb_.groups.keys():
                tmp_ids = self.gb_.get_group(g).index.values
                tmp_vec = np.sum(np.log(R[tmp_ids, :] + \
                                        np.finfo(float).tiny), 0)
                vec = np.exp(tmp_vec + 700 - np.max(tmp_vec))
                vec = vec/np.sum(vec)

                # Now update the rows of 'normalized_mat'.
                R[tmp_ids, :] = vec                
        
        return R

    def augment_df_with_traj_info(self, target_names, predictor_names, df,
                                  groupby=None):
        """Compute the probability that each data instance belongs to each of
        the 'k' clusters. Note that 'X' and 'Y' can be "new" data; that is,
        data that was not necessarily used to train the model.

        TODO: Test implementation of binary target accomodation

        Parameters
        ----------

        Returns
        -------
        R : array, shape ( N, K )
            Each element of this matrix represents the probability that
            instance 'n' belongs to cluster 'k'.
        """
        df_ = pd.DataFrame(df)
        
        N = df_.shape[0]
        D = len(target_names)

        assert set(target_names) == set(self.target_names_), \
            "Target name discrepancy"
        assert set(predictor_names) == set(self.predictor_names_), \
            "Predictor name discrepancy"

        Y = df_[self.target_names_].values
        X = df_[self.predictor_names_].values

        gb = None
        if groupby is not None:
            gb = df_.groupby(groupby)
        
        traj_probs = np.sum(self.R_, 0)/np.sum(self.R_)
            
        R_tmp = np.ones([N, self.K_])
        for k in range(self.K_):
            for d in range(D):
                ids = ~np.isnan(Y[:, d])
                mu_tmp = np.dot(X, self.w_mu_[:, d, k])

                if self.target_type_[d] == 'gaussian':
                    var_tmp = self.lambda_b_[d, k]/self.lambda_a_[d, k]                    
                    R_tmp[ids, k] *= (1/(np.sqrt(2*np.pi*var_tmp)))*\
                        np.exp(-((Y[ids, d] - mu_tmp[ids])**2)/(2*var_tmp))
                else:
                    # Target assumed binary
                    R_tmp[ids, k] *= (np.exp(mu_tmp[ids])**Y[ids, d])/\
                        (1 + np.exp(mu_tmp[ids]))
                    
            R_tmp[:, k] *= traj_probs[k]

        zero_ids = np.sum(R_tmp, 1) == 0
        R_tmp[zero_ids] = np.ones([np.sum(zero_ids), self.K_])/self.K_
        R = np.array((R_tmp.T/np.sum(R_tmp, 1)).T)
        
        # Within a group, all data instances must have the same probability of
        # belonging to each of the K trajectories
        if gb is not None:
            for g in gb.groups.keys():
                tmp_ids = gb.get_group(g).index.values
                tmp_vec = np.sum(np.log(R[tmp_ids, :] + \
                                        np.finfo(float).tiny), 0)
                vec = np.exp(tmp_vec + 700 - np.max(tmp_vec))
                vec = vec/np.sum(vec)

                # Now update the rows of 'normalized_mat'.
                R[tmp_ids, :] = vec                

        # Now augment the dataframe with trajectory info
        traj = []
        for i in range(N):
            traj.append(np.where(np.max(R[i, :]) == R[i, :])[0][0])
        df_['traj'] = traj

        for s in np.where(self.sig_trajs_)[0]:
            df_['traj_{}'.format(s)] = R[:, s]

        return df_
            
    def compute_lower_bound(self):
        """Compute the variational lower bound

        NOTE: Currently not implemented.

        Returns
        -------
        lower_bound : float
            The variational lower bound
        """
        pass
        
    def to_df(self):
        """Adds to the current data frame columns containing trajectory 
        assignments and probabilities.

        Returns
        -------
        df : Pandas dataframe
            Column 'traj' contains integer values indicating which trajectory
            the data instance belongs to. If the current data frame already has 
            a 'traj' column, the new columns will be called 'traj_'. 'traj_*' 
            contain actual probabilities that the data instance belongs to a 
            particular trajectory.
        """
        traj = []
        for i in range(0, self.R_.shape[0]):
            traj.append(where(max(self.R_[i, :]) == self.R_[i, :])[0][0]) 

        self.df_['traj'] = traj

        for s in np.where(self.sig_trajs_)[0]:
            self.df_['traj_{}'.format(s)] = self.R_[:, s]
        
        return self.df_  

    def plot(self, x_axis, y_axis, which_trajs=None, show=True,
             min_traj_prob=0, max_traj_prob=1, traj_map=None):
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

        """
        # Compute the probability vector for each trajectory

        if self.gb_ is not None:
            num_individuals = self.gb_.ngroups
            R_per_individual = np.zeros([self.gb_.ngroups, self.K_])
            for (ii, kk) in enumerate(self.gb_.groups.keys()):
                tmp_index = self.gb_.get_group(kk).index[0]
                R_per_individual[ii, :] = self.R_[tmp_index, :]
            traj_prob_vec = np.sum(R_per_individual, 0)/np.sum(R_per_individual)
        else:
            num_individuals = self.N_
            traj_prob_vec = np.sum(self.R_, 0)/np.sum(self.R_)
        
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
        ax.scatter(df_traj[x_axis].values,
                   df_traj[y_axis].values,
                   edgecolor='k', color='None', alpha=0.1)

        for (traj_inc, tt) in enumerate(traj_ids):
            if traj_prob_vec[tt] >= min_traj_prob and \
               traj_prob_vec[tt] <= max_traj_prob:
                
                ids_tmp = df_traj.traj.values == tt
                ax.scatter(df_traj[ids_tmp][x_axis].values,
                           df_traj[ids_tmp][y_axis].values,
                           edgecolor='k',
                           color=cmap(traj_id_to_cmap_index[traj_map_[tt]]),
                           alpha=0.5)

                n_traj = int(traj_prob_vec[tt]*num_individuals)
                perc_traj = traj_prob_vec[tt]*100

                co = self.w_mu_[:, target_index, tt]
                if self.target_type_[target_index] == 'gaussian':
                    std = np.sqrt(self.lambda_b_[target_index][tt]/\
                                  self.lambda_a_[target_index][tt])
                    y_tmp = np.dot(co, X_tmp.T)
                    ax.fill_between(x_dom, y_tmp-2*std, y_tmp+2*std,
                            color=cmap(traj_id_to_cmap_index[traj_map_[tt]]),
                            alpha=0.3)
                else:
                    # Target assumed binary
                    y_tmp = np.exp(np.dot(co, X_tmp.T))/\
                        (1 + np.exp(np.dot(co, X_tmp.T)))
                    
                ax.plot(x_dom, y_tmp,
                        color=cmap(traj_id_to_cmap_index[traj_map_[tt]]),
                        linewidth=3,
                        label='Traj {} (N={}, {:.1f}%)'.\
                        format(traj_map_[tt], n_traj, perc_traj))

        ax.set_xlabel(x_axis, fontsize=16)
        ax.set_ylabel(y_axis, fontsize=16)    
        plt.tight_layout()
        ax.legend()

        if show:
            plt.show()

        return ax    


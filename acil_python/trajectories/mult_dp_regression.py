from argparse import ArgumentParser
from acil_python.trajectories.get_longitudinal_constraints_graph \
  import get_longitudinal_constraints_graph
import numpy as np
from numpy import abs, dot, mean, log, sum, exp, tile, max, sum, isnan, diag, \
     sqrt, pi, newaxis, outer, genfromtxt, where
from numpy.random import multivariate_normal, randn, gamma
import networkx as nx
from scipy.optimize import minimize_scalar
from scipy.misc import logsumexp
from scipy.special import psi, gammaln
import pandas as pd
import pdb, sys, pickle
    
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
        target variable. 

    lambda_b : array, shape ( D, K ), optional
        For component 'K' and target dimension 'D', this is the second parameter 
        of the posterior Gamma distribution describing the precision of the 
        target variable. 
    """
    def __init__(self, w_mu0, w_var0, lambda_a0, lambda_b0, alpha, K=10, 
                 prob_thresh=0.001):
        self.w_mu0_ = w_mu0
        self.w_var0_ = w_var0
        self.lambda_a0_ = lambda_a0
        self.lambda_b0_ = lambda_b0        
        self.K_ = K
        self.M_ = self.w_mu0_.shape[0]
        self.D_ = self.w_mu0_.shape[1]        
        self.alpha_ = alpha
        self.prob_thresh_ = prob_thresh

        # For recording the lower-bound terms
        self.lower_bounds_ = []
        
        # X_ and Y_ will become defined when 'fit' is called
        self.X_ = None 
        self.Y_ = None
        self.data_names_ = None
        self.target_names_ = None
        self.predictor_names_ = None                
        
        assert self.w_mu0_.shape[0] == self.w_var0_.shape[0] and \
          self.w_mu0_.shape[1] == self.w_var0_.shape[1], \
          "Shape mismatch for w_mu0_ and w_var0_"

        assert self.lambda_a0_.shape[0] == self.lambda_b0_.shape[0], \
          "Shape mismatch for lambda_a0_ and lambda_b0_"

        assert self.w_mu0_.shape[1] == self.lambda_a0_.shape[0], \
          "Target dimension mismatch"
          
    def fit(self, X, Y, iters=None, tol=None, R=None, v_a=None, v_b=None, 
            w_mu=None, w_var=None, lambda_a=None, lambda_b=None, 
            constraints=None, data_names=None, target_names=None,
            predictor_names=None):
        """Run the DP proportion mixture model algorithm using predictors 'X'
        and proportion data 'Y'. 

        Parameters
        ----------
        X : array, shape ( N, M )
            Each row is an M-dimensional predictor vector for the nth data
            sample.

        Y : array, shape ( N, D )
            Each row is a D-dimensional target vector for the nth data sample.

        iters : int, optional
            Number of variational inference iterations to run. Note that if both
            'iters' and 'tol are specified, computation will terminate when both
            conditions are met.

        tol : float, optional
            The tolerance used for assessing convergence using the variational
            lower bound. Tolerance is defined as the percentage change in 
            variational lower bound from one iteration to the next. Note that 
            if both 'iters' and 'tol' are specified, computation will 
            terminate when both conditions are met. Also, note that 
            specification of 'tol' requires computation of the  variational 
            lower bound at each iteration. 
            
        R : array, shape ( N, K ), optional
            Each element of this matrix represents the posterior probability 
            that instance 'n' belongs to cluster 'k'. If specified, the 
            algorithm will be initialized with this matrix, otherwise a default
            matrix (randomly generated) will be used.
            
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
            be initialized with this matrix. 

        lambda_b : array, shape ( D, K ), optional
            For component 'K' and target dimension 'D', this is the second
            parameter of the posterior Gamma distribution describing the 
            precision of the target variable. If specified, the algorithm will
            be initialized with this matrix.             

        constraints : networkx graph, optional
            The constraints are encoded in a networkx graph, each node should
            indicate an instance index, and edges indicate constraints. Each
            edge should have a string attribute called 'constraint', which must
            take a value of 'must_link', 'cannot_link', or 'longitudinal'. If
            constraints are specified, the variation inference update equations
            will take them into account. See
            'ConstrainedNonparametricGaussianProcessRegression' in the 
            repositrory 'Documentation' folder for a description of how the 
            updates are incorporated.

        data_names : list of N strings, optional
            Stores a name for each data item represented in the X and Y
            matrices. E.g. ['Joe_baseline', 'Jane_baseline',
            'Joe_followup', ...].

        target_names : list of D strings, optional
            Stores the names of the D target variables.

        predictor_names : list of M strings, optional
            Stores the names of the M predictors
        """
        if tol is None and iters is None:
            raise ValueError('Neither tol nor iters has been set')
        
        if len(np.array(X).shape) == 0:
            self.X_ = np.atleast_2d(X)
        elif len(np.array(X).shape) == 1:
            self.X_ = np.atleast_2d(X).T
        else:
            self.X_ = np.atleast_2d(X)

        if len(np.array(Y).shape) == 0:
            self.Y_ = np.atleast_2d(Y)
        elif len(np.array(Y).shape) == 1:
            self.Y_ = np.atleast_2d(Y).T
        else:
            self.Y_ = np.atleast_2d(Y)

        if data_names is not None:
            assert len(data_names) == self.X_.shape[0], \
              "Number of data names and number of data points does not match"
            assert len(set(data_names)) == len(data_names), \
              "Duplicate data name found"
            self.data_names_ = data_names

        if target_names is not None:
            assert len(target_names) == self.Y_.shape[1], \
              "Number of target names does not match target dimension"
            assert len(set(target_names)) == len(target_names), \
              "Duplicate target name found"
            self.target_names_ = target_names            

        if predictor_names is not None:
            assert len(predictor_names) == self.X_.shape[1], \
              "Number of predictor names does not match predictor dimension"
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
        self.R_ = R 
        self.lambda_a_ = lambda_a
        self.lambda_b_ = lambda_b
        self.w_mu_ = w_mu
        self.w_var_ = w_var     
        self.v_a_ = v_a
        self.v_b_ = v_b
        self.constraints_ = constraints

        # If constraints have been specified, get the connected subgraphs
        self.constraint_subgraphs_ = []
        if self.constraints_ is not None:
            self.constraint_subgraphs_ = \
              self.get_constraint_subgraphs_(self.constraints_)
        
        # Initialize the latent variables if needed
        if self.R_ is None:
            self.init_R_mat()
            
        if self.lambda_a_ is None:
            self.lambda_a_ = np.zeros([self.D_, self.K_])
            for k in xrange(0, self.K_):
                self.lambda_a_[:, k] = self.lambda_a0_

        if self.lambda_b_ is None:
            self.lambda_b_ = np.zeros([self.D_, self.K_])
            for k in xrange(0, self.K_):
                self.lambda_b_[:, k] = self.lambda_b0_

        if self.w_mu_ is None:
            self.w_mu_ = np.zeros([self.M_, self.D_, self.K_])
            for k in xrange(0, self.K_):
                self.w_mu_[:, :, k] = self.w_mu0_

        if self.w_var_ is None:
            self.w_var_ = np.zeros([self.M_, self.D_, self.K_])
            for k in xrange(0, self.K_):
                self.w_var_[:, :, k] = self.w_var0_

        if self.v_a_ is None:
            self.v_a_ = np.ones(self.K_)

        if self.v_b_ is None:
            self.v_b_ = self.alpha_*np.ones(self.K_)

        if iters is None:
            iters = 1

        compute_lower_bound = True
        if tol is None:
            compute_lower_bound = False

        inc = 0
        prev = -sys.float_info.max
        perc_change = sys.float_info.max
        while inc < iters or (perc_change > tol and tol is not None):
            inc += 1
            self.update_w()
            self.update_lambda()
            self.update_v()
            self.R_ = self.update_z(self.X_, self.Y_,
                                    self.constraint_subgraphs_)

            print("iter {},  {}".format(inc, sum(self.R_, 0)))
              
            if compute_lower_bound:                
                curr = self.compute_lower_bound()
                self.lower_bounds_.append(curr)                
                perc_change = 100.*((curr-prev)/np.abs(prev))
                prev = curr

    def update_v(self):
        """Updates the parameters of the Beta distributions for latent
        variable 'v' in the variational approximation.        
        """    
        self.v_a_ = np.zeros(self.K_)
        self.v_b_ = np.zeros(self.K_)

        for k in np.arange(0, self.K_):
            self.v_a_[k] = 1.0 + np.sum(self.R_[:, k]) 
    
            tmpSum = 0.0
            for i in np.arange(k+1, self.K_):
                tmpSum = tmpSum + np.sum(self.R_[:, i])
    
            self.v_b_[k] = self.alpha_ + tmpSum
              
    def update_z(self, X, Y, constraint_subgraphs):
        """Update the variational distribution over latent variable Z.

        Parameters
        ----------
        X : array, shape ( N, M )
            Each row is an M-dimensional predictor vector for the nth data
            sample.

        Y : array, shape ( N, D )
            Each row is a D-dimensional target vector for the nth data sample.            

        constraint_subgraphs : list of networkx graphs
            The constraints are encoded in networkx graphs, each node should
            indicate an instance index, and edges indicate constraints. Each
            edge should have a string attribute called 'constraint', which must
            take a value of 'must_link', 'cannot_link', or 'longitudinal'. Each
            element of the list is a graph that corresponds to a collection
            of linked data instances.
            
        Returns
        -------
        R : array, shape ( N, K )
            Each element of this matrix represents the probability that
            instance 'n' belongs to cluster 'k'.                 
        """
        expec_ln_v = psi(self.v_a_) - psi(self.v_a_ + self.v_b_)
        expec_ln_1_minus_v = psi(self.v_b_) - psi(self.v_a_ + self.v_b_)

        N = X.shape[0]
        ln_rho = np.zeros([N, self.K_])
        for k in xrange(0, self.K_):
            ln_rho[:, k] += expec_ln_v[k]
            for i in xrange(0, k):
                ln_rho[:, k] += expec_ln_1_minus_v[i]

            for d in xrange(0, self.D_):
                non_nan_ids = ~np.isnan(Y[:, d])
                tmp = 0.0
                for i in xrange(0, self.M_):
                    for j in xrange(i, self.M_):
                        if i == j:
                            tmp += (self.w_var_[i, d, k] + \
                                self.w_mu_[i, d, k]**2)*X[non_nan_ids, i]**2
                        else:
                            tmp += 2*self.w_mu_[i, d, k]*\
                              self.w_mu_[j, d, k]*X[non_nan_ids, i]*\
                              X[non_nan_ids, j]
                
                ln_rho[non_nan_ids, k] += 0.5*(psi(self.lambda_a_[d, k]) - \
                    log(self.lambda_b_[d, k]) - log(2*np.pi) - \
                    (self.lambda_a_[d, k]/self.lambda_b_[d ,k])*\
                    (tmp - 2*Y[non_nan_ids, d]*dot(X[non_nan_ids, :], \
                    self.w_mu_[:, d, k]) + Y[non_nan_ids, d]**2))

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

        # Now apply constraints
        self.apply_constraints(constraint_subgraphs, R)

        return R

    def apply_constraints(self, constraint_subgraphs, R):
        """
        """
        num_subgraphs = len(constraint_subgraphs)
    
        for i in xrange(0, num_subgraphs):
            is_longitudinal = True
            for e in constraint_subgraphs[i].edges():
                if constraint_subgraphs[i][e[0]][e[1]]['constraint'] != \
                  'longitudinal':
                  is_longitudinal = False
                  break
            if is_longitudinal:
                self.update_R_rows_longitudinal_(constraint_subgraphs[i], R)
            else:
                self.update_R_rows_(constraint_subgraphs[i],
                                    self.prob_thresh_, R)
            
    def update_w(self):
        """Update the variational distribution over latent variable w.
        """
        for m in xrange(0, self.M_):
            ids = np.ones(self.M_, dtype=bool)
            ids[m] = False
            for d in xrange(0, self.D_):
                non_nan_ids = ~np.isnan(self.Y_[:, d])
                for k in xrange(0, self.K_):
                    if sum(self.R_[:, k]) > 0.0:
                        self.w_var_[m, d, k] = ((self.lambda_a_[d, k]/\
                            self.lambda_b_[d, k])*\
                            sum(self.R_[:, k]*self.X_[:, m]**2) + \
                            1.0/self.w_var0_[m, d])**-1

                        self.w_mu_[m, d, k] = self.w_var_[m, d, k]*\
                          (-(self.lambda_a_[d, k]/\
                            self.lambda_b_[d, k])*sum(self.R_[non_nan_ids, k]*\
                                self.X_[non_nan_ids, m]*\
                                (dot(self.X_[:, ids][non_nan_ids, :], \
                                                   self.w_mu_[ids, d, k]) - \
                                 self.Y_[non_nan_ids, d])) + \
                                 self.w_mu0_[m, d]/self.w_var0_[m, d])

    def update_lambda(self):
        """Update the variational distribution over latent variable lambda.
        """
        for d in xrange(0, self.D_):
            non_nan_ids = ~np.isnan(self.Y_[:, d])
            for k in xrange(0, self.K_):
                if sum(self.R_[:, k]) > 0.0:
                    self.lambda_a_[d, k] = self.lambda_a0_[d] + \
                      0.5*sum(self.R_[:, k])
    
                    tmp = 0.0
                    for i in xrange(0, self.M_):
                        for j in xrange(i, self.M_):
                            if i == j:
                                tmp += (self.w_var_[i, d, k] + \
                                        self.w_mu_[i, d, k]**2)*self.X_[:, i]**2
                            else:
                                tmp += 2*self.w_mu_[i, d, k]*\
                                  self.w_mu_[j, d, k]*self.X_[:, i]*\
                                  self.X_[:, j]

                    self.lambda_b_[d, k] = self.lambda_b0_[d] + \
                      0.5*sum(self.R_[non_nan_ids, k]*(tmp[non_nan_ids] - 2*self.Y_[non_nan_ids, d]*\
                        np.dot(self.X_[non_nan_ids, :], self.w_mu_[:, d, k]) + \
                        self.Y_[non_nan_ids, d]**2))                    

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
            indices = xrange(0, self.N_)
            y_rep = np.zeros([self.N_, self.D_])
            X = self.X_
        else:
            if not (type(index) == int or type(index) == np.int64):
                raise ValueError('index must be an integer if specified')

            indices = xrange(index, index+1)
            y_rep = np.zeros([1, self.D_])

            if x is not None:
                if len(x.shape) != 1:
                    raise ValueError('x must be a vector')

                if x.shape[0] != self.M_:
                    raise ValueError('x has incorrect dimension')
                            
        for n in indices:
            z = np.random.multinomial(1, self.R_[n, :])
            for d in xrange(0, self.D_):        
                co = multivariate_normal(self.w_mu_[:, d, z.astype(bool)][:, 0],
                    diag(self.w_var_[:, d, z.astype(bool)][:, 0]), 1)
                if x is not None:
                    mu = dot(co, x)
                else:
                    mu = dot(co, self.X_[n, :])

                # Draw a precision value from the gamma distribution. Note
                # that numpy uses a slightly different parameterization
                scale = 1./self.lambda_b_[d, z.astype(bool)]
                shape = self.lambda_a_[d, z.astype(bool)]
                var = 1./gamma(shape, scale, size=1)                

                if index is None:
                    y_rep[n, d] = sqrt(var)*randn(1) + mu
                else:
                    y_rep[0, d] = sqrt(var)*randn(1) + mu

        return y_rep

    def lppd(self, y, index, S=20, x=None):
        """Compute the log pointwise predictive density (lppd) at a specified 
        point.

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

        log_dens = 0.0
        accums = np.zeros([self.D_, S])
        for s in xrange(0, S):  
            z = np.random.multinomial(1, self.R_[index, :])            
            for d in xrange(0, self.D_):
                co = multivariate_normal(self.w_mu_[:, d, z.astype(bool)][:, 0],
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

        log_dens = np.sum(np.log(np.mean(accums, 1)))

        return log_dens
    
    def compute_waic2(self, S=20):
        """Computes the Watanabe-Akaike (aka widely available) information
        criterion, using the variance of individual terms in the log predictive
        density summed over the n data points.

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
        for s in xrange(0, S):
            for n in xrange(0, self.N_):
                z = np.random.multinomial(1, self.R_[n, :])
                for d in xrange(0, self.D_):
                    co = multivariate_normal(self.w_mu_[:, d, 
                                                        z.astype(bool)][:, 0],
                        diag(self.w_var_[:, d, z.astype(bool)][:, 0]), 1)
                    mu = dot(co, self.X_[n, :])

                    # Draw a precision value from the gamma distribution. Note
                    # that numpy uses a slightly different parameterization
                    scale = 1./self.lambda_b_[d, z.astype(bool)]
                    shape = self.lambda_a_[d, z.astype(bool)]
                    var = 1./gamma(shape, scale, size=1)

                    accum[n, d, s] = (1/sqrt(2*np.pi*var))*exp(-(1/(2*var))*\
                        (mu - self.Y_[n, d])**2)

        lppd = np.nansum(log(np.nanmean(accum, axis=2)))

        mean_ln_accum = np.nanmean(log(accum), axis=2)
        tmp = np.zeros([self.N_, self.D_])
        for s in xrange(0, S):
            tmp += (1./(S-1.))*(log(accum[:, :, s]) - mean_ln_accum)**2

        p_waic2 = np.nansum(tmp)
        waic2 = -2.*(lppd - p_waic2)

        return waic2

    def update_R_rows_(self, constraint_subgraph, prob_thresh, R):
        """Updates 'R' according to the constraints supplied in
        'constraint_subgraph'
    
        The nodes in the subgraph designate which rows (variables) to update in
        'R', the matrix that encodes the expected values of each indicator
        variable, z. 
    
        Parameters
        ----------
        constraint_subgraph : networkx graph
            The constraints are encoded in a networkx graph, each node should
            indicate an instance index, and edges indicate constraints. Each
            edge should have a string attribute called 'constraint', which must
            take a value of 'must_link', 'cannot_link', or 'longitudinal'.
    
        prob_thresh : float
            The probability threshold is a scalar value in the interval (0, 1). 
            It indicates the minimum value of component 'k' of the latent 
            variable, 'z', that a given point needs to have to be considered a 
            possible member of regression curve 'k'. Any state configuration 
            having an individual instance with a probability less than this 
            threshold will be considered an impossible configuration. This 
            greatly increase computational efficiency

        R : array, shape ( N, K )
            Each element of this matrix represents the probability that
            instance 'n' belongs to cluster 'k'.            
        """
        # Collect information about the instance IDs represented by the nodes in
        # the 'constraint_subgraph'. Note that the 'nodes()' operation on the
        # graph produces a sorted list by default. We sort again for security,
        # esp. given that we don't expect a large number of nodes in any one
        # subgraph, so the sort operation introduces minimal overhead.
        num_nodes = len(constraint_subgraph.nodes())    
        constraint_subgraph.nodes().sort()
        node_ids = constraint_subgraph.nodes()    

        node_to_index = {}
        inc = 0
        for n in node_ids:
            node_to_index[n] = inc
            inc += 1
        
        # We will need to consider all possible 'state' configurations. Each of
        # variables can be in one of K states, for a total of K^(num_nodes)
        # possibilities. We will represent the state of the collection of 
        # variables with the matrix 'state_mat'
        num_states = R.shape[1]
    
        # Now loop over all possible state matrices. For each state matrix, 
        # compute the unnormalized probability (this is the quantity in Eq. 20 
        # in the 'ConstraintedNonparametricGaussianProcessRegression' document, 
        # without the partition function). Each computed probability is 
        # multiplied by the corresponding state matrix and added to 
        # 'state_mat_accum'. The total unnormalized probability is accumulated 
        # in 'prob_accum'. We take advantage of the fact that if a given 
        # instance has a probability lower than 'prob_thresh' for a given 
        # state, we can effectively consider any configuration including that 
        # state to be "impossible". This greatly reduces the computational 
        # burden required: for each instance in the subgraph, we need only 
        # record the columns in 'R' for which its probability is >= 
        # 'prob_thresh'. We do this for each of the instances in the subgraph 
        # and only loop over those state configurations.
        sig_cols = []
        for i in xrange(0, num_nodes):
            cols = \
              (np.nonzero(R[node_ids[i], :] > prob_thresh)[0]).tolist()
            sig_cols.append(cols)
            
        # The above initialization of 'sig_cols' does not take into 
        # consideration the relationship between the rows (nodes).
        # For example, suppose there were 2 nodes that were longitudinally 
        # linked and only two states for each node. Then if 'sig_cols' wound
        # up being [[0], [1]], the code below would generate an error because
        # the only configuration tested would be [[1, 0], [0, 1]], which would
        # given a probability of zero (an impossible configuration, given that
        # the data points are longitudinally linked). Therefore, we need to 
        # look at all the edges of this subgraph and augment 'sig_cols' such 
        # each pair of nodes that are longitudinally linked or must-linked
        # correspond to the same set of columns (cluster assignments)
        for e in constraint_subgraph.edges():
            if constraint_subgraph[e[0]][e[1]]['constraint'] == \
              'longitudinal' or constraint_subgraph[e[0]][e[1]]['constraint'] \
              == 'must_link':
              tmp = list(np.sort(list(set(sig_cols[node_to_index[e[0]]] + \
                                          sig_cols[node_to_index[e[1]]]))))
              sig_cols[node_to_index[e[0]]] = tmp
              sig_cols[node_to_index[e[1]]] = tmp

        num_state_mats = 1.0
        for i in xrange(0, num_nodes):
            num_state_mats *= len(sig_cols[i])
                                  
        # Initialize the state matrix
        state_mat = np.zeros([num_nodes, num_states])    
        for n in xrange(0, num_nodes):
            state_mat[n, sig_cols[n][0]] = 1.0
    
        state_mat_accum = np.zeros([num_nodes, num_states])
        prob_accum = 0.0

        for i in np.arange(0, num_state_mats):        
            prob = self.compute_subgraph_unnormalized_(constraint_subgraph, 
                state_mat)            
            state_mat_accum += prob*state_mat
            prob_accum += prob
            self.inc_state_mat_(state_mat, sig_cols)

        # Now create the normalized matrix
        assert prob_accum > 0.0, 'prob_accum is 0'
        normalized_mat = (1.0/prob_accum)*state_mat_accum
        
        # Lastly, we have to insert the rows of 'normalized_mat' properly into
        # 'R_'. Each row of 'normalized_mat' corresponds to an instance, and
        # the rows are ordered so that the first row corresponds to the smallest
        # instance ID, the second row corresponds to the next smallest instance 
        # ID, etc.     
        for i in np.arange(0, num_nodes):
            R[node_ids[i], :] = normalized_mat[i, :]
        
    def update_R_rows_longitudinal_(self, constraint_subgraph, R):
        """Updates 'R_' by enforcing longitudinal points to have the same row
        probabilities.
    
        The nodes in the subgraph designate which rows (variables) to update in
        'R_', the matrix that encodes the expected values of each indicator
        variable, z. 
    
        Parameters
        ----------
        constraint_subgraph : networkx graph
            The constraints are encoded in a networkx graph, each node should
            indicate an instance index, and edges indicate constraints. Each
            edge should have a string attribute called 'constraint' which takes
            a value of 'longitudinal'.

        R : array, shape ( N, K )
            Each element of this matrix represents the probability that
            instance 'n' belongs to cluster 'k'.
        
        """
        node_ids = [n for n in constraint_subgraph.nodes()]

        # The following operation computes the expectation. Because of the 
        # longitudinal constraint, the only state configurations that have
        # non-zero probability are those that put all the data instances in
        # this group into the same cluster. Note that we take the sum of the
        # log to deal with the product of (potentially) many small values.
        tmp_vec = np.sum(np.log(R[node_ids, :] + np.finfo(float).tiny), 0)
        vec = np.exp(tmp_vec + 700 - np.max(tmp_vec))
        vec = vec/np.sum(vec)
        
        # Now update the rows of 'normalized_mat'.
        R[node_ids, :] = vec
            
    def compute_subgraph_unnormalized_(self, constraint_subgraph, state_mat):
        """Computes the quantity inside the brackets in Eq. 20 in
        'ConstrainedNonparametricGaussianProcessRegression' (without the 
        partition function), for a given state matrix.
    
        Parameters
        ----------
        constraint_subgraph : networkx graph
            The constraints are encoded in a networkx graph, each node should
            indicate an instance index, and edges indicate constraints. Each
            edge should have a string attribute called 'constraint', which must
            either take a value of 'must_link' or 'cannot_link'. 
        
        state_mat : array, shape ( num_rows, num_cols )
            A matrix in which each row has exactly one element equal to 1.0. The
            other elements are equal to 0.0. There should be the same number of
            rows as there are nodes in 'constraint_subgraph'
    
        Returns
        -------
        unnormalized_prob : float
            The quantity inside the brackets in Eq. 20 in
            'ConstrainedNonparametricGaussianProcessRegression', without the
            partition function.
        """    
        num_nodes = len(constraint_subgraph.nodes())
        constraint_subgraph.nodes().sort()
        node_ids = constraint_subgraph.nodes()
        edges = constraint_subgraph.edges()
    
        unnormalized_term = 1.0
        for i in np.arange(0, num_nodes):
            id1 = node_ids[i]
            row1 = node_ids.index(id1)
            state_mat_row1 = state_mat[row1, :]
            energy_accum = 0.0
            for j in np.arange(0, num_nodes):
                id2 = node_ids[j]
                row2 = node_ids.index(id2)
                state_mat_row2 = state_mat[row2, :]
                if (id1, id2) in edges:
                    energy = \
                      self.compute_energy_(state_mat_row1, state_mat_row2,
                        constraint_subgraph[id1][id2]['constraint'])     
                    energy_accum += energy
                elif (id2, id1) in edges:
                    energy = \
                      self.compute_energy_(state_mat_row1, state_mat_row2,
                        constraint_subgraph[id2][id1]['constraint'])                            
                    energy_accum += energy

                if unnormalized_term > 0.0 and np.exp(-energy_accum) > 0.0:
                    unnormalized_term = np.max([np.nextafter(0, 1), 
                        unnormalized_term*np.exp(-energy_accum)])
                else:
                    unnormalized_term = 0.0

                assert not np.isnan(unnormalized_term), \
                  'unnormalized_term is NaN'

        # Now we need to compute the contribution of the r_(n,k) terms show in
        # Eq. 20
        r_term = 1.0
        for i in np.arange(0, num_nodes):
            col = np.nonzero(state_mat[i, :] == 1.0)[0][0]
            r_term *= self.R_[node_ids[i], col]

        unnormalized_term *= r_term
        
        return unnormalized_term        

    def init_R_mat(self):
        """Initializes 'R_', using the stick-breaking construction. Also
        enforces any longitudinal constraints.
        """
        tmp = np.random.beta(1, self.alpha_, (self.N_, self.K_))
        one_tmp = 1. - tmp

        self.R_ = np.array([np.prod(one_tmp[:, 0:k], 1)*tmp[:, k] \
            for k in np.arange(0, self.K_)]).T

        #self.R_[:, self.K_-1] = 1-np.sum(self.R_[:, 0:self.K_-1], 1)

        num_subgraphs = len(self.constraint_subgraphs_)
    
        for i in xrange(0, num_subgraphs):
            is_longitudinal = True
            for e in self.constraint_subgraphs_[i].edges():
                if self.constraint_subgraphs_[i][e[0]][e[1]]['constraint'] != \
                  'longitudinal':
                  is_longitudinal = False
                  break
            if is_longitudinal:
                node_ids = [n for n in self.constraint_subgraphs_[i].nodes()]
                self.R_[node_ids, :] = self.R_[node_ids[0], :]

    def compute_energy_(self, state_mat_row1, state_mat_row2, constraint):
        """Computes the energy function value represented by 'H' in Eq. 20 of
        'ConstrainedNonparametricGaussianProcessRegression'
    
        Parameters
        ----------
        state_mat_row1 : array, shape( 1, K )
            A selected row from the state matrix. This row indicates the state 
            ie. which Gaussian Process regression curve this instance belongs 
            to). 'K' is the number of elements in the truncated Dirichlet 
            process.
    
        state_mat_row2 : array, shape( 1, K )
            A selected row from the state matrix. This row indicates the state 
            (ie. which Gaussian Process regression curve this instance belongs 
            to). 'K' is the number of elements in the truncated Dirichlet 
            process.
    
        constraints : string
            Takes on 'must_link' or 'cannot_link' to indicate the constraint 
            type between the two instances corresponding to 'state_mat_row1' 
            and 'state_mat_row2'
    
        Returns
        -------
        energy : float
            The computed energy term.
        """    
        # Compute the inner product of the two rows. If the inner product is 
        # 1.0, then the two instances are in the same state, otherwise they are 
        # not.
        inner_product = np.dot(state_mat_row1, state_mat_row2)

        # Get the weight for computation of the energy function
        weight = 10.
    
        energy = 0.0
        if inner_product == 1.0:
            if constraint == 'must_link':
                energy = -weight
        else:
            if constraint == 'cannot_link':
                energy = -weight
            elif constraint == 'longitudinal':
                energy = np.inf
    
        return energy

    def inc_state_mat_(self, state_mat, sig_cols=None):
        """Increment 'state_mat' by one unit.
    
        This function takes as input a matrix, 'state_mat', in which each row 
        has one element equal to 1.0 and every other element equal to 0.0. The
        algorithm proceeds by moving down the rows and moving the row element 
        equal to 1.0 over by one place. If this means "resetting" that element 
        to the first position, then the next row will be considered for the 
        same procedure. This continues until no "resets" occur.
    
        Parameters
        ----------
        state_mat : array, shape ( num_rows, num_cols )
            A matrix in which each row has exactly one element equal to 1.0. The
            other elements are equal to 0.0
    
        sig_cols : list of 'num_rows' arrays shaped ( n_cols ), optional
            Each row can have a different number of entries. Each list entry
            indicates the set of columns (possible states) that the 
            corresponding data instance can take on. If not specified, it's 
            assumed that each row (instance) can take on every possible state. 
            If specified, columns taking on the value 1.0 will be moved to the 
            next allowable column (state)        
        """
        num_rows = state_mat.shape[0]
        num_cols = state_mat.shape[1]
        
        row_inc = 0
        reset = True
        while row_inc < num_rows and reset:
            # For the current row, find the column whose element is 1.0
            col = np.nonzero(state_mat[row_inc,:]==1.0)[0][0]
    
            if sig_cols is not None:
                num_sigs = len(sig_cols[row_inc])
                index = sig_cols[row_inc].index(col)
                if num_sigs - 1 == index:
                    reset = True
                    state_mat[row_inc, col] = 0.0
                    state_mat[row_inc, sig_cols[row_inc][0]] = 1.0
                    row_inc += 1
                else:
                    reset = False
                    state_mat[row_inc, col] = 0.0
                    state_mat[row_inc, sig_cols[row_inc][index+1]] = 1.0
            else:
                if col == num_cols-1:
                    reset = True
                    state_mat[row_inc, col] = 0.0
                    state_mat[row_inc, 0] = 1.0
                    row_inc += 1
                else:
                    reset = False
                    state_mat[row_inc, col] = 0.0
                    state_mat[row_inc, col+1] = 1.0
    
    def get_constraint_subgraphs_(self, constraints):
        """Pulls out the connected subgraphs within the 'constraints' graph and
        puts them into a list for easier access.
    
        Parameters
        ----------
        constraints : networkx graph
            The constraints are encoded in a networkx graph, each node should
            indicate an instance index, and edges indicate constraints. Each
            edge should have a string attribute called 'constraint', which must
            either take a value of 'must_link', 'cannot_link', or
            'longitudinal'. Generally, the graph describing 'constraints' will
            be composed of connected subgraphs. This routine isolates the
            connected subgraphs.
    
        Returns
        -------
        constraint_subgraphs : list of networkx graphs
            Each element in the list is a connected sub-graph of 'constraints'
        """
        tmp = list(nx.connected_components(constraints))
        num_subgraphs = len(tmp)
    
        constraint_subgraphs = []
        for i in np.arange(0, num_subgraphs):
            tmp_graph = nx.subgraph(constraints, tmp[i])
            constraint_subgraphs.append(tmp_graph)
    
        return constraint_subgraphs

    def predict_proba(self, X, Y, constraints=None):
        """Compute the probability that each data instance belongs to each of
        the 'k' clusters. Note that 'X' and 'Y' can be "new" data; that is,
        data that was not necessarily used to train the model.

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

        constraints : networkx graph, optional
            The constraints are encoded in a networkx graph, each node should
            indicate an instance index, and edges indicate constraints. Each
            edge should have a string attribute called 'constraint', which must
            take a value of 'must_link', 'cannot_link', or 'longitudinal'.
            
        Returns
        -------
        R : array, shape ( N, K )
            Each element of this matrix represents the probability that
            instance 'n' belongs to cluster 'k'.         
        """
        # If constraints have been specified, get the connected subgraphs
        constraint_subgraphs = []
        if constraints is not None:
            constraint_subgraphs = \
              self.get_constraint_subgraphs_(constraints)

        R = self.update_z(X, Y, constraint_subgraphs)

        return R
    
    def compute_lower_bound(self):
        """Compute the variational lower bound

        Returns
        -------
        lower_bound : float
            The variational lower bound
        """
        term_2 = 0.5*dot(sum(self.R_, 0), sum(psi(self.lambda_a_) - \
            log(self.lambda_b_), 0))
            
        term_3 = 0.        
        for d in xrange(0, self.D_):
            for k in xrange(0, self.K_):
                term_3 += (self.lambda_a_[d, k]/self.lambda_b_[d, k])*\
                  (dot(dot((self.X_**2), self.w_var_[:, d, k]), 
                         self.R_[:, k]) + \
                    sum((self.X_[:, :, newaxis]*self.X_[:, newaxis, :])*\
                           outer(self.w_mu_[:, d, k], self.w_mu_[:, d, k])*\
                           self.R_[:, k, newaxis, newaxis]) + \
                           dot(self.R_[:, k], self.Y_[:, d]**2) + \
                      -2*dot(self.R_[:, k]*self.Y_[:, d], dot(self.X_[:, :], 
                            self.w_mu_[:, d, k])))
        term_3 *= -0.5

        term_6 = np.sum(np.dot(self.R_, psi(self.v_a_) - \
                               psi(self.v_a_ + self.v_b_)))

        term_7 = 0.
        for k in xrange(1, self.K_):
            term_7 += \
              sum(self.R_[:, k]*sum(psi(self.v_b_[0:k]) - \
                                    psi(self.v_a_[0:k] + self.v_b_[0:k])))

        term_8 = (self.alpha_ - 1.)*sum(psi(self.v_b_) - \
                                        psi(self.v_a_ + self.v_b_))

        term_12 = 0.
        for k in xrange(0, self.K_):
            term_12 += sum(-(0.5/self.w_var0_)*(self.w_var_[:, :, k] + \
                self.w_mu_[:, :, k]**2 - 2*self.w_mu0_*self.w_mu_[:, :, k]))

        term_15 = np.sum(np.dot(self.lambda_a0_ - 1, \
                         (psi(self.lambda_a_) - log(self.lambda_b_))))
        
        term_16 = -np.sum(np.dot(self.lambda_b0_, \
                          (self.lambda_a_/self.lambda_b_)))

        ids = self.R_ > 0.
        term_17 = -np.sum(self.R_[ids]*log(self.R_[ids]))
        
        term_18 = 0.        
        for k in xrange(0, self.K_):
            alpha = 1 + np.sum(self.R_, 0)[k]
            beta = self.alpha_ + np.sum(np.sum(self.R_, 0)[(k+1):self.K_])
            term_18 += -gammaln(alpha + beta) + gammaln(alpha) + \
              gammaln(beta) - (alpha - 1)*(psi(alpha) - psi(alpha + beta)) - \
              (beta - 1)*(psi(beta) - psi(alpha + beta))
        
        term_19 = 0.5*np.sum(log(self.w_var_))
        
        term_20 = np.sum(self.lambda_a_ - log(self.lambda_b_) + \
            gammaln(self.lambda_a_) + (1 - self.lambda_a_)*psi(self.lambda_a_))

        lower_bound = term_2 + term_3 + term_6 + term_7 + term_8 + term_12 + \
          term_15 + term_16 + term_17 + term_18 + term_19 + term_20

        self.lower_bounds_.append(lower_bound)
        return lower_bound

    def to_df(self, ref_target_name=None, preserve_col=False):
        """Converts the current model to a pandas data frame, with columns
        indicating trajectory membership.

        Parameters
        ----------
        ref_target_name : str, optional
            Name of the target variable to be used as a reference for trajetory
            ordering. The trajectory with the lowest average (reference) target
            value will be assigned a value of 0, the next lowest average value
            will be assigned a value of 1, etc. If none specified, the first
            target variable will be used.        

        preserve_col : boolean
            If true, the column names in the returned data frame correpsonding
            to the trajectory names will refer to the columns of the R matrix
            from which they are derived. This can be useful when you want to
            refer back to the model from the data frame.
            
        Returns
        -------
        df : Pandas dataframe
            Additional columns 'traj', 'traj_0', 'traj_1', ... . 'traj'
            contains integer values indicating which trajectory the data
            instance belongs to. 'traj_*' contain actual probabilities that the
            data instance belongs to a particular trajectory.    
        """
        tmp_dict = {}
        for i, n in enumerate(self.predictor_names_):
            tmp_dict[n] = pd.Series(self.X_[:, i])

        for i, n in enumerate(self.target_names_):
            tmp_dict[n] = pd.Series(self.Y_[:, i])

        df = pd.DataFrame(tmp_dict)

        df = df.assign(data_names = pd.Series(self.data_names_,
                                               index=df.index))

        cluster_ids = np.max(self.R_, 0) > self.prob_thresh_
        n_clusters = np.sum(cluster_ids)

        traj = []
        traj_cols = []
        for i in xrange(0, self.R_.shape[0]):
            traj.append(where(max(self.R_[i, cluster_ids]) == \
                                self.R_[i, cluster_ids])[0][0])
            traj_cols.append(where(max(self.R_[i, :]) == \
                                self.R_[i, :])[0][0])
                                
        col_to_traj = {}
        traj_to_col = {}
    
        j = 0
        if ref_target_name is not None:
            j = self.target_names_.index(ref_target_name)
        
        tmp = np.argsort(np.sum((self.R_[:, cluster_ids].T*self.Y_[:, j]), 1)/\
            np.sum(self.R_[:, cluster_ids], 0))
        for i in xrange(0, n_clusters):
            col_to_traj[tmp[i]] = i
            traj_to_col[i] = tmp[i]
            
        traj_sort = np.ones(len(traj))
        for i in xrange(0, n_clusters):
            traj_sort[np.array(traj) == i] = 1 + col_to_traj[i]

        if preserve_col:
            df = df.assign(traj = pd.Series(np.array(traj_cols), index=df.index))
        else:
            df = df.assign(traj = pd.Series(traj_sort, index=df.index))
        
        for i in xrange(0, n_clusters):
            df = df.assign(tmp_col_name = \
                pd.Series(self.R_[:, cluster_ids][:, traj_to_col[i]],
                          index=df.index))
            if preserve_col:
                df.rename(columns={'tmp_col_name' : 'traj_' + \
                    str(np.where(cluster_ids)[0][i])}, inplace=True)
            else:
                df.rename(columns={'tmp_col_name' : 'traj_' + str(i+1)},
                        inplace=True)
            
        return df        
        
if __name__ == "__main__":        
    desc = """Run the multiple Dirichlet Process regression algorithm"""
        
    parser = ArgumentParser(description=desc)
    parser.add_argument('--data_file', help='csv file containing the data on \
                        which to run the algorithm', dest='data_file',
                        metavar='<string>', default=None)
    parser.add_argument('--out_file', help='Pickle file name for output data.',
                        dest='out_file', metavar='<string>', default=None)    
    parser.add_argument('--preds', help='Comma-separated list of predictors \
                        to use. Each predictor must correspond to a column in \
                        the data file', dest='preds', metavar='<string>', 
                        default=None)
    parser.add_argument('--targets', help='Comma-separated list of target \
                        variables: each must correspond to a column in \
                        the data file', dest='targets', metavar='<string>', 
                        default=None)
    parser.add_argument('--sid_col', help='Data file column name \
                        corresponding to subject IDs. Repeated entries in \
                        this column are interpreted as corresponding to \
                        different visits for a given subject. Use of this \
                        flag is only necessary if the user wants to impose \
                        longitudinal constraints on the algorithm \
                        (recommended for best performance). If this flag is \
                        not specified, longitudinal constraints will not be \
                        used.', dest='sid_col', metavar='<string>', 
                        default=None)
    parser.add_argument('-K', help='Integer indicating the number of elements \
                        in the truncated Dirichlet Process.', dest='K',
                        metavar='<int>', default=20)
    parser.add_argument('--alpha', help='Hyper parameter of the Beta \
                        destribution involved in the stick-breaking \
                        construction of the Dirichlet Process. Increasing \
                        alpha tends to produce more clusters and vice-versa.',
                        dest='alpha', metavar='<int>', default=20)
    parser.add_argument('--iters', help='Number of variational inference \
                        iterations to run.', dest='iters', metavar='<int>',
                        default=10000)
    parser.add_argument('--names_col', help='Data file column corresponding \
                        names of data points. Entries in this columns must \
                        be unique (optional).', dest='names_col',
                        metavar='<int>', default=None)
    parser.add_argument('--w_mu0_file', help='The coefficient for each \
                        predictor is drawn from a normal distribution. \
                        This is the file name of the M x D matrix of \
                        hyperparameters of the mean values of those \
                        normal distributions. M is the dimension of the \
                        predictor space, and D is the dimension of the \
                        target variable space. Values should be separated by \
                        commas, and the ordering of the rows and columns \
                        must be the same as the ordering of predictors \
                        and targets specified on the command line.', 
                        dest='w_mu0_file', metavar='<string>', default=None)
    parser.add_argument('--w_var0_file', help='The coefficient for each \
                        predictor is drawn from a normal distribution. \
                        This is the file name of the M x D matrix of \
                        hyperparameters of the variance values of those \
                        normal distributions. M is the dimension of the \
                        predictor space, and D is the dimension of the \
                        target variable space. Values should be separated by \
                        commas, and the ordering of the rows and columns \
                        must be the same as the ordering of predictors \
                        and targets specified on the command line.', 
                        dest='w_var0_file', metavar='<string>', default=None)
    parser.add_argument('--lambdas0_file', help='File containing the 2 x D \
                        dimensional array of hyperparameters for the Gamma \
                        priors over the precisions for each of the d target \
                        variables. The first row in the file should \
                        correspond to the first parameter of the \
                        distribution, and the second row of the file should \
                        correspond to the second parameter of the \
                        distribution. Values should be comma-separated, and \
                        ordering of the columns must be the same as the \
                        ordering of targets specified on the command line.', 
                        dest='lambdas0_file', metavar='<string>', default=None)
    parser.add_argument('--R_file', help='Pickle file containing NxK values \
                        using the "R" key. These values will be starting point \
                        trajectory assignment probabilities for the fitting \
                        process. If specified, the value of K will be set to \
                        the number of columns in this matrix, and any value \
                        set with the -K flag will be ignored. If there is a \
                        mismatch between the number of rows in this matrix and \
                        the number of data points in the data_file, a value \
                        error will be raised. If any row in this matrix \
                        contains a nan, the entire row will be replaced with \
                        a normalized, K-dimensional random vector.',
                        dest='R_file', metavar='<string>', default=None)
    parser.add_argument('--v_a_file', help='Pickle file containing K values \
                        using the "v_a" key. These are the first values of \
                        the posterior Beta distribution describing the \
                        latent vector, v, which is involved in the stick-\
                        breaking construction of the DP. If this file is \
                        specified, you must also specify the --v_b_file. \
                        The fitting process will begin with these values.',
                        dest='v_a_file', metavar='<string>', default=None)
    parser.add_argument('--v_b_file', help='Pickle file containing K values \
                        using the "v_b" key. These are the second values of \
                        the posterior Beta distribution describing the \
                        latent vector, v, which is involved in the stick-\
                        breaking construction of the DP. If this file is \
                        specified, you must also specify the --v_a_file. \
                        The fitting process will begin with these values.',
                        dest='v_b_file', metavar='<string>', default=None)    
    parser.add_argument('--w_mu_file', help='Pickle file containing MxDxK \
                        values using the "w_mu" key. M is the dimension of \
                        the predictor space, D is the dimension of the target \
                        space, and K is the number of elements in the \
                        truncated Dirichlet process. These values will be \
                        used as the fitting starting point for the predictor \
                        coefficients.', dest='w_mu_file', metavar='<string>',
                        default=None)
    parser.add_argument('--w_var_file', help='Pickle file containing MxDxK \
                        values using the "w_var" key. M is the dimension of \
                        the predictor space, D is the dimension of the target \
                        space, and K is the number of elements in the \
                        truncated Dirichlet process. These values will be \
                        used as the fitting starting point for the variance \
                        values of the distributions over the predictor \
                        coefficients.', dest='w_var_file', metavar='<string>',
                        default=None)           
    parser.add_argument('--lambda_a_file', help='Pickle file containing D x K \
                        values using the key "lambda_a". D is the dimension \
                        of the target space and K is the number of elements \
                        in the truncated Dirichlet process. These values will \
                        be used as the fitting starting point for the first \
                        parameter of the Gamma distributions describing the \
                        precision of the target variables. If this file is \
                        specified, you should also specify --lambda_a_file.',
                        dest='lambda_a_file', metavar='<string>', default=None)
    parser.add_argument('--lambda_b_file', help='Pickle file containing D x K \
                        values using the key "lambda_b". D is the dimension \
                        of the target space and K is the number of elements \
                        in the truncated Dirichlet process. These values will \
                        be used as the fitting starting point for the second \
                        parameter of the Gamma distributions describing the \
                        precision of the target variables. If this file is \
                        specified, you should also specify --lambda_a_file.',
                        dest='lambda_b_file', metavar='<string>', default=None)        
    
    op = parser.parse_args()

    if op.preds is None:
        raise ValueError('Must specify at least one predictor')
    if op.targets is None:
        raise ValueError('Must specify at least one target variable')
    if op.data_file is None:
        raise ValueError('Must specify a data file')
    
    preds = op.preds.split(',')
    targets = op.targets.split(',')
    
    df = pd.read_csv(op.data_file).dropna(how='any', subset=preds+targets,
                                          inplace=False)

    data_names = None
    if op.names_col is not None:
        data_names = df[op.names_col].values

    K = int(op.K)        
    X = df[preds].values
    Y = df[targets].values
    R = None
    if op.R_file is not None:
        R = pickle.load(open(op.R_file, 'rb'))['R']

        if R.shape[0] != df.shape[0]:
            raise ValueError("Size mismatch between prior and data matrix")
        replacement_rows = np.where(np.isnan(np.sum(R, 1)))[0]

        for i in replacement_rows:
            vec = np.random.rand(K)
            R[i, :] = vec/np.sum(vec)

    alpha = float(op.alpha)
    
    graph = None
    if op.sid_col is not None:
        graph = get_longitudinal_constraints_graph(df[op.sid_col].values)

    w_mu0 = pd.read_csv(op.w_mu0_file, header=None).values.T
    w_var0 = pd.read_csv(op.w_var0_file, header=None).values.T
    lambdas0 = pd.read_csv(op.lambdas0_file, header=None).values
    lambda_a0 = lambdas0[0, :]
    lambda_b0 = lambdas0[1, :]    

    v_a = v_b = w_mu = w_var = lambda_a = lambda_b = None

    if op.v_a_file is not None:    
        v_a = pickle.load(open(op.v_a_file, 'rb'))['v_a']
    if op.v_b_file is not None:            
        v_b = pickle.load(open(op.v_b_file, 'rb'))['v_b']
    if op.w_mu_file is not None:            
        w_mu = pickle.load(open(op.w_mu_file, 'rb'))['w_mu']
    if op.w_var_file is not None:            
        w_var = pickle.load(open(op.w_var_file, 'rb'))['w_var']
    if op.lambda_a_file is not None:            
        lambda_a = pickle.load(open(op.lambda_a_file, 'rb'))['lambda_a']
    if op.lambda_b_file is not None:            
        lambda_b = pickle.load(open(op.lambda_b_file, 'rb'))['lambda_b']
    
    mm = MultDPRegression(w_mu0, w_var0, lambda_a0, lambda_b0, alpha,
                          K=K)
    mm.fit(X, Y, R=R, v_a=v_a, v_b=v_b, w_mu=w_mu, w_var=w_var,
        lambda_a=lambda_a, lambda_b=lambda_b, iters=int(op.iters), tol=0.01,
        constraints=graph, data_names=data_names, target_names=targets,
        predictor_names=preds)
    
    pickle.dump({'mm': mm}, open(op.out_file, "wb"))

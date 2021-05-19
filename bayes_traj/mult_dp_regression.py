from argparse import ArgumentParser
import matplotlib.pyplot as plt
import numpy as np
from numpy import abs, dot, mean, log, sum, exp, tile, max, sum, isnan, diag, \
     sqrt, pi, newaxis, outer, genfromtxt, where
from numpy.random import multivariate_normal, randn, gamma
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
            batch_size=None, verbose=False):
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
            be initialized with this matrix.

        lambda_b : array, shape ( D, K ), optional
            For component 'K' and target dimension 'D', this is the second
            parameter of the posterior Gamma distribution describing the
            precision of the target variable. If specified, the algorithm will
            be initialized with this matrix.

        batch_size : int, optional
            The size (number of individuals) of the minibatch to use for 
            stochastic variational inference. It must be less than the total 
            number of individuals.

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

        self.init_traj_params()

        if self.v_a_ is None:
            self.v_a_ = np.ones(self.K_)

        if self.v_b_ is None:
            self.v_b_ = self.alpha_*np.ones(self.K_)

        # Initialize the latent variables if needed
        if self.R_ is None:
            self.init_R_mat(traj_probs, traj_probs_weight)

        if batch_size is not None:
            self.fit_svi(batch_size, iters, verbose)
        else:
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
            self.update_w_accel()
            self.update_lambda_accel() 
            self.R_ = self.update_z_accel(self.X_, self.Y_)            
                
            self.sig_trajs_ = np.max(self.R_, 0) > self.prob_thresh_

            if verbose:
                print("iter {},  {}".format(inc, sum(self.R_, 0)))

    def fit_svi(self, batch_size, iters, verbose):
        """This function contains the iteratrion loop for stochastic variational 
       inference.

        Parameters
        ----------
        batch_size : int, optional
            The size (number of individuals) of the minibatch to use for 
            stochastic variational inference. It must be less than the total 
            number of individuals.

        iters : int, optional
            Number of variational inference iterations to run.

        verbose : bool, optional
            If true, a printout of the sum along rows of the R_ matrix will
            be provided during optimization. This sum indicates how many data
            instances are being assigned to each of the K possible
            trajectories.

        References
        ----------
        Hoffman MD, Blei DM, Wang C, Paisley J. Stochastic variational 
        inference. Journal of Machine Learning Research. 2013 May 1;14(5).
        """            
        # Batch size taken to mean number of individuals.
        if self.gb_ is not None:
            assert batch_size > 0 & batch_size <= self.gb_.ngroups, \
                "Batch size misspecified"
        else:
            assert batch_size > 0 & batch_size <= self.N_, \
                "Batch size misspecified"
        self.batch_size_ = batch_size

        if self.gb_ is not None:
            assert self.batch_size_ <= self.gb_.ngroups and \
                self.batch_size_ >= 1, "Batch size error"
            indicator = np.zeros(self.gb_.ngroups)
        else:
            assert self.batch_size_ <= self.N_ and \
                self.batch_size_ >= 1, "Batch size error"            
            indicator = np.zeros(self.N_)
        indicator[0:self.batch_size_] = 1
                
        # tau and kappa set to reasonable values as investigated in 
        tau = 1.
        kappa = 0.9
        inc =  0
        while inc < iters:
            inc += 1
            rho = (inc + tau)**(-kappa)
            
            # Get batch indices. If the data has been grouped, retrieve the data
            # indices corresponding to the randomly selected individuals
            self.batch_indices_ = np.zeros(self.N_, dtype=bool)
            if self.gb_ is not None:
                self.batch_subjects_ = np.array(list(self.gb_.groups.keys()))\
                    [np.where(np.random.permutation(indicator))[0]]

                for ww in self.batch_subjects_:
                    self.batch_indices_[self.gb_.get_group(ww).index] = True
            else:
                self.batch_indices_ = np.random.permutation(indicator) == 1
            
            v_a, v_b = self.update_v_batch()             
            self.v_a_[self.sig_trajs_] = \
                (1 - rho)*self.v_a_[self.sig_trajs_] + rho*v_a[self.sig_trajs_]
            self.v_b_[self.sig_trajs_] = \
                (1 - rho)*self.v_b_[self.sig_trajs_] + rho*v_b[self.sig_trajs_]

            w_mu, w_var = self.update_w_batch(self.lambda_a_, self.lambda_b_)             
            self.w_mu_[:, :, self.sig_trajs_] = \
                (1 - rho)*self.w_mu_[:, :, self.sig_trajs_] + \
                rho*w_mu[:, :, self.sig_trajs_]
            self.w_var_[:, :, self.sig_trajs_] = \
                1./((1 - rho)*1./self.w_var_[:, :, self.sig_trajs_] + \
                    rho*1./w_var[:, :, self.sig_trajs_])

            lambda_a, lambda_b = \
                self.update_lambda_batch(self.w_mu_, self.w_var_)            
            self.lambda_a_[:, self.sig_trajs_] = \
                (1 - rho)*self.lambda_a_[:, self.sig_trajs_] + \
                rho*lambda_a[:, self.sig_trajs_]
            self.lambda_b_[:, self.sig_trajs_] = \
                (1 - rho)*self.lambda_b_[:, self.sig_trajs_] + \
                rho*lambda_b[:, self.sig_trajs_]

            self.R_[self.batch_indices_, :] = \
                self.update_z_batch(self.X_, self.Y_)                
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

    def update_v_batch(self):
        """Updates the parameters of the Beta distributions for latent
        variable 'v' in the variational approximation.
        """
        v_a = np.zeros(self.K_)
        v_b = np.zeros(self.K_)        
        
        sum_factor = self.N_/np.sum(self.batch_indices_)
        
        v_a = 1.0 + sum_factor*np.sum(self.R_[self.batch_indices_, :], 0)

        for k in np.arange(0, self.K_):
            v_b[k] = self.alpha_ + \
                sum_factor*np.sum(self.R_[self.batch_indices_, k+1:])

        return v_a, v_b
                                 
    def update_z_batch(self, X, Y):
        """
        """
        expec_ln_v = psi(self.v_a_) - psi(self.v_a_ + self.v_b_)
        expec_ln_1_minus_v = psi(self.v_b_) - psi(self.v_a_ + self.v_b_)

        tmp = np.array(expec_ln_v)
        for k in range(1, self.K_):
            tmp[k] += np.sum(expec_ln_1_minus_v[0:k])

        ln_rho = np.ones([self.N_, self.K_])*tmp[newaxis, :]

        for d in range(0, self.D_):
            sel_ids = ~np.isnan(Y[:, d]) & self.batch_indices_

            tmp = (dot(self.w_mu_[:, d, :].T, \
                X[sel_ids, :].T)**2).T + \
                np.sum((X[sel_ids, newaxis, :]**2)*\
                (self.w_var_[:, d, :].T)[newaxis, :, :], 2)

            ln_rho[sel_ids, :] += \
              0.5*(psi(self.lambda_a_[d, :]) - \
                log(self.lambda_b_[d, :]) - \
                log(2*np.pi) - \
                (self.lambda_a_[d, :]/self.lambda_b_[d, :])*\
                (tmp - \
                 2*Y[sel_ids, d, newaxis]*dot(X[sel_ids, :], \
                self.w_mu_[:, d, :]) + \
                Y[sel_ids, d, newaxis]**2))

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
            for g in self.batch_subjects_:
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

        return R[self.batch_indices_, :]

    def update_z_accel(self, X, Y):
        """
        """
        expec_ln_v = psi(self.v_a_) - psi(self.v_a_ + self.v_b_)
        expec_ln_1_minus_v = psi(self.v_b_) - psi(self.v_a_ + self.v_b_)

        tmp = np.array(expec_ln_v)
        for k in range(1, self.K_):
            tmp[k] += np.sum(expec_ln_1_minus_v[0:k])

        ln_rho = np.ones([self.N_, self.K_])*tmp[newaxis, :]

        for d in range(0, self.D_):
            non_nan_ids = ~np.isnan(Y[:, d])

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
    
    def update_w_accel(self):
        """ Updates the variational distribution over latent variable w.
        """
        mu0_DIV_var0 = self.w_mu0_/self.w_var0_
        for m in range(0, self.M_):
            ids = np.ones(self.M_, dtype=bool)
            ids[m] = False
            for d in range(0, self.D_):
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

    def update_w_batch(self, lambda_a, lambda_b):
        """ Updates the variational distribution over latent variable w.
        """
        w_mu = np.array(self.w_mu_)
        w_var = np.array(self.w_var_)
        
        mu0_DIV_var0 = self.w_mu0_/self.w_var0_
        for m in range(0, self.M_):
            ids = np.ones(self.M_, dtype=bool)
            ids[m] = False
            for d in range(0, self.D_):
                sel_ids = ~np.isnan(self.Y_[:, d]) & self.batch_indices_
                sum_factor = self.N_/np.sum(sel_ids)
                
                tmp1 = sum_factor*(lambda_a[d, self.sig_trajs_]/\
                                lambda_b[d, self.sig_trajs_])*\
                                (np.sum(self.R_[:, self.sig_trajs_, newaxis]\
                                        [sel_ids, :, :]*\
                                        self.X_[sel_ids, newaxis, :]**2, 0).T)\
                                        [:, newaxis, :]
                
                w_var[:, :, self.sig_trajs_] = \
                    (tmp1 + (1.0/self.w_var0_)[:, :, newaxis])**-1
                
                sum_term = \
                    sum_factor*sum(self.R_[sel_ids, :][:, self.sig_trajs_]*\
                                   self.X_[sel_ids, m, newaxis]*\
                                   (dot(self.X_[:, ids][sel_ids, :], \
                                        w_mu[ids, d, :][:, self.sig_trajs_]) - \
                                    self.Y_[sel_ids, d][:, newaxis]), 0)
    
                w_mu[m, d, self.sig_trajs_] = \
                    w_var[m, d, self.sig_trajs_]*\
                    (-(lambda_a[d, self.sig_trajs_]/\
                       lambda_b[d, self.sig_trajs_])*\
                     sum_term + mu0_DIV_var0[m, d])
                
        return w_mu, w_var    
                
    def update_lambda_accel(self):
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

    def update_lambda_batch(self, w_mu, w_var):
        """Updates the variational distribution over latent variable lambda.
        """
        lambda_a = np.nan*np.zeros([self.D_, self.K_])
        lambda_b = np.nan*np.zeros([self.D_, self.K_])
        
        for d in range(self.D_):
            sel_ids = ~np.isnan(self.Y_[:, d]) & self.batch_indices_
            sum_factor = self.N_/np.sum(sel_ids)

            lambda_a[d, self.sig_trajs_] = self.lambda_a0_[d, newaxis] + \
                sum_factor*0.5*sum(self.R_[:, self.sig_trajs_][sel_ids, :], 0)\
                [newaxis, :]
            
            tmp = (dot(w_mu[:, d, self.sig_trajs_].T, \
                       self.X_[sel_ids, :].T)**2).T + \
                       np.sum((self.X_[sel_ids, newaxis, :]**2)*\
                              (w_var[:, d, self.sig_trajs_].T)\
                              [newaxis, :, :], 2)
    
            lambda_b[d, self.sig_trajs_] = \
                self.lambda_b0_[d, newaxis] + \
                0.5*sum_factor*sum(self.R_[sel_ids, :][:, self.sig_trajs_]*\
                                   (tmp - 2*self.Y_[sel_ids, d, newaxis]*\
                                    np.dot(self.X_[sel_ids, :], \
                                           w_mu[:, d, self.sig_trajs_]) + \
                                    self.Y_[sel_ids, d, newaxis]**2), 0)
            
        return lambda_a, lambda_b
            
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
        for s in range(0, S):  
            z = np.random.multinomial(1, self.R_[index, :])
            for d in range(0, self.D_):
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

    def log_likelihood(self):
        """Compute the log-likelihood given expected values of the latent 
        variables.

        Returns
        -------
        log_likelihood : float
            The log-likelihood
        """
        tmp_k = np.zeros(self.N_)
        for k in range(self.K_):
            tmp_d = np.ones(self.N_)
            for d in range(self.D_):
                mu = np.dot(self.X_, self.w_mu_[:, d, k])
                v = self.lambda_b_[d, k]/self.lambda_a_[d, k]
                co = 1/sqrt(2.*np.pi*v)

                # Some of the target variables can be missing (NaNs). Exclude
                # these from the computation.
                ids = ~np.isnan(self.Y_[:, d])
                tmp_d[ids] *= co*np.exp(-((self.Y_[ids, d]-mu[ids])**2)/(2.*v))
            
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
                co = multivariate_normal(self.w_mu_[:, d, k],
                    diag(self.w_var_[:, d, k]), S)
                mu = dot(co, self.X_.T)

                # Draw a precision value from the gamma distribution. Note
                # that numpy uses a slightly different parameterization
                scale = 1./self.lambda_b_[d, k]
                shape = self.lambda_a_[d, k]
                var = 1./gamma(shape, scale, size=1)

                prob = ((1/sqrt(2*np.pi*var))[:, newaxis]*\
                    exp(-(1/(2*var))[:, newaxis]*(mu - self.Y_[:, d])**2)).T

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
        """
        tmp_K = w_mu.shape[2]
        R_tmp = np.ones([self.N_, tmp_K])
        var = self.lambda_b0_/self.lambda_a0_
        for k in range(tmp_K):
            for d in range(self.D_):
                ids = ~np.isnan(self.Y_[:, d])
                
                mu_tmp = np.dot(self.X_, w_mu[:, d, k])
    
                R_tmp[ids, k] *= (1/(np.sqrt(2*np.pi*var[d])))*\
                    np.exp(-((self.Y_[ids, d] - mu_tmp[ids])**2)/(2*var[d]))
    
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
        if self.w_var_ is None:
            self.w_var_ = np.zeros([self.M_, self.D_, self.K_])
            for k in range(self.K_):
                self.w_var_[:, :, k] = np.array(self.w_var0_)
            
        if self.lambda_a_ is None:
            self.lambda_a_ = np.ones([self.D_, self.K_])
            for k in range(self.K_):
                self.lambda_a_[:, k] = np.array(self.lambda_a0_)
            
        if self.lambda_b_ is None:
            self.lambda_b_ = np.ones([self.D_, self.K_])
            for k in range(self.K_):
                self.lambda_b_[:, k] = np.array(self.lambda_b0_)            
            
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
                var_tmp = self.lambda_b_[d, k]/self.lambda_a_[d, k]
                mu_tmp = np.dot(X, self.w_mu_[:, d, k])

                R_tmp[ids, k] *= (1/(np.sqrt(2*np.pi*var_tmp)))*\
                    np.exp(-((Y[ids, d] - mu_tmp[ids])**2)/(2*var_tmp))
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
                var_tmp = self.lambda_b_[d, k]/self.lambda_a_[d, k]
                mu_tmp = np.dot(X, self.w_mu_[:, d, k])

                R_tmp[ids, k] *= (1/(np.sqrt(2*np.pi*var_tmp)))*\
                    np.exp(-((Y[ids, d] - mu_tmp[ids])**2)/(2*var_tmp))
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

        # Now augment the data frame with trajectory info
        traj = []
        for i in range(N):
            traj.append(np.where(np.max(R[i, :]) == R[i, :])[0][0])
        df_['traj'] = traj

        for s in np.where(self.sig_trajs_)[0]:
            df_['traj_{}'.format(s)] = R[:, s]

        return df_
            
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
        for d in range(0, self.D_):
            non_nan_ids = ~np.isnan(self.Y_[:, d])
            for k in range(0, self.K_):
                term_3 += (self.lambda_a_[d, k]/self.lambda_b_[d, k])*\
                  (dot(dot((self.X_[non_nan_ids, :]**2), self.w_var_[:, d, k]),
                         self.R_[non_nan_ids, k]) + \
                    sum((self.X_[non_nan_ids, :, newaxis]*\
                         self.X_[non_nan_ids, newaxis, :])*\
                           outer(self.w_mu_[:, d, k], self.w_mu_[:, d, k])*\
                           self.R_[non_nan_ids, k, newaxis, newaxis]) + \
                           dot(self.R_[non_nan_ids, k], \
                               self.Y_[non_nan_ids, d]**2) + \
                      -2*dot(self.R_[non_nan_ids, k]*self.Y_[non_nan_ids, d], \
                             dot(self.X_[non_nan_ids, :], 
                            self.w_mu_[:, d, k])))
        term_3 *= -0.5

        term_6 = np.sum(np.dot(self.R_, psi(self.v_a_) - \
                               psi(self.v_a_ + self.v_b_)))

        term_7 = 0.
        for k in range(1, self.K_):
            term_7 += \
              sum(self.R_[:, k]*sum(psi(self.v_b_[0:k]) - \
                                    psi(self.v_a_[0:k] + self.v_b_[0:k])))

        term_8 = (self.alpha_ - 1.)*sum(psi(self.v_b_) - \
                                        psi(self.v_a_ + self.v_b_))

        term_12 = 0.
        for k in range(0, self.K_):
            term_12 += sum(-(0.5/self.w_var0_)*(self.w_var_[:, :, k] + \
                self.w_mu_[:, :, k]**2 - 2*self.w_mu0_*self.w_mu_[:, :, k]))

        term_15 = np.sum(np.dot(self.lambda_a0_ - 1, \
                         (psi(self.lambda_a_) - log(self.lambda_b_))))

        term_16 = -np.sum(np.dot(self.lambda_b0_, \
                          (self.lambda_a_/self.lambda_b_)))

        ids = self.R_ > 0.
        term_17 = -np.sum(self.R_[ids]*log(self.R_[ids]))

        term_18 = 0.
        for k in range(0, self.K_):
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
            traj_prob_vec = np.sum(mm.R_, 0)/np.sum(mm.R_)
        
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
    
        if traj_ids.shape[0] <= 10:
            cmap = plt.cm.get_cmap('tab10')
        else:
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

                std = np.sqrt(self.lambda_b_[target_index][tt]/\
                              self.lambda_a_[target_index][tt])
            
                co = self.w_mu_[:, target_index, tt]
                y_tmp = np.dot(co, X_tmp.T)

                n_traj = int(traj_prob_vec[tt]*num_individuals)
                perc_traj = traj_prob_vec[tt]*100
                ax.plot(x_dom, y_tmp,
                        color=cmap(traj_id_to_cmap_index[traj_map_[tt]]),
                        linewidth=3,
                        label='Traj {} (N={}, {:.1f}%)'.\
                        format(traj_map_[tt], n_traj, perc_traj))
                ax.fill_between(x_dom, y_tmp-2*std, y_tmp+2*std,
                        color=cmap(traj_id_to_cmap_index[traj_map_[tt]]),
                        alpha=0.3)

        ax.set_xlabel(x_axis, fontsize=16)
        ax.set_ylabel(y_axis, fontsize=16)    
        plt.tight_layout()
        ax.legend()

        if show:
            plt.show()

        return ax
    
#if __name__ == "__main__":
#    desc = """Run the multiple Dirichlet Process regression algorithm"""
#
#    parser = ArgumentParser(description=desc)
#    parser.add_argument('--data_file', help='csv file containing the data on \
#                        which to run the algorithm', dest='data_file',
#                        metavar='<string>', default=None)
#    parser.add_argument('--out_file', help='Pickle file name for output data.',
#                        dest='out_file', metavar='<string>', default=None)
#    parser.add_argument('--preds', help='Comma-separated list of predictors \
#                        to use. Each predictor must correspond to a column in \
#                        the data file', dest='preds', metavar='<string>',
#                        default=None)
#    parser.add_argument('--targets', help='Comma-separated list of target \
#                        variables: each must correspond to a column in \
#                        the data file', dest='targets', metavar='<string>',
#                        default=None)
#    parser.add_argument('--sid_col', help='Data file column name \
#                        corresponding to subject IDs. Repeated entries in \
#                        this column are interpreted as corresponding to \
#                        different visits for a given subject. Use of this \
#                        flag is only necessary if the user wants to impose \
#                        longitudinal constraints on the algorithm \
#                        (recommended for best performance). If this flag is \
#                        not specified, longitudinal constraints will not be \
#                        used.', dest='sid_col', metavar='<string>',
#                        default=None)
#    parser.add_argument('-K', help='Integer indicating the number of elements \
#                        in the truncated Dirichlet Process.', dest='K',
#                        metavar='<int>', default=20)
#    parser.add_argument('--alpha', help='Hyper parameter of the Beta \
#                        destribution involved in the stick-breaking \
#                        construction of the Dirichlet Process. Increasing \
#                        alpha tends to produce more clusters and vice-versa.',
#                        dest='alpha', metavar='<int>', default=20)
#    parser.add_argument('--iters', help='Number of variational inference \
#                        iterations to run.', dest='iters', metavar='<int>',
#                        default=10000)
#    parser.add_argument('--w_mu0_file', help='The coefficient for each \
#                        predictor is drawn from a normal distribution. \
#                        This is the file name of the M x D matrix of \
#                        hyperparameters of the mean values of those \
#                        normal distributions. M is the dimension of the \
#                        predictor space, and D is the dimension of the \
#                        target variable space. Values should be separated by \
#                        commas, and the ordering of the rows and columns \
#                        must be the same as the ordering of predictors \
#                        and targets specified on the command line.',
#                        dest='w_mu0_file', metavar='<string>', default=None)
#    parser.add_argument('--w_var0_file', help='The coefficient for each \
#                        predictor is drawn from a normal distribution. \
#                        This is the file name of the M x D matrix of \
#                        hyperparameters of the variance values of those \
#                        normal distributions. M is the dimension of the \
#                        predictor space, and D is the dimension of the \
#                        target variable space. Values should be separated by \
#                        commas, and the ordering of the rows and columns \
#                        must be the same as the ordering of predictors \
#                        and targets specified on the command line.',
#                        dest='w_var0_file', metavar='<string>', default=None)
#    parser.add_argument('--lambdas0_file', help='File containing the 2 x D \
#                        dimensional array of hyperparameters for the Gamma \
#                        priors over the precisions for each of the d target \
#                        variables. The first row in the file should \
#                        correspond to the first parameter of the \
#                        distribution, and the second row of the file should \
#                        correspond to the second parameter of the \
#                        distribution. Values should be comma-separated, and \
#                        ordering of the columns must be the same as the \
#                        ordering of targets specified on the command line.', 
#                        dest='lambdas0_file', metavar='<string>', default=None)
#    parser.add_argument('--R_file', help='Pickle file containing NxK values \
#                        using the "R" key. These values will be starting point \
#                        trajectory assignment probabilities for the fitting \
#                        process. If specified, the value of K will be set to \
#                        the number of columns in this matrix, and any value \
#                        set with the -K flag will be ignored. If there is a \
#                        mismatch between the number of rows in this matrix and \
#                        the number of data points in the data_file, a value \
#                        error will be raised. If any row in this matrix \
#                        contains a nan, the entire row will be replaced with \
#                        a normalized, K-dimensional random vector.',
#                        dest='R_file', metavar='<string>', default=None)
#    parser.add_argument('--v_a_file', help='Pickle file containing K values \
#                        using the "v_a" key. These are the first values of \
#                        the posterior Beta distribution describing the \
#                        latent vector, v, which is involved in the stick-\
#                        breaking construction of the DP. If this file is \
#                        specified, you must also specify the --v_b_file. \
#                        The fitting process will begin with these values.',
#                        dest='v_a_file', metavar='<string>', default=None)
#    parser.add_argument('--v_b_file', help='Pickle file containing K values \
#                        using the "v_b" key. These are the second values of \
#                        the posterior Beta distribution describing the \
#                        latent vector, v, which is involved in the stick-\
#                        breaking construction of the DP. If this file is \
#                        specified, you must also specify the --v_a_file. \
#                        The fitting process will begin with these values.',
#                        dest='v_b_file', metavar='<string>', default=None)
#    parser.add_argument('--w_mu_file', help='Pickle file containing MxDxK \
#                        values using the "w_mu" key. M is the dimension of \
#                        the predictor space, D is the dimension of the target \
#                        space, and K is the number of elements in the \
#                        truncated Dirichlet process. These values will be \
#                        used as the fitting starting point for the predictor \
#                        coefficients.', dest='w_mu_file', metavar='<string>',
#                        default=None)
#    parser.add_argument('--w_var_file', help='Pickle file containing MxDxK \
#                        values using the "w_var" key. M is the dimension of \
#                        the predictor space, D is the dimension of the target \
#                        space, and K is the number of elements in the \
#                        truncated Dirichlet process. These values will be \
#                        used as the fitting starting point for the variance \
#                        values of the distributions over the predictor \
#                        coefficients.', dest='w_var_file', metavar='<string>',
#                        default=None)
#    parser.add_argument('--lambda_a_file', help='Pickle file containing D x K \
#                        values using the key "lambda_a". D is the dimension \
#                        of the target space and K is the number of elements \
#                        in the truncated Dirichlet process. These values will \
#                        be used as the fitting starting point for the first \
#                        parameter of the Gamma distributions describing the \
#                        precision of the target variables. If this file is \
#                        specified, you should also specify --lambda_a_file.',
#                        dest='lambda_a_file', metavar='<string>', default=None)
#    parser.add_argument('--lambda_b_file', help='Pickle file containing D x K \
#                        values using the key "lambda_b". D is the dimension \
#                        of the target space and K is the number of elements \
#                        in the truncated Dirichlet process. These values will \
#                        be used as the fitting starting point for the second \
#                        parameter of the Gamma distributions describing the \
#                        precision of the target variables. If this file is \
#                        specified, you should also specify --lambda_a_file.',
#                        dest='lambda_b_file', metavar='<string>', default=None)
#
#    op = parser.parse_args()
#
#    if op.preds is None:
#        raise ValueError('Must specify at least one predictor')
#    if op.targets is None:
#        raise ValueError('Must specify at least one target variable')
#    if op.data_file is None:
#        raise ValueError('Must specify a data file')
#
#    preds = op.preds.split(',')
#    targets = op.targets.split(',')
#
#    df = pd.read_csv(op.data_file).dropna(how='any', subset=preds+targets,
#                                          inplace=False)
#
#    K = int(op.K)
#    X = df[preds].values
#    Y = df[targets].values
#    R = None
#    if op.R_file is not None:
#        R = pickle.load(open(op.R_file, 'rb'))['R']
#
#        if R.shape[0] != df.shape[0]:
#            raise ValueError("Size mismatch between prior and data matrix")
#        replacement_rows = np.where(np.isnan(np.sum(R, 1)))[0]
#
#        for i in replacement_rows:
#            vec = np.random.rand(K)
#            R[i, :] = vec/np.sum(vec)
#
#    alpha = float(op.alpha)
#
#    graph = None
#    if op.sid_col is not None:
#        graph = get_longitudinal_constraints_graph(df[op.sid_col].values)
#
#    w_mu0 = pd.read_csv(op.w_mu0_file, header=None).values.T
#    w_var0 = pd.read_csv(op.w_var0_file, header=None).values.T
#    lambdas0 = pd.read_csv(op.lambdas0_file, header=None).values
#    lambda_a0 = lambdas0[0, :]
#    lambda_b0 = lambdas0[1, :]
#
#    v_a = v_b = w_mu = w_var = lambda_a = lambda_b = None
#
#    if op.v_a_file is not None:
#        v_a = pickle.load(open(op.v_a_file, 'rb'))['v_a']
#    if op.v_b_file is not None:
#        v_b = pickle.load(open(op.v_b_file, 'rb'))['v_b']
#    if op.w_mu_file is not None:
#        w_mu = pickle.load(open(op.w_mu_file, 'rb'))['w_mu']
#    if op.w_var_file is not None:
#        w_var = pickle.load(open(op.w_var_file, 'rb'))['w_var']
#    if op.lambda_a_file is not None:
#        lambda_a = pickle.load(open(op.lambda_a_file, 'rb'))['lambda_a']
#    if op.lambda_b_file is not None:
#        lambda_b = pickle.load(open(op.lambda_b_file, 'rb'))['lambda_b']
#
#    mm = MultDPRegression(w_mu0, w_var0, lambda_a0, lambda_b0, alpha,
#                          K=K)
#    mm.fit(target_names=targets, predictor_names=preds, R=R, v_a=v_a, v_b=v_b,
#           w_mu=w_mu, w_var=w_var, lambda_a=lambda_a, lambda_b=lambda_b,
#           iters=int(op.iters), tol=0.01)
#
#    pickle.dump({'mm': mm}, open(op.out_file, "wb"))

import torch
from torch.distributions import Normal, Gamma, Beta, constraints
from torch.distributions import MultivariateNormal, Bernoulli
import pyro
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
import pyro.distributions as dist

from argparse import ArgumentParser
import matplotlib.pyplot as plt
import numpy as np
from numpy import abs, dot, mean, log, sum, exp, tile, max, sum, isnan, diag, \
     sqrt, pi, newaxis, outer, genfromtxt, where
from numpy.random import multivariate_normal, randn, gamma, binomial
from bayes_traj.utils import *
from scipy.optimize import minimize_scalar
from scipy.special import psi, gammaln, logsumexp
from scipy.stats import norm
import pandas as pd
import pdb, sys, pickle, time, warnings
import copy



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
    v_a_ : torch.Tensor, shape ( K )
        For each of the 'K' elements in the truncated DP, this is the first
        parameter of posterior Beta distribution describing the latent vector,
        'v', which is involved in the stick-breaking construction of the DP.

    v_b_ : torch.Tensor, shape ( K )
        For each of the 'K' elements in the truncated DP, this is the second
        parameter of posterior Beta distribution describing the latent vector,
        'v', which is involved in the stick-breaking construction of the DP.

    R_ : torch.Tensor, shape ( N, K )
        Each element of this matrix represents the posterior probability that
        instance 'n' belongs to cluster 'k'.

    w_mu_ : torch.Tensor, shape ( M, D, K )
        The posterior means of the Normal distributions describing each of the
        predictor coefficients, for each dimension of the proportion vector,
        for each of the 'K' components.

    w_var_ : torch.Tensor, shape ( M, D, K )
        The posterior variances of the Normal distributions describing each of
        the predictor coefficients, for each dimension of the proportion vector,
        for each of the 'K' components.

    lambda_a : torch.Tensor, shape ( D, K ), optional
        For component 'K' and target dimension 'D', this is the first parameter
        of the posterior Gamma distribution describing the precision of the
        target variable. Only relevant for continuous (Gaussian) target 
        variables.

    lambda_b : torch.Tensor, shape ( D, K ), optional
        For component 'K' and target dimension 'D', this is the second parameter
        of the posterior Gamma distribution describing the precision of the
        target variable. Only relevant for continuous (Gaussian) target 
        variables.
    """
    def __init__(self, *args, **kwargs):
        if len(args) == 1 and len(kwargs.keys()) == 0:
            self.copy(args[0])
        else:
            if isinstance(args[0], np.ndarray):
                self.w_mu0_ = torch.from_numpy(args[0])
            else:
                self.w_mu0_ = torch.clone(args[0])
            if isinstance(args[1], np.ndarray):
                self.w_var0_ = torch.from_numpy(args[1])
            else:
                self.w_var0_ = torch.clone(args[1])
            if isinstance(args[2], np.ndarray):
                self.lambda_a0_ = torch.from_numpy(args[2])
            else:
                self.lambda_a0_ = torch.clone(args[2])
            if isinstance(args[3], np.ndarray):                
                self.lambda_b0_ = torch.from_numpy(args[3])
            else:
                self.lambda_b0_ = torch.clone(args[3])
                
            self.prec_prior_weight_ = args[4]
            self.alpha_ = args[5]

            self.lambda_a0_mod_ = None
            self.lambda_b0_mod_ = None
            
            self.w_mu_ = None
            self.w_var_ = None
            self.lambda_a_ = None
            self.lambda_b_ = None
            self.v_a_ = None
            self.v_b_ = None
            self.R_ = None
            self.gb_ = None
            self.N_ = None
            
            self.K_ = 10
            self.prob_thresh_ = 0.001
            if len(args) > 6:
                self.K_ = args[6]
            if len(args) > 7:
                self.prob_thresh_ = 0.001

            if 'K' in kwargs.keys():
                self.K_ = kwargs['K']
            if 'prob_thresh' in kwargs.keys():
                self.prob_thresh_ = kwargs['prob_thresh']
                
            self.M_ = self.w_mu0_.shape[0]
            self.D_ = self.w_mu0_.shape[1]

            self.target_type_ = {}

            self.lower_bounds_ = []

            self.X_ = None
            self.Y_ = None
            self.target_names_ = None
            self.predictor_names_ = None

            self.group_first_index_ = None
        
            self.sig_trajs_ = torch.ones(self.K_, dtype=bool)

        assert self.w_mu0_.shape[0] == self.w_var0_.shape[0] and \
          self.w_mu0_.shape[1] == self.w_var0_.shape[1], \
          "Shape mismatch for w_mu0_ and w_var0_"

        assert self.lambda_a0_.shape[0] == self.lambda_b0_.shape[0], \
          "Shape mismatch for lambda_a0_ and lambda_b0_"

        assert self.w_mu0_.shape[1] == self.lambda_a0_.shape[0], \
          "Target dimension mismatch"


    def copy(self, mm):
        """Performs a deep copy of the member variables of the input model

        Parameters
        ----------
        mm : MultDPRegression instance
        The model instance that will be copied.

        """
        # There are currently 31 member variables. Check that this holds.
        members = [attr for attr in dir(mm) \
               if not callable(getattr(mm, attr)) \
               and not attr.startswith("__")]

        #assert len(members) == 33, "Member variables unaccounted for"

        self.D_ = mm.D_
        self.K_ = mm.K_
        self.M_ = mm.M_
        self.N_ = mm.N_
        self.R_ = mm.R_.clone()
        self.X_ = mm.X_.clone()
        self.Y_ = mm.Y_.clone()

        try:
            self.alpha_ = mm.alpha_
        except AttributeError as error:
            print("WARNING: alpha_ is not an attribue of input model.")
            print("Setting copy version to None")
            self.alpha_ = None
    
        try:
            self.lambda_a0_mod_ = mm.lambda_a0_mod_.clone()
            self.lambda_b0_mod_ = mm.lambda_b0_mod_.clone()           
        except AttributeError as error:
            print("WARNING: lambda_a0_mod_, lambda_b0_mod_ \
            not attribues of input model.")
            print("Setting copy versions to None")
            self.lambda_a0_mod_ = None
            self.lambda_b0_mod_ = None         
    
        # Note: for non-tensor attributes, we still use copy.deepcopy()
        self.df_ = copy.deepcopy(mm.df_)
    
        try:
            self.gb_ = copy.deepcopy(mm.gb_)
        except AttributeError as error:
            print("WARNING: gb_ is not an attribue of input model.")
            print("Setting copy version to None")
            self.gb_ = None
    
        self.lambda_a0_ = mm.lambda_a0_.clone()
        self.lambda_a_ = mm.lambda_a_.clone()
        self.lambda_b0_ = mm.lambda_b0_.clone()
        self.lambda_b_ = mm.lambda_b_.clone()
        self.lower_bounds_ = mm.lower_bounds_.clone()
        self.predictor_names_ = copy.deepcopy(mm.predictor_names_)
        self.prob_thresh_ = mm.prob_thresh_
        self.sig_trajs_ = mm.sig_trajs_.clone()
        self.target_names_ = copy.deepcopy(mm.target_names_)
        self.v_a_ = mm.v_a_.clone()
        self.v_b_ = mm.v_b_.clone()
        self.w_covmat_ = mm.w_covmat_.clone()
        self.w_mu0_ = mm.w_mu0_.clone()
        self.w_mu_ = mm.w_mu_.clone()
        self.w_var0_ = mm.w_var0_.clone()
        self.w_var_ = mm.w_var_.clone()
    
        try:
            self.num_binary_targets_ = mm.num_binary_targets_
        except AttributeError as error:
            print("WARNING: num_binary_targets_ not an attribue of input model.")
            print("Setting copy version value to 0")
            self.num_binary_targets_ = 0
    
        try: 
            self.target_type_ = copy.deepcopy(mm.target_type_)
        except:
            self.target_type_ = {}
            for d in range(self.D_):
                if set(self.Y_[:, d]).tolist() == [1.0, 0.0]:
                    self.target_type_[d] = 'binary'
                    self.num_binary_targets_ += 1
                else:
                    self.target_type_[d] = 'gaussian'
    
        try:
            self.prec_prior_weight_ = mm.prec_prior_weight_
        except AttributeError as error:
            print("WARNING: prec_prior_weight_ not an attribue of input model.")
            print("Setting copy version value to 0.25")
            self.prec_prior_weight_ = 0.25
    
        try:
            self.group_first_index_ = np.array(mm.group_first_index_).astype(bool)
        except:
            self._set_group_first_index(self.df_, self.gb_)
    
        try:
            self.xi_ = mm.xi_.clone()
        except AttributeError as error:
            print("WARNING: xi_ is not an attribue of input model.")
            print("Setting copy version to None")
            self.xi_ = None


    def _set_group_first_index(self, df, gb):
        """
        """
        self.group_first_index_ = self._get_group_first_index(df, gb)

        
    def _get_group_first_index(self, df, gb=None):
        """This function returns a boolean vector corresponding to the rows in 
        df. Vector elements are false except at those locations corresponding to
        the first entry for a group. This vector is intended to faciliate update
        of the R matrix.

        Parameters
        ----------
        df : pandas DataFrame
            Data structure from which group-based information will be extracted

        gb : pandas DataFrameGroupBy, optional
            Grouped object corresponding to df. These groups will be used to 
            set the index values. If not gb object is specified, a boolean
            vector of all true will be returned.

        Returns
        -------
        group_first_index : array, shape ( N )
            Boolean vector. For a given group of indices, the first is set to
            true; the other indices in the group are set to false.
        """
        group_first_index = np.zeros(df.shape[0], dtype=bool)
        if gb is not None: 
            for kk in gb.groups.keys():
                group_first_index[gb.get_group(kk).index[0]] = True
        else:
            group_first_index = np.ones(N, dtype=bool) 

        return group_first_index
            
            
    def fit(self, target_names, predictor_names, df, groupby=None, iters=100,
            R=None, traj_probs=None, traj_probs_weight=None, v_a=None,
            v_b=None, w_mu=None, w_var=None, lambda_a=None, lambda_b=None,
            verbose=False, weights_only=False, num_init_trajs=None):
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

        weights_only : bool, optional
            If true, the fitting routine will be fored to only optimize the 
            trajectory weights. The assumption is that the specified prior file 
            contains previously modeled trajectory information, and that those 
            trajectories should be used for the current fit. This option can be 
            useful if a model learned from one cohort is applied to another 
            cohort, where it is possible that the relative proportions of 
            different trajectory subgroups differs. By using this flag, the 
            proportions of previously determined trajectory subgroups will be 
            determined for the current data set.

        num_init_trajs : int, optional
            If specified, the initialization procedure will attempt to ensure 
            that the number of initial trajectories in the fitting routine
            equals the specified number.       
        """
        if traj_probs_weight is not None:
            assert traj_probs_weight >= 0 and traj_probs_weight <=1, \
                "Invalid traj_probs_weightd value"

        self.X_ = torch.tensor(df[predictor_names].values, dtype=torch.float64)
        self.Y_ = torch.tensor(df[target_names].values, dtype=torch.float64)

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
        if lambda_a is not None:
            self.lambda_a_ = torch.from_numpy(lambda_a).double()
        else:
            self.lambda_a_ is None
            
        if lambda_b is not None:
            self.lambda_b_ = torch.from_numpy(lambda_b).double()
        else:
            self.lambda_b_ is None
            
        if w_mu is not None:
            self.w_mu_ = torch.from_numpy(w_mu).double()
        else:
            self.w_mu_ = None
            
        if w_var is not None:
            self.w_var_ = torch.from_numpy(w_var).double()
        else:
            self.w_var_ is None
            
        if v_a is not None:
            if torch.is_tensor(v_a):                
                self.v_a_ = v_a.clone().detach()
            else:
                self.v_a_ = torch.from_numpy(v_a).double()                
        else:
            self.v_a_ = None
            
        if v_b is not None:
            if torch.is_tensor(v_b):
                self.v_b_ = v_b.clone().detach()
            else:
                self.v_b_ = torch.from_numpy(v_b).double()
        else:
            self.v_b_ = None
            
        if R is not None:
            self.R_ = torch.from_numpy(R).double()
        else:
            R = None

        self.df_ = df

        # df_helper_ is introduced as a data structure that will facilitate
        # tallying the likelihood terms during the update of Z parameters.
        # It has a column that mirrors the subject ID of the input df and will
        # enable grouped tallying of likelihood terms. It is initialized here,
        # and it is updated with each call to 'update_z'.
        self.df_helper_ = pd.DataFrame(df[[groupby]])
        for k in range(self.K_):
            self.df_helper_['like_accum_' + str(k)] = np.nan*np.zeros(self.N_)
        
        self.gb_ = None        
        if groupby is not None:
            self.gb_ = self.df_helper_.groupby(groupby)
    
        self._set_group_first_index(self.df_, self.gb_)

        # w_covmat_ is used for binary target variables. The EM algorithm
        # that is used to estimate w_mu_ and w_var_ for binary targets
        # actually gives us a full covariance matrix which we take advantage
        # of when performing sampling based estimates elsewhere. Although
        # we only need/have w_covmat_ for binary targets, we allocate space
        # over all dimensions, D, to make implementation clearer (i.e. when
        # we iterate over D)
        self.w_covmat_ = torch.full([self.M_, self.M_, self.D_, self.K_],
                                    torch.tensor(float('nan'))).double()

        self.num_binary_targets_ = 0
        for d in range(self.D_):
            if set(self.Y_[:, d].tolist()) <= {1.0, 0.0}:
                self.target_type_[d] = 'binary'
                self.num_binary_targets_ += 1
            else:
                self.target_type_[d] = 'gaussian'
        print("Initializing parameters...")
        self.init_traj_params(traj_probs)

        # The prior over the residual precision can get overwhelmed by the
        # data -- so much so that residual precision posteriors can wind up
        # in regimes that have near-zero mass in the prior. Given this, we
        # scale the prior params (essentially lowering the variance of the
        # prior) by an amount proportional to the number of subjects in the
        # data set. Note that this step needs to be done AFTER
        # init_traj_params, which uses the original prior to randomly
        # initialize trajectory precisions.
        self.lambda_a0_mod_ = self.lambda_a0_*self.prec_prior_weight_
        self.lambda_b0_mod_ = self.lambda_b0_*self.prec_prior_weight_

        if self.v_a_ is None:
            self.v_a_ = torch.ones(self.K_)
    
        if self.v_b_ is None:
            self.v_b_ = self.alpha_*torch.ones(self.K_)
    
        if self.R_ is None:
            if num_init_trajs is None:
                self.init_R_mat(traj_probs, traj_probs_weight)
            else:
                for ii in range(100):
                    self.init_R_mat(traj_probs, traj_probs_weight)
                    if torch.sum(self.sig_trajs_).item() == num_init_trajs:
                        break
                
        self.fit_coordinate_ascent(iters, verbose, weights_only)

                
    def fit_coordinate_ascent(self, iters, verbose, weights_only=False):
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
    
        weights_only : bool, optional
            If true, the fitting routine will be fored to only optimize the 
            trajectory weights. The assumption is that the specified prior file 
            contains previously modeled trajectory information, and that those 
            trajectories should be used for the current fit. This option can be 
            useful if a model learned from one cohort is applied to another 
            cohort, where it is possible that the relative proportions of 
            different trajectory subgroups differs. By using this flag, the 
            proportions of previously determined trajectory subgroups will be 
            determined for the current data set.
        """
        inc = 0
        while inc < iters:
            inc += 1

            self.update_v()
            if self.num_binary_targets_ > 0:
                self.update_w_logistic(em_iters=1)
            if (self.D_ - self.num_binary_targets_ > 0) and \
               (not weights_only):
                self.update_w_gaussian()
                self.update_lambda() 

            self.R_ = self.update_z(self.X_, self.Y_)
            
            self.sig_trajs_ = \
                torch.max(self.R_, dim=0).values > self.prob_thresh_

            if verbose:
                torch.set_printoptions(precision=2)
                print(f"iter {inc}, {torch.sum(self.R_, dim=0).numpy()}")

                
    def update_v(self):
        """Updates the parameters of the Beta distributions for latent
        variable 'v' in the variational approximation.
        """
        self.v_a_ = 1.0 + \
            torch.sum(self.R_[self.group_first_index_, :], dim=0)        

        for k in torch.arange(0, self.K_):
            self.v_b_[k] = self.alpha_ + \
                torch.sum(self.R_[self.group_first_index_, k+1:])


    def get_R_matrix(self, df=None, gb_col=None, df_helper=None):
        """For each individual, computes the probability that he/she belongs to
        each of the trajectories.
        """        
        expec_ln_v = psi(self.v_a_) - psi(self.v_a_ + self.v_b_)
        expec_ln_1_minus_v = psi(self.v_b_) - psi(self.v_a_ + self.v_b_)
    
        expec_ln_v_terms = expec_ln_v.clone().detach()
        for k in range(1, self.K_):
            expec_ln_v_terms[k] = expec_ln_v_terms[k] + \
                torch.sum(expec_ln_1_minus_v[0:k])

        if df is not None:
            N = df.shape[0]
            Y = torch.from_numpy(df[self.target_names_].values).double()
            X = torch.from_numpy(df[self.predictor_names_].values).double()
        else:
            N = self.N_
            Y = self.Y_
            X = self.X_

        if df_helper is not None:
            # If we're here, it means this function has been called
            # from update_z
            df_helper = df_helper
            gb = self.gb_
        else:
            # If we're here, it means this function has been called
            # from augment_df_with_traj_info
            if gb_col is None:
                df_helper = pd.DataFrame(index=range(df.shape[0]))
            else:
                df_helper = pd.DataFrame(df[[gb_col]])                
            for k in range(self.K_):
                df_helper['like_accum_' + str(k)] = np.nan*np.zeros(N)            

            if gb_col is None:
                gb = df_helper.groupby(df.index)
            else:
                gb = df_helper.groupby(gb_col)
                
        likelihood_accum = torch.zeros([N, self.K_]).double()
        
        if self.num_binary_targets_ > 0:
            num_samples = 100 # Arbitrary. Should be "big enough"
            mc_term = torch.zeros([N, self.K_]).double()
        for d in range(0, self.D_):
            non_nan_ids = ~torch.isnan(Y[:, d])
            if self.target_type_[d] == 'gaussian':
                tmp = (torch.matmul(self.w_mu_[:, d, :].T, \
                    X[non_nan_ids, :].T)**2).T + \
                    torch.sum((X[non_nan_ids, None, :]**2)*\
                    (self.w_var_[:, d, :].T)[None, :, :], 2)

                likelihood_accum[non_nan_ids, :] = \
                    likelihood_accum[non_nan_ids, :] + \
                  0.5*(psi(self.lambda_a_[d, :]) - \
                    torch.log(self.lambda_b_[d, :]) - \
                    torch.log(torch.tensor(2*np.pi)) - \
                    (self.lambda_a_[d, :]/self.lambda_b_[d, :])*\
                    (tmp - \
                     2*Y[non_nan_ids, d, None]*torch.matmul(X[non_nan_ids, :], \
                    self.w_mu_[:, d, :]) + \
                    Y[non_nan_ids, d, None]**2))
            elif self.target_type_[d] == 'binary':
                for k in range(self.K_):
                    dist = MultivariateNormal(self.w_mu_[:, d, k],
                                              self.w_covmat_[:, :, d, k])
                    samples = dist.sample((num_samples,))

                    mc_term[non_nan_ids, k] = \
                        torch.mean(torch.log1p(torch.exp(\
                        torch.matmul(X[non_nan_ids, :], samples.T))), dim=1)

                    likelihood_accum[non_nan_ids, k] = \
                        likelihood_accum[non_nan_ids, k] + \
                        Y[non_nan_ids, d]*\
                        torch.matmul(X[non_nan_ids, :], self.w_mu_[:, d, k]) - \
                        mc_term[non_nan_ids, k]

        # Add columns to df_ for computational purposes
        like_accum_cols = []
        for k in range(self.K_):
            like_accum_cols.append('like_accum_' + str(k))
            df_helper[like_accum_cols[k]] = likelihood_accum[:, k]

        ln_rho_deb = torch.ones([gb.ngroups, self.K_], \
                            dtype=torch.float64)*expec_ln_v_terms.unsqueeze(0) + \
                            torch.from_numpy(\
                                gb[like_accum_cols].sum().values).double()
        
        # The values of 'ln_rho' will in general have large magnitude, causing
        # exponentiation to result in overflow. All we really care about is the
        # normalization of each row. We can use the identity exp(a) =
        # 10**(a*log10(e)) to put things in base ten, and then subtract from
        # each row the max value and also clipping the resulting row vector to
        # lie within -300, 300 to ensure that when we exponentiate we don't
        # have any overflow issues.
        rho_10 = ln_rho_deb*np.log10(np.e)
        
        # The following line ensures that once a trajectory has been assigned 0
        # weight (which sig_trajs_ keeps track of), it won't be resurrected.
        rho_10[:, ~self.sig_trajs_] = -sys.float_info.max
        rho_10_shift = \
            10**((rho_10.T - torch.max(rho_10, dim=1).values).T + 300).\
            clip(-300, 300)

        R_grouped = (rho_10_shift.T/torch.sum(rho_10_shift, dim=1)).T
        
        # Any instance that has miniscule probability of belonging to a
        # trajectory, set it's probability of belonging to that trajectory to 0
        R_grouped[R_grouped <= self.prob_thresh_] = 0
        R_grouped = R_grouped/torch.sum(R_grouped, dim=1).unsqueeze(1)

        R_indices = [index for group_indices in gb.groups.values() \
                     for index in group_indices]

        gb_indices_rep = [kk for kk, gg in enumerate(gb.groups.keys()) \
                          for _ in range(gb.get_group(gg).shape[0])]        

        R = np.zeros([N, self.K_])
        R[R_indices, :] = R_grouped[gb_indices_rep, :]

        return torch.from_numpy(R).double()            
            
    def update_z(self, X, Y):
        """
        """
        return self.get_R_matrix(df_helper=self.df_helper_)

    
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
            
                for k in range(self.K_):
                    non_nan_ids = torch.isnan(self.Y_[:, d]).logical_not()
    
                    for i in range(em_iters):
                        # E-step
                        Z_vec = 0.5*self.R_[non_nan_ids, k]*\
                            (1/self.xi_[non_nan_ids, d_bin, k])*\
                            torch.tanh(0.5*self.xi_[non_nan_ids, d_bin, k])
                        
                        sig_mat_0 = torch.diag(self.w_var0_[:, d])
                        mu_0 = self.w_mu0_[:, d]
    
                        self.w_covmat_[:, :, d, k] = \
                            torch.inverse(torch.inverse(sig_mat_0) + \
                                torch.mm(self.X_[non_nan_ids, :].t(), \
                                    Z_vec[:, None]*self.X_[non_nan_ids, :]))

                        self.w_var_[:, d, k] = \
                            torch.diag(self.w_covmat_[:, :, d, k])

                        self.w_mu_[:, d, k] = \
                            torch.mv(self.w_covmat_[:, :, d, k], \
                                   torch.mv(self.X_[non_nan_ids, :].t(),
                                self.R_[non_nan_ids, k]*\
                                      (self.Y_[non_nan_ids, d] - 0.5)) + \
                                    torch.mv(torch.inverse(sig_mat_0), mu_0))

                        # M-step
                        self.xi_[non_nan_ids, d_bin, k] = \
                            torch.sqrt(torch.sum((self.X_[non_nan_ids, :]*\
                                torch.mm(self.w_covmat_[:, :, d, k], \
                                self.X_[non_nan_ids, :].t()).t()), 1) + \
                                torch.pow(torch.mv(self.X_[non_nan_ids, :], \
                                            self.w_mu_[:, d, k]), 2))
  
    def update_w_gaussian(self):
        """ Updates the variational distributions over predictor coefficients 
        corresponding to continuous (Gaussian) target variables. 
        """
        mu0_DIV_var0 = self.w_mu0_/self.w_var0_
        for m in range(0, self.M_):
            ids = torch.ones(self.M_, dtype=bool)
            ids[m] = False
            for d in range(0, self.D_):
                if self.target_type_[d] == 'gaussian':
                    non_nan_ids = ~torch.isnan(self.Y_[:, d])
    
                    tmp1 = (self.lambda_a_[d, self.sig_trajs_]/\
                            self.lambda_b_[d, self.sig_trajs_])*\
                            (torch.sum(self.R_[:, self.sig_trajs_, None]\
                                    [non_nan_ids, :, :]*\
                                    self.X_[non_nan_ids, None, :]**2, 0).T)\
                                    [:, None, :]
    
                    self.w_var_[:, :, self.sig_trajs_] = \
                        (tmp1 + (1.0/self.w_var0_)[:, :, None])**-1
    
                    sum_term = \
                        torch.sum(self.R_[non_nan_ids, :][:, self.sig_trajs_]*\
                                  self.X_[non_nan_ids, m, None]*\
                                (torch.matmul(self.X_[:, ids][non_nan_ids, :], \
                                                self.w_mu_[ids, d, :]\
                                                [:, self.sig_trajs_]) - \
                                   self.Y_[non_nan_ids, d][:, None]), 0)
                    self.w_mu_[m, d, self.sig_trajs_] = \
                        self.w_var_[m, d, self.sig_trajs_]*\
                        (-(self.lambda_a_[d, self.sig_trajs_]/\
                           self.lambda_b_[d, self.sig_trajs_])*\
                         sum_term + mu0_DIV_var0[m, d])


    def update_lambda(self):
        """Updates the variational distribution over latent variable lambda.
        """    
        for d in range(self.D_):
            if self.target_type_[d] == 'gaussian':
                non_nan_ids = ~torch.isnan(self.Y_[:, d])

                self.lambda_a_[d, self.sig_trajs_] = \
                    self.lambda_a0_mod_[d, None] + \
                    0.5*torch.sum(self.R_[:, self.sig_trajs_][non_nan_ids, :], 0)\
                    [None, :]

                tmp = (torch.mm(self.w_mu_[:, d, self.sig_trajs_].T, \
                           self.X_[non_nan_ids, :].T)**2).T + \
                           torch.sum((self.X_[non_nan_ids, None, :]**2)*\
                                  (self.w_var_[:, d, self.sig_trajs_].T)\
                                  [None, :, :], 2)
    
                self.lambda_b_[d, self.sig_trajs_] = \
                    self.lambda_b0_mod_[d, None] + \
                    0.5*torch.sum(self.R_[non_nan_ids, :][:, self.sig_trajs_]*\
                            (tmp - 2*self.Y_[non_nan_ids, d, None]*\
                             torch.mm(self.X_[non_nan_ids, :], \
                                    self.w_mu_[:, d, self.sig_trajs_]) + \
                             self.Y_[non_nan_ids, d, None]**2), 0)


    def sample(self, index=None, x=None):
        """sample from the posterior distribution using the input data.

        Parameters
        ----------
        index : int, optional
            The data sample index at which to sample. If none specified, samples
            will be drawn for all of the 'N' sample points.

        x : torch.Tensor, shape ( M ), optional
            Only relevant if 'index' is also specified. If both 'index' and 'x'
            are specified, a sample will be drawn for the data point specified
            by 'index', using the predictor values specified by 'x'.  If 'x' is
            not specified, the original predictor values for the 'index' data
            point will be used to draw the sample. Here, 'M' is the dimension of
            the predictors.

        Returns
        -------
        y_rep : torch.Tensor, shape ( N, M ) or ( M )
            The sample(s) randomly drawn from the posterior distribution
        """
        if index is None:
            indices = range(0, self.N_)
            y_rep = torch.zeros([self.N_, self.D_])
            X = self.X_
        else:
            if not isinstance(index, int) and not isinstance(index, np.int64):
                raise ValueError('index must be an integer if specified')

            indices = range(index, index+1)
            y_rep = torch.zeros([1, self.D_])

            if x is not None:
                if len(x.shape) != 1:
                    raise ValueError('x must be a vector')

                if x.shape[0] != self.M_:
                    raise ValueError('x has incorrect dimension')
                    
        for n in indices:
            z = torch.multinomial(self.R_[n, :]/torch.sum(self.R_[n, :]), 1)
            for d in range(0, self.D_):
                if self.target_type_[d] == 'gaussian':

                    scale = 1./self.lambda_b_[d, z]
                    shape = self.lambda_a_[d, z]
                    var = 1./torch.distributions.Gamma(shape, scale).sample()

                    mean = self.w_mu_[:, d, z][:, 0]
                    covariance_matrix = torch.diag(self.w_var_[:, d, z][:, 0])
                    dist = torch.distributions.\
                        MultivariateNormal(mean, covariance_matrix)
                    co = dist.sample()

                    mu = torch.matmul(co, x if x is not None else self.X_[n, :])

                    y_rep[n if index is None else 0, d] = \
                        torch.sqrt(var)*torch.randn(1) + mu

                else:
                    # Target assumed to be binary
                    mean = self.w_mu_[:, d, z][:, 0]
                    covariance_matrix = self.w_covmat_[:, :, d, z][:, 0]
                    dist = torch.distributions.\
                        MultivariateNormal(mean, covariance_matrix)
                    co = dist.sample()

                    mu = torch.matmul(co, x if x is not None else self.X_[n, :])
                    p = torch.exp(mu)/(1 + torch.exp(mu))
                    y_rep[n if index is None else 0, d] = \
                        torch.distributions.Binomial(1, p).sample()

        return y_rep


    def lppd(self, y, index, S=20, x=None):
        """Compute the log pointwise predictive density (lppd) at a specified
        point.

        # TODO: test binary implementation

        This function implements equation 7.5 of 'Bayesian Data Analysis, Third
        Edition' for a single point.

        Parameters
        ----------
        y : torch.Tensor, shape ( D )
            The vector of target values at which to compute the log pointwise
            predictive density.

        index : int, optional
            The data sample index at which to sample.

        S : int, optional
            The number of samples to draw in order to compute the lppd.

        x : torch.Tensor, shape ( M ), optional
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
        if not isinstance(index, int):
            raise ValueError('index must be an integer')
    
        if x is not None:
            if len(x.shape) != 1:
                raise ValueError('x must be a vector')
    
            if x.shape[0] != self.M_:
                raise ValueError('x has incorrect dimension')
    
        y = torch.from_numpy(y).float()
        accums = torch.zeros([self.D_, S])
        
        for s in range(0, S):  
            z = torch.multinomial(self.R_[index, :], 1)
            for d in range(0, self.D_):
                if self.target_type_[d] == 'gaussian':
                    mu = self.w_mu_[:, d, z][0]
                    var = self.w_var_[:, d, z][0]
                    co = MultivariateNormal(mu, torch.diag(var)).sample()
    
                    if x is not None:
                        mu = torch.dot(co, x)
                    else:
                        mu = torch.dot(co, self.X_[index, :])
    
                    shape = self.lambda_a_[d, z]
                    scale = 1./self.lambda_b_[d, z]
                    var = 1./Gamma(shape, scale).sample()
    
                    accums[d, s] = torch.clamp((1/torch.sqrt(var*2*torch.pi))*\
                      torch.exp((-(0.5/var)*(y[d] - mu)**2).float()),
                      1e-300, 1e300) 
                else:
                    mu = self.w_mu_[:, d, z][0]
                    cov = self.w_covmat_[:, :, d, z]
                    co = MultivariateNormal(mu, cov).sample()
    
                    if x is not None:
                        mu = torch.dot(co, x)
                    else:
                        mu = torch.dot(co, self.X_[index, :])
    
                    accums[d, s] = \
                        torch.clamp((torch.exp(mu)**y[d])/(1 + torch.exp(mu)),
                                    1e-300, 1e300)
    
        log_dens = torch.sum(torch.log(torch.mean(accums, 1)))
    
        return log_dens.item()
    

    def cast_to_torch(self):
        """Casts numpy arrays to torch tensors
        """
        if not torch.is_tensor(self.Y_):
            self.Y_ = torch.from_numpy(self.Y_).double()

        if not torch.is_tensor(self.X_):
            self.X_ = torch.from_numpy(self.X_).double()

        if not torch.is_tensor(self.R_):
            self.R_ = torch.from_numpy(self.R_).double()

        if not torch.is_tensor(self.w_mu_):
            self.w_mu_ = torch.from_numpy(self.w_mu_).double()

        if not torch.is_tensor(self.w_var_):
            self.w_var_ = torch.from_numpy(self.w_var_).double()
            
        if not torch.is_tensor(self.lambda_a_):
            self.lambda_a_ = torch.from_numpy(self.lambda_a_).double()
            self.lambda_b_ = torch.from_numpy(self.lambda_b_).double()            

        if not torch.is_tensor(self.v_a_):
            self.v_a_ = torch.from_numpy(self.v_a_)
            self.v_b_ = torch.from_numpy(self.v_b_)            
            
    
    def log_likelihood(self):
        """Compute the log-likelihood given expected values of the latent 
        variables.
    
        TODO: Test implementation of binary targets
    
        Returns
        -------
        log_likelihood : float
            The log-likelihood
        """
        if not torch.is_tensor(self.Y_):
            Y = torch.from_numpy(self.Y_).double()
        else:
            Y = self.Y_

        if not torch.is_tensor(self.X_):
            X = torch.from_numpy(self.X_).double()
        else:
            X = self.X_

        if not torch.is_tensor(self.R_):
            R = torch.from_numpy(self.R_).double()
        else:
            R = self.R_

        if not torch.is_tensor(self.w_mu_):
            w_mu = torch.from_numpy(self.w_mu_).double()
        else:
            w_mu = self.w_mu_
            
        if not torch.is_tensor(self.lambda_a_):
            lambda_a = torch.from_numpy(self.lambda_a_).double()
            lambda_b = torch.from_numpy(self.lambda_b_).double()            
        else:
            lambda_a = self.lambda_a_
            lambda_b = self.lambda_b_                        
            
        tmp_k = torch.zeros(self.N_, dtype=torch.float64)
        for k in range(self.K_):
            tmp_d = torch.ones(self.N_, dtype=torch.float64)
            for d in range(self.D_):
                # Some of the target variables can be missing (NaNs). Exclude
                # these from the computation.
                ids = torch.isnan(Y[:, d]) == 0
                mu = torch.mv(X, w_mu[:, d, k])
                
                if self.target_type_[d] == 'gaussian':
                    v = lambda_b[d, k]/lambda_a[d, k]
                    co = 1/torch.sqrt(2.*torch.pi*v)
    
                    tmp_d[ids] = tmp_d[ids]*(co*torch.exp(-((Y[ids, d]-
                                               mu[ids])**2)/(2.*v)))
                else:
                    # Target assumed to be binary
                    tmp_d[ids] = tmp_d[ids]*\
                        ((torch.exp(mu)**Y[ids, d])/(1 + torch.exp(mu)))
                    
            tmp_k = tmp_k + R[:, k]*tmp_d
        log_likelihood = torch.sum(torch.log(tmp_k))
                
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
        self.cast_to_torch()
        
        ll = self.log_likelihood()
        num_trajs = torch.sum(torch.sum(self.R_, 0) > 0.0)
    
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
            num_obs_tally = num_obs_tally + \
                torch.sum(torch.isnan(self.Y_[:, d]) == False)
        
        bic_obs = ll - 0.5*num_params*torch.log(num_obs_tally)
        
        if self.gb_ is not None:
            num_subjects = self.gb_.ngroups
            bic_groups = ll - 0.5*num_params*np.log(num_subjects)
    
            return (bic_obs, bic_groups)
        else:
            return bic_obs


    def compute_waic2(self, S=100):
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
        self.cast_to_torch()
        accum = np.zeros([self.N_, self.K_, S])
        selector = np.zeros([self.N_, self.K_, S], dtype=bool)
        
        inc = 0
        for gg in self.gb_.groups:
            indices = self.gb_.get_group(gg).index

            traj_probs = self.R_[self.gb_.get_group(gg).index[0], :]
            selector[indices, :] = \
                np.random.multinomial(1, traj_probs, size=S).astype(bool).T

            inc = inc + 1
            for ii, k in enumerate(np.where(self.sig_trajs_)[0]):
                probs = np.ones([S, indices.shape[0]])
                for d in range(0, self.D_):
                    co = multivariate_normal(self.w_mu_.numpy()[:, d, k],
                        diag(self.w_var_.numpy()[:, d, k]), S)

                    mu = dot(co, self.X_[indices, :].T)

                    scale = 1./self.lambda_b_.numpy()[d, k]
                    shape = self.lambda_a_.numpy()[d, k]
                    var = 1./gamma(shape, scale, size=S)
                    
                    probs = probs*((1/sqrt(2*np.pi*var))[:, newaxis]*\
                            exp(-(1/(2*var))[:, newaxis]*\
                                (mu - self.Y_[indices, d].numpy())**2))
                accum[indices, k, :] = probs.T
                
        masked = np.sum(accum*selector, 1)

        lppd = np.sum(np.log(np.nanmean(masked, 1).clip(1e-320, 1e320)))
        p_waic = np.sum(np.nanvar(np.log(masked.clip(1e-320, 1e320)), 1))
        waic2 = -2*(lppd - p_waic)
        
        return waic2 
        

    def init_traj_params(self, traj_probs=None):
        """Initializes trajectory parameters.

        Parameters
        ----------
        traj_probs : array, shape ( K ), optional
            A priori probabilitiey of each of the K trajectories. Each element 
            must be >=0 and <= 1, and all elements must sum to one. Use of this
            vector assumes that w_mu_, w_var_, lambda_a_, lambda_b_, v_a_, and
            v_b_ have already been set (in the calling routine). These values
            are assumed to come from a prior, which in turn was informed by some
            previous model fit. The current function attempts to initialize 
            w_mu_, w_var_, lambda_a_, and lambda_b_ with reasonable values. 
            However, if these have already been set (in the calling routine), 
            we want to use those values instead of the ones produced by this
            function. If 'traj_probs' is 0 for any element, this function will
            graft the values determined by the method implemented here onto
            w_mu_, etc.
        """
        if self.w_var_ is None:
            self.w_var_ = torch.zeros([self.M_, self.D_, self.K_],
                                      dtype=torch.float64)
            for k in range(self.K_):
                self.w_var_[:, :, k] = self.w_var0_.clone().detach()

        if torch.isnan(torch.sum(self.w_var_)):
            for kk in range(self.K_):
                ids = torch.isnan(self.w_var_[:, :, kk])
                if torch.sum(ids) > 0:
                    self.w_var_[:, :, kk][ids] = \
                        torch.tensor(self.w_var0_[ids])

        if self.w_mu_ is not None:
            if torch.isnan(torch.sum(self.w_mu_)):
                ids = torch.isnan(self.w_mu_)
                self.w_mu_[ids] = 0

        if self.lambda_a_ is None and self.lambda_b_ is None:
            if self.gb_ is not None:
                scale_factor = self.gb_.ngroups
            else:
                scale_factor = self.N_
            self.lambda_a_ = \
                (scale_factor*torch.ones([self.D_, self.K_])).double()
            self.lambda_b_ = \
                (scale_factor*torch.ones([self.D_, self.K_])).double()
            for d in range(self.D_):
                scale = 1./self.lambda_b0_[d]
                shape = self.lambda_a0_[d]                                
                self.lambda_a_[d, :] = self.lambda_a_[d, :]*(\
                    torch.distributions.Gamma(shape, scale).sample((self.K_,)))

        if torch.isnan(torch.sum(self.lambda_a_)):
            for dd in range(self.D_):
                for kk in range(self.K_):
                    if torch.isnan(self.lambda_a_[dd, kk]):
                        self.lambda_b_[dd, kk] = 1
                        scale = 1./self.lambda_b0_[dd]
                        shape = self.lambda_a0_[dd]                     
                        self.lambda_a_[dd, kk] = \
                            torch.distributions.Gamma(shape, scale).sample()

        w_mu_tmp = torch.from_numpy(sample_cos(self.w_mu0_, self.w_var0_,
                                               num_samples=self.K_)).double()
        if self.w_mu_ is None:
            self.w_mu_ = w_mu_tmp
        else:
            self.w_mu_[:, :, traj_probs==0] = \
                w_mu_tmp[:, :, traj_probs==0].double()

        #-----------------------------------------------------------------------
        # Initialize xi if needed
        #-----------------------------------------------------------------------            
        if self.num_binary_targets_ > 0:
            self.xi_ = \
                torch.ones([self.N_, self.num_binary_targets_, self.K_]).\
                double()
        else:
            self.xi_ = None 
    
        d_bin = -1
        for d in range(self.D_):
            if self.target_type_[d] == 'binary':
                d_bin += 1

                for k in np.where(self.sig_trajs_)[0]:
                    non_nan_ids = ~torch.isnan(self.Y_[:, d])
                    self.w_covmat_[:, :, d, k] = \
                        torch.diag(self.w_var_[:, d, k]).double()

                    self.xi_[non_nan_ids, d_bin, k] = \
                        torch.sqrt(torch.sum((self.X_[non_nan_ids, :]*\
                            torch.mm(self.w_covmat_[:, :, d, k],
                                     self.X_[non_nan_ids, :].T).T), 1) + \
                                   torch.mv(self.X_[non_nan_ids, :],
                                            self.w_mu_[:, d, k])**2)


    def init_R_mat(self, traj_probs=None, traj_probs_weight=None):
        """
        Initializes 'R_', using the stick-breaking construction.
    
        Parameters
        ----------
        traj_probs : torch.Tensor, shape ( K ), optional
            A priori probabilitiey of each of the K trajectories. Each element 
            must be >=0 and <= 1, and all elements must sum to one.
    
        traj_probs_weight : float, optional
            Value between 0 and 1 inclusive that controls how traj_probs are 
            combined with randomly generated trajectory probabilities using 
            stick-breaking: 
            traj_probs_weight*traj_probs + (1-traj_probs_weight)*random_probs.
        """
        # Draw a weight vector from the stick-breaking process
        tmp = torch.distributions.Beta(1, self.alpha_).sample((self.K_,))
        one_tmp = 1. - tmp
        vec = torch.Tensor([torch.prod(one_tmp[0:k])*\
                            tmp[k] for k in range(self.K_)]).double()
    
        if (traj_probs is None and traj_probs_weight is not None) or \
           (traj_probs_weight is None and traj_probs is not None):
            warnings.warn('Both traj_probs and traj_probs_weight \
            should be None or non-None')

        if traj_probs is not None and traj_probs_weight is not None:
            assert traj_probs_weight >= 0 and traj_probs_weight <= 1, \
                "Invalid traj_probs_weight"
            assert torch.isclose(torch.sum(torch.from_numpy(traj_probs)),
                                 torch.tensor(1.).double()), \
                "Invalid traj_probs"
            init_traj_probs = \
                traj_probs_weight*torch.from_numpy(traj_probs) + \
                (1-traj_probs_weight)*vec
        else:
            init_traj_probs = vec

        if torch.sum(init_traj_probs) < 0.95:
            warnings.warn("Initial trajectory probabilities sum to {}. \
            Alpha may be too high.".format(torch.sum(init_traj_probs)))

        self.R_ = torch.ones([self.N_, self.K_]).double()
        self.R_[:] = init_traj_probs
        self.sig_trajs_ = torch.max(self.R_, 0)[0] > self.prob_thresh_

    def augment_df_with_traj_info(self, df, gb_col=None):
        """Compute the probability that each data instance belongs to each of
        the 'k' clusters. Note that 'X' and 'Y' can be "new" data; that is,
        data that was not necessarily used to train the model.

        TODO: Test implementation of binary target accomodation

        Parameters
        ----------
        df : pandas DataFrame
            Input data frame to augment with trajectory info

        gb_col : str, optional
            df column to groupby. Should correspond to subject identifier.

        Returns
        -------
        df_aug : pandas DataFrame
            Corresponds to input data frame, but with extra columns: 'traj'
            indicates the most probable trajectory assignment, and 
            'traj_<num>', <num> indicates each of the trajectories and the 
            column values are the probabilities of assignment. 
        """        
        R = self.get_R_matrix(df, gb_col).numpy()
        N = df.shape[0]
        # Now augment the dataframe with trajectory info
        traj = []
        for i in range(N):
            traj.append(np.where(np.max(R[i, :]) == R[i, :])[0][0])
        df['traj'] = traj

        for s in np.where(self.sig_trajs_)[0]:
            df['traj_{}'.format(s)] = R[:, s]

        return df
        
            
    def compute_lower_bound(self):
        """Compute the variational lower bound

        NOTE: Currently not implemented.

        Returns
        -------
        lower_bound : float
            The variational lower bound
        """
        pass

    def get_traj_probs(self):
        """Computes the probability of each trajectory based on the marginal 
        of the assignment matrix, where the marginalization is computed over 
        individuals, not data points (individuals will in general have more
        or less data points than others). This function assumes that 'fit'
        has already been called.

        Returns
        -------
        traj_probs : array, shape ( K )
            Each element is the probability of the corresponding trajectory.
        """
        if torch.is_tensor(self.R_):
            traj_probs = \
                np.sum(self.R_.numpy()\
                       [self.group_first_index_.astype(bool), :], 0)/\
                np.sum(self.R_.numpy()\
                       [self.group_first_index_.astype(bool), :])
        else:            
            traj_probs = \
                np.sum(self.R_[self.group_first_index_.astype(bool), :], 0)/\
                np.sum(self.R_[self.group_first_index_.astype(bool), :])

        return traj_probs
    
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
            if torch.is_tensor(self.R_):
                traj.append(where(max(self.R_.numpy()[i, :]) == \
                                  self.R_.numpy()[i, :])[0][0])
            else:
                traj.append(where(max(self.R_[i, :]) == self.R_[i, :])[0][0])

        # Older models might not have self.df_ defined at this point. If not,
        # create it
        try:
            self.df_['traj'] = traj
        except AttributeError as error:
            self.df_ = pd.DataFrame()
            self.df_['traj'] = traj
            for ii, nn in enumerate(self.predictor_names_):
                if torch.is_tensor(self.X_):
                    self.df_[nn] = self.X_.numpy()[:, ii]
                else:
                    self.df_[nn] = self.X_[:, ii]
            for ii, nn in enumerate(self.target_names_):
                if torch.is_tensor(self.Y_):
                    self.df_[nn] = self.Y_.numpy()[:, ii]
                else:
                    self.df_[nn] = self.Y_[:, ii]
                    
        for s in np.where(self.sig_trajs_)[0]:
            if torch.is_tensor(self.R_):
                self.df_['traj_{}'.format(s)] = self.R_.numpy()[:, s]
            else:
                self.df_['traj_{}'.format(s)] = self.R_[:, s]
                
        return self.df_  

    def plot(self, x_axis, y_axis, x_label=None, y_label=None, which_trajs=None,
             show=True, min_traj_prob=0, max_traj_prob=1, traj_map=None,
             hide_traj_details=False, hide_scatter=False, traj_markers=None,
             traj_colors=None, fill_alpha=0.3):
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

        x_label : str, optional
            Label to display on x-axis. If none given, the variable name 
            specified with x_axis will be used

        y_label : str, optional
            Label to display on y-axis. If none given, the variable name 
            specified with y_axis will be used

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

        hide_traj_details : bool, optional
            If true, trajectory details (N and percentage of sample) will not 
            appear in the legend.

        hide_scatter : bool, optional
            If true, data scatter plot will not render

        traj_markers : list of strings, optional
            List of markers to use for each trajectory's line plot. Length of
            list should match number of trajectories to plot.

        traj_colors : list of strings, optional
            List of colors to use for each trajectory's line plot. Length of
            list should match number of trajectories to plot.

        fill_alpha : float, optional
            Value between 0 and 1 that controls opacity of each trajectorys 
            fill region (which indicates +\- 2 residual standard deviations 
            about the mean)
        """
        # Compute the probability vector for each trajectory
        traj_probs = self.get_traj_probs()
        
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
        if not hide_scatter:
            ax.scatter(df_traj[x_axis].values,
                       df_traj[y_axis].values,
                       edgecolor='k', color='None', alpha=0.1)

        if torch.is_tensor(self.lambda_a_):
            lambda_a = self.lambda_a_.numpy()
            lambda_b = self.lambda_b_.numpy()
        else:
            lambda_a = self.lambda_a_
            lambda_b = self.lambda_b_
            
        for (traj_inc, tt) in enumerate(traj_ids):
            if traj_probs[tt] >= min_traj_prob and \
               traj_probs[tt] <= max_traj_prob:
                
                ids_tmp = df_traj.traj.values == tt
                if not hide_scatter:
                    if traj_colors is not None:
                        color = traj_colors[traj_inc]
                    else:
                        color = cmap(traj_id_to_cmap_index[traj_map_[tt]])
                    ax.scatter(df_traj[ids_tmp][x_axis].values,
                               df_traj[ids_tmp][y_axis].values,
                               edgecolor='k',
                               color=color,
                               alpha=0.5)

                if self.gb_ is None:
                    n_traj = np.sum(df_traj.traj.values == tt)
                    perc_traj = 100*n_traj/df_traj.shape[0]
                else:
                    groupby_col = self.gb_.count().index.name                
                    n_traj = df_traj[df_traj.traj.values == tt].\
                        groupby(groupby_col).ngroups
                    perc_traj = 100*n_traj/self.gb_.ngroups

                co = self.w_mu_[:, target_index, tt]
                if self.target_type_[target_index] == 'gaussian':
                    std = np.sqrt(lambda_b[target_index][tt]/\
                                  lambda_a[target_index][tt])
                    y_tmp = np.dot(co, X_tmp.T)

                    if traj_colors is not None:
                        color = traj_colors[traj_inc]
                    else:
                        color = cmap(traj_id_to_cmap_index[traj_map_[tt]])
                    
                    ax.fill_between(x_dom, y_tmp-2*std, y_tmp+2*std,
                            color=color, alpha=fill_alpha)
                else:
                    # Target assumed binary
                    y_tmp = np.exp(np.dot(co, X_tmp.T))/\
                        (1 + np.exp(np.dot(co, X_tmp.T)))

                if hide_traj_details:
                    label = 'Traj {}'.format(traj_map_[tt])
                else:
                    label = 'Traj {} (N={}, {:.1f}%)'.\
                        format(traj_map_[tt], n_traj, perc_traj)

                marker = None
                if traj_markers is not None:
                    marker = traj_markers[traj_inc]

                if traj_colors is not None:
                    color = traj_colors[traj_inc]
                else:
                    color = cmap(traj_id_to_cmap_index[traj_map_[tt]])
                ax.plot(x_dom, y_tmp,
                        color=color,
                        linewidth=3,
                        label=label, marker=marker, ms=8, markevery=5)

        
        ax.set_xlabel(x_axis if x_label is None else x_label, fontsize=16)
        ax.set_ylabel(y_axis if y_label is None else y_label, fontsize=16)    
        plt.tight_layout()
        ax.legend(loc='upper right', framealpha=1)

        if show:
            plt.show()

        return ax    

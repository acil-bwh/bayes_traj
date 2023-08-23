#!/usr/bin/env python

from argparse import ArgumentParser
import torch
import pickle
import pandas as pd
import statsmodels.api as sm
import numpy as np
import copy, pdb
from provenance_tools.write_provenance_data import write_provenance_data

class PriorGenerator:
    """
    """
    def __init__(self, targets, preds, num_trajs=2,
                 min_num_trajs=None, max_num_trajs=None, alpha=None):
        """
        """
        assert np.issubdtype(type(num_trajs), np.integer) and \
            num_trajs > 0, "num_trajs not specified correctly"
        if min_num_trajs is not None:
            assert np.issubdtype(type(min_num_trajs), np.integer) and \
                min_num_trajs > 0, "min_num_trajs not specified correctly"
        if max_num_trajs is not None:
            assert np.issubdtype(type(max_num_trajs), np.integer) and \
                max_num_trajs > 0, "max_num_trajs not specified correctly"
        if min_num_trajs is not None and max_num_trajs is not None:
            assert min_num_trajs < max_num_trajs, \
                "Min num of trajs not less than mas num trajs"

        if min_num_trajs is not None and max_num_trajs is not None:
            self.min_num_trajs_ = min_num_trajs
            self.max_num_trajs_ = max_num_trajs
        else:
            self.min_num_trajs_ = num_trajs - 1
            self.max_num_trajs_ = num_trajs + 1
        
        self.targets_ = targets
        self.preds_ = preds
        
        self.groupby_col_ = None # Refers to input data, not input model
        self.gb_ = None
        
        self.mm_ = None

        self.df_traj_model_ = None # Corresponds to model data
        self.df_traj_data_ = None # Input data with traj assignmentes computed
                                  # internally here
        self.df_data_ = None # Input data (no traj assignments assumed)
        
        self.D_ = len(targets)
        self.M_ = len(preds)
        self.K_ = None
        self.N_ = None
        
        self.prior_info_ = {}
        self.prior_info_['w_mu0'] = {}
        self.prior_info_['w_var0'] = {}
        self.prior_info_['lambda_a0'] = {}
        self.prior_info_['lambda_b0'] = {}
        self.prior_info_['w_mu'] = None
        self.prior_info_['w_var'] = None
        self.prior_info_['lambda_a'] = None
        self.prior_info_['lambda_b'] = None
        self.prior_info_['v_a'] = None
        self.prior_info_['v_b'] = None
        self.prior_info_['traj_probs'] = None

        # The prior over the residual precision can get overwhelmed by the
        # data -- so much so that residual precision posteriors can wind up
        # in regimes that have near-zero mass in the prior. Given this, we
        # scale the prior params (essentially lowering the variance of the
        # prior) by an amount proportional to the number of subjects in the
        # data set. The following value is a heuristic and has worked in
        # practice.
        self.prec_prior_weight_ = 0.009

        if alpha is not None:
            self.prior_info_['alpha'] = alpha
        else:
            # Guesstimate of how big a data sample. Will be used to generate an
            # estimate of alpha. Will be overwritten if a data file has been
            # specified.
            N = 10000
            self.prior_info_['alpha'] = \
                (self.min_num_trajs_ + self.max_num_trajs_)/(2*np.log10(N))
        
        for tt in targets:
            self.prior_info_['lambda_a0'][tt] = 1
            self.prior_info_['lambda_b0'][tt] = 1
            #self.prior_info_['lambda_a'][tt] = None
            #self.prior_info_['lambda_b'][tt] = None            
            
            self.prior_info_['w_mu0'][tt] = {}
            self.prior_info_['w_var0'][tt] = {}
            #self.prior_info_['w_mu'][tt] = {}
            #self.prior_info_['w_var'][tt] = {}            
            for pp in self.preds_:
                self.prior_info_['w_mu0'][tt][pp] = 0
                self.prior_info_['w_var0'][tt][pp] = 5

                #self.prior_info_['w_mu'][tt][pp] = None
                #self.prior_info_['w_var'][tt][pp] = None

    def set_model(self, mm, model_trajs=None):
        """Sets input model. Once set, a data frame corresponding to the model
        will be computed and set as well.

        Parameters
        ----------
        mm : MultDPRegression instance
            Input model

        model_trajs : array, optional
            Array of integers indicating which trajectories to use for creating
            the prior. If None, all model trajectories with non-zero probability
            will be considered.
        """
        self.mm_ = mm

        # The following are independent of the targets and predictors
        self.K_ = mm.K_

        if model_trajs is not None:
            self.prior_info_['traj_probs'] = np.zeros(self.K_)
            self.prior_info_['traj_probs'][model_trajs] = \
                self.mm_.get_traj_probs()[model_trajs]/\
                np.sum(self.mm_.get_traj_probs()[model_trajs])            
        else:         
            self.prior_info_['traj_probs'] = self.mm_.get_traj_probs()
            
        self.prior_info_['v_a'] = self.mm_.v_a_
        self.prior_info_['v_b'] = self.mm_.v_b_
    
        self.df_traj_model_ = mm.to_df()
        self.update_df_traj_data()

        df_traj_data_computed = self.update_df_traj_data()
        self.init_per_traj_params()
        
    def set_data(self, df, groupby):
        """
        Parameters
        ----------
        df : pandas DataFrame
            Input data

        groupby : string
            Name of subject identifier column in dataframe
        """
        self.df_data_ = df
        self.N_ = df.shape[0]
        self.groupby_col_ = groupby
        if self.groupby_col_ is None:
            df_tmp = pd.DataFrame(index=range(self.N_))
            self.gb_ = df_tmp.groupby(df_tmp.index)
        else:
            self.gb_ = self.df_data_[[groupby]].groupby(groupby)
            
        df_traj_data_computed = self.update_df_traj_data()
        if df_traj_data_computed:
            self.init_per_traj_params()

    def init_per_traj_params(self):
        """Initializes per-trajectory param containers. 
        """
        self.prior_info_['lambda_a'] = {}
        self.prior_info_['lambda_b'] = {}
        self.prior_info_['w_mu'] = {}
        self.prior_info_['w_var'] = {}
        
        for m in self.preds_:            
            self.prior_info_['w_mu'][m] = {}
            self.prior_info_['w_var'][m] = {}

        for d in self.targets_:
            self.prior_info_['lambda_a'][d] = np.nan*np.ones(self.K_)
            self.prior_info_['lambda_b'][d] = np.nan*np.ones(self.K_)

            for m in self.preds_:
                self.prior_info_['w_mu'][m][d] = np.nan*np.ones(self.K_)
                self.prior_info_['w_var'][m][d] = np.nan*np.ones(self.K_) 
            
    def update_df_traj_data(self):
        """If possible, this function will update df_traj_data, which 
        corresponds to the input data (if specified) with trajectory assignments
        computed with an input model (if specified). 

        Returns
        -------
        compute_df_traj_data : bool
            True if df_traj_data has been computed. False otherwise.
        """
        compute_df_traj_data = False        
        if self.mm_ is not None and self.df_data_ is not None:
            if set(self.mm_.predictor_names_) <= set(self.df_data_.columns):
                for tt in self.mm_.target_names_:
                    if tt not in self.df_data_.columns:
                        self.df_data_[tt] = np.nan*np.ones(self.df_data_.shape)
                    else:
                        compute_df_traj_data = True
    
            if compute_df_traj_data:
                self.df_traj_data_ = \
                    self.mm_.augment_df_with_traj_info(self.df_data_,
                        self.groupby_col_)

        return compute_df_traj_data

    def traj_prior_info_from_df(self, target, traj):
        """This function will compute prior information (w_mu, w_var, 
        lambda_a, lambda_b) for a specific trajectory based on data in the 
        df_traj_data_ class member variable. The procedure is to perform linear
        regression on the data and to use the estimated predictor coefficients,
        standard errors, and residual information to set the desired quantities.

        Paramters
        ---------
        target : string
            Name of target variable for which to compute prior information.

        traj : int
            Prior information will be computed for those data instances 
            belonging to this trajectory.
      
        """
        assert self.df_traj_data_ is not None, "df_traj_data_ is None"

        indices = self.df_traj_data_['traj'].values == traj

        # What is the minimum number of points needed for regression?
        if np.sum(indices) == 0:
            return
        
        target_index = np.where(np.array(self.targets_) == target)[0][0]
        
        res_tmp = sm.OLS(self.df_traj_data_[indices][target],
                         self.df_traj_data_[indices][self.preds_],
                         missing='drop').fit()

        for (i, m) in enumerate(self.preds_):
            self.prior_info_['w_mu'][m][target][traj] = \
                res_tmp.params.values[i]
            self.prior_info_['w_var'][m][target][traj] = \
                res_tmp.HC0_se.values[i]**2

        gamma_mean = 1/np.var(res_tmp.resid.values)
        gamma_var = 0.005 # Heuristic

        self.prior_info_['lambda_b'][target][traj] = gamma_mean/gamma_var
        self.prior_info_['lambda_a'][target][traj] = gamma_mean**2/gamma_var

    def traj_prior_info_from_model(self, target, traj):
        """This function will retrieve prior information (w_mu, w_var, 
        lambda_a, lambda_b) for a specific trajectory based on a previously fit
        model. Assumes that the model predictors and the predictors for which 
        to compute priors are the same. Also assumes that 'target' is in the
        model.

        Paramters
        ---------
        target : string
            Name of target variable for which to get prior information.

        traj : int
            Prior information will be retrieved for those data instances 
            belonging to this trajectory.    
        """
        assert self.mm_ is not None, \
            "Trying to set prior info from model, but no model specified"

        target_index = \
            np.where(np.array(self.mm_.target_names_) == target)[0][0]

        self.prior_info_['lambda_a'][target][traj] = \
            self.mm_.lambda_a_[target_index][traj]
        self.prior_info_['lambda_b'][target][traj] = \
            self.mm_.lambda_b_[target_index][traj]
        
        for m in self.preds_:
            pred_index = \
                np.where(np.array(self.mm_.predictor_names_) == m)[0][0]

            self.prior_info_['w_mu'][m][target][traj] = \
                self.mm_.w_mu_[pred_index, target_index, traj]
            self.prior_info_['w_var'][m][target][traj] = \
                self.mm_.w_var_[pred_index, target_index, traj]            
        
    def prior_info_from_df(self, target):
        """Computes w_mu0, w_var0, lambda_a0, lambda_b0

        Parameters
        ----------
        target : string
            Name of target variable for which to compute prior information
        """
        # If w_mu, w_var, lambda_a, and lambda_b have been set, then we can
        # derive overall prior info from these. Otherwise, we will get prior
        # info from the data alone.
        # TODO

        # Per-traj prior info has not been set:
        res_tmp = sm.OLS(self.df_data_[target],
                         self.df_data_[self.preds_], missing='drop').fit()

        num_trajs = np.array([self.min_num_trajs_, self.max_num_trajs_])

        # 'precs' will be a 2D vector expressing high and low estimates for the
        # mean precision (assuming a range of possible trajectory subgroups). From
        # this, we will estimate a variance for the gamma distribution wide enough
        # to cover these high and low estimates.
        precs = num_trajs/np.var(res_tmp.resid.values)
        gamma_mean = np.mean(precs)
        gamma_var = ((np.max(precs) - np.min(precs))/4)**2

        # The prior over the residual precision can get overwhelmed by the
        # data -- so much so that residual precision posteriors can wind up
        # in regimes that have near-zero mass in the prior. Given this, we
        # scale the prior params (essentially lowering the variance of the
        # prior) by an amount proportional to the number of subjects in the
        # data set. The following value is a heuristic and has worked in
        # practice.
        self.prior_info_['lambda_b0'][target] = (gamma_mean/gamma_var)*\
            self.prec_prior_weight_*(self.gb_.ngroups \
                                     if self.gb_ is not None else self.N_)
        self.prior_info_['lambda_a0'][target] = (gamma_mean**2/gamma_var)*\
            self.prec_prior_weight_*(self.gb_.ngroups \
                                     if self.gb_ is not None else self.N_)

        # Use a fudge factor to arrive at a reasonable value for w_var0 values. This
        # value has been empirically established and appears to give reasonable
        # results in practice. w_var0 will be estimated based on the sample size,
        # SEs from the regression, and the fudge factor.
        fudge_factor = 0.005 
        vars = fudge_factor*self.df_data_.shape[0]*res_tmp.HC0_se.values**2 

        for (i, m) in enumerate(self.preds_):
            self.prior_info_['w_mu0'][target][m] = res_tmp.params.values[i]
            self.prior_info_['w_var0'][target][m] = vars[i]        
    
    def prior_info_from_model(self, target):
        """
        """
        assert target in self.mm_.target_names_, \
            "Specified target is not among model targets"
        assert set(self.preds_) == set(self.mm_.predictor_names_), \
            "Specified predictors differ from model predictors"

        target_index = np.where(np.array(self.mm_.target_names_) == \
                                target)[0][0]

        if torch.is_tensor(self.mm_.w_mu_):
            w_mu = self.mm_.w_mu_.numpy()
            w_var = self.mm_.w_var_.numpy()            
        else:
            w_mu = self.mm_.w_mu_
            w_var = self.mm_.w_var_            

        if torch.is_tensor(self.mm_.sig_trajs_):
            sig_trajs = self.mm_.sig_trajs_.numpy()
        else:
            sig_trajs = self.mm_.sig_trajs_
            
        probs = self.prior_info_['traj_probs']
        for m in self.preds_:
            pred_index = \
                np.where(np.array(self.mm_.predictor_names_) == m)[0][0]

            # w_mu0 is a weighted combination of the trajectory means
            self.prior_info_['w_mu0'][target][m] = \
                np.dot(probs, w_mu[pred_index, target_index, :])

            if np.sum(sig_trajs) > 1:
                # Normally, the variance of a r.v. that is the sum of other
                # normally distributed r.v.s has a variance equal to a weighted
                # sum of the constituent r.v.s (where the weight is squared).
                # Howver, it is known that variational inference underestimates
                # variances. Therefore, if we set w_var0 to the weighted sum of
                # the variances, it too would be underestimated. Instead, we
                # heuristically compute w_var0 by considering the max distance
                # between the w_mu0 and each trajectory mean. We assume that the
                # mean +/- 2.5*sig will cover each of the observed trajectory
                # values.
                self.prior_info_['w_var0'][target][m] = \
                    (np.max(np.abs(w_mu[pred_index, target_index, sig_trajs] - \
                                self.prior_info_['w_mu0'][target][m]))/2.5)**2
            else:
                # If we're here, there is only 1 trajectory 
                self.prior_info_['w_var0'][target][m] = \
                    w_var[pred_index, target_index, sig_trajs][0]

        if torch.is_tensor(self.mm_.lambda_a_):
            lambda_a = self.mm_.lambda_a_.numpy()
            lambda_b = self.mm_.lambda_b_.numpy()
        else:
            lambda_a = self.mm_.lambda_a_
            lambda_b = self.mm_.lambda_b_
            
        if np.sum(sig_trajs) > 1:
            gamma_mean = 0
            gamma_var = 0

            gamma_means = lambda_a[target_index]/lambda_b[target_index]
            gamma_mean = np.dot(probs, gamma_means)
            # See argument above about estimating variances
            gamma_var = \
                (np.max(np.abs(gamma_means[np.where(sig_trajs)[0]] - \
                               gamma_mean))/2.5)**2
            if gamma_var > 0:
                self.prior_info_['lambda_b0'][target] = gamma_mean/gamma_var
                self.prior_info_['lambda_a0'][target] = gamma_mean**2/gamma_var
            else:
                # If we're here, all trajectories have the exact same residual
                # variance, so gamma_var = 0. In this case, we can just set the
                # prior to be equal to the residual variance of any one
                # trajectory (as they are all the same)
                self.prior_info_['lambda_b0'][target] = \
                    lambda_b[target_index][0]
                self.prior_info_['lambda_a0'][target] = \
                    lambda_a[target_index][0]
        else:
            # If we're here, there is only one trajectory
            self.prior_info_['lambda_a0'][target] = \
                lambda_a[target_index][sig_trajs][0]
            self.prior_info_['lambda_b0'][target] = \
                lambda_b[target_index][sig_trajs][0]
            
    def compute_prior_info(self):
        """
        """
        #prior_info_from_model
        #prior_info_from_df
        #traj_prior_info_from_model
        #traj_prior_info_from_df
        assert self.mm_ is not None or self.df_data_ is not None, \
            "Cannot compute prior info without data or a model"
        
        for tt in self.targets_:
            if self.mm_ is not None:
                if set(self.preds_) == set(self.mm_.predictor_names_):
                    if tt in self.mm_.target_names_:
                        self.prior_info_from_model(tt)
                        for kk in range(self.K_):
                            self.traj_prior_info_from_model(tt, kk)
                    elif self.df_data_ is not None:
                        self.prior_info_from_df(tt)
                        if self.df_traj_data_ is not None:
                            for kk in range(self.K_):
                                self.traj_prior_info_from_df(tt, kk)
                    else:
                        raise RuntimeError('{} is not in model'.format(tt))    
                else:
                    # Model is defined, but predictors differ. In this case,
                    # everything is retrieved from data
                    if self.df_data_ is not None:
                        self.prior_info_from_df(tt)
                        if self.df_traj_data_ is not None:
                            for kk in range(self.K_):
                                self.traj_prior_info_from_df(tt, kk)
            else:
                # No model, but we have data. Because we have no model, we also
                # won't have df_traj_data_, so we won't be able to get per-traj
                # information
                self.prior_info_from_df(tt)

#-------------------------               
# END OF CLASS DEFINITION
#-------------------------        
    
def main():        
    desc = """Generates a pickled file containing Bayesian trajectory prior 
    information"""
    
    parser = ArgumentParser(description=desc)
    parser.add_argument('--preds', help='Comma-separated list of predictor names',
        dest='preds', type=str, default=None)
    parser.add_argument('--targets', help='Comma-separated list of target names',
        dest='targets', type=str, default=None)
    parser.add_argument('--out_file', help='Output (pickle) file that will \
        contain the prior', dest='out_file', type=str, default=None)
    parser.add_argument('--tar_resid', help='Use this flag to specify the residual \
        precision mean and variance for the corresponding target value. Specify as \
        a comma-separated tuple: target_name,mean,var. Note that precision is the \
        inverse of the variance. Only applies to continuous targets', type=str,
        default=None, action='append', nargs='+')
    parser.add_argument('--coef', help='Coefficient prior for a specified \
        target and predictor. Specify as a comma-separated tuple: \
        target_name,predictor_name,mean,std', type=str,
        default=None, action='append', nargs='+')
    parser.add_argument('--coef_std', help='Coefficient prior standard deviation \
        for a specified target and predictor. Specify as a comma-separated tuple: \
        target_name,predictor_name,std', type=str, default=None,
        action='append', nargs='+')
    parser.add_argument('--in_data', help='If a data file is specified, it will be \
        read in and used to set reasonable prior values using regression. It \
        is assumed that the file contains data columns with names corresponding \
        to the predictor and target names specified on the command line.',
        type=str, default=None)
    parser.add_argument('--num_trajs', help='Rough estimate of the number of \
        trajectories expected in the data set. Can be specified as a single \
        value or as a dash-separated range, such as 4-6. If a single value is \
        specified, a range will be assumed as -1 to +1 the specified value.',
        type=str, default='3')
    parser.add_argument('--model', help='Pickled bayes_traj model that \
        has been fit to data and from which information will be extracted to \
        produce an updated prior file', type=str, default=None)
    parser.add_argument('--model_trajs', help='Comma-separated list of integers \
        indicating which trajectories to use from the specified model. If a model \
        is not specified, the values specified with this flag will be ignored. If \
        a model is specified, and specific trajectories are not specified with \
        this flag, then all trajectories will be used to inform the prior', \
        default=None)
    parser.add_argument('--groupby', help='Column name in input data file \
        indicating those data instances that must be in the same trajectory. This \
        is typically a subject identifier (e.g. in the case of a longitudinal data \
        set).', dest='groupby', metavar='<string>', default=None)
    parser.add_argument('--alpha', help='Dirichlet process scaling parameter. \
        Higher values indicate belief that more trajectoreis are present. \
        Must be a positive real value if specified.', dest='alpha', \
        type=float, metavar=float, default=None)
    
    op = parser.parse_args()
    
    preds = op.preds.split(',')
    targets = op.targets.split(',')
            
    if op.alpha is not None:
        assert op.alpha > 0, \
            "alpha  must be a positive real value"
        
    pg = PriorGenerator(targets, preds, alpha=op.alpha)
    
    #---------------------------------------------------------------------------
    # Set the number of trajs
    #---------------------------------------------------------------------------
    num_trajs = np.zeros(2, dtype='float')
    tmp = op.num_trajs.split('-')

    if len(tmp) > 1:
        pg.min_num_trajs_ = float(op.num_trajs.split('-')[0])
        pg.max_num_trajs_ = float(op.num_trajs.split('-')[1])        
    else:
        pg.min_num_trajs_ = np.max([0.001, float(tmp[0]) - 1])
        pg.max_num_trajs_ = float(tmp[0]) + 1
        
    #---------------------------------------------------------------------------
    # Read in and process data and models as availabe
    #---------------------------------------------------------------------------
    if op.model is not None:
        with open(op.model, 'rb') as f:
            print("Reading model...")
            mm = pickle.load(f)['MultDPRegression']
            if op.model_trajs is not None:
                pg.set_model(mm, \
                    np.array(op.model_trajs.split(','), dtype='int'))
            else:
                pg.set_model(mm)                    

    if op.in_data is not None:
        print("Reading data...")
        pg.set_data(pd.read_csv(op.in_data), op.groupby)

    pg.compute_prior_info()        
    prior_info = copy.deepcopy(pg.prior_info_)
    
    #---------------------------------------------------------------------------
    # Override prior settings with user-specified preferences
    #---------------------------------------------------------------------------
    if op.tar_resid is not None:
        for i in range(len(op.tar_resid)):
            tt = op.tar_resid[i][0].split(',')[0]
            mean_tmp = float(op.tar_resid[i][0].split(',')[1])
            var_tmp = float(op.tar_resid[i][0].split(',')[2])        
            assert tt in targets, "{} not among specified targets".format(tt)
                    
            prior_info['lambda_b0'][tt] = mean_tmp/var_tmp
            prior_info['lambda_a0'][tt] = (mean_tmp**2)/var_tmp
    
    if op.coef is not None:
        for i in range(len(op.coef)):
            tt = op.coef[i][0].split(',')[0]
            pp = op.coef[i][0].split(',')[1]
            m = float(op.coef[i][0].split(',')[2])
            s = float(op.coef[i][0].split(',')[3])
    
            assert tt in targets, "{} not among specified targets".format(tt)
            assert pp in preds, "{} not among specified predictors".format(pp)        
    
            prior_info['w_mu0'][tt][pp] = m
            prior_info['w_var0'][tt][pp] = s**2
    
    if op.coef_std is not None:
        for i in range(len(op.coef_std)):
            tt = op.coef_std[i][0].split(',')[0]
            pp = op.coef_std[i][0].split(',')[1]
            s = float(op.coef_std[i][0].split(',')[2])
    
            assert tt in targets, "{} not among specified targets".format(tt)
            assert pp in preds, "{} not among specified predictors".format(pp)        
    
            prior_info['w_var0'][tt][pp] = s**2

    #---------------------------------------------------------------------------
    # Summarize prior info and save to file
    #---------------------------------------------------------------------------        
    print('---------- Prior Info ----------')
    print('alpha: {:.2e}'.format(prior_info['alpha']))        
    for tt in targets:
        print(" ")
        if prior_info['lambda_a0'][tt] is not None:
            prec_mean = prior_info['lambda_a0'][tt]/\
                prior_info['lambda_b0'][tt]
            prec_var = prior_info['lambda_a0'][tt]/\
                (prior_info['lambda_b0'][tt]**2)
            print("{} residual (precision mean, precision variance): \
            ({:.2e}, {:.2e})".format(tt, prec_mean, prec_var))
        for pp in preds:
            tmp_mean = prior_info['w_mu0'][tt][pp]
            tmp_std = np.sqrt(prior_info['w_var0'][tt][pp])
            print("{} {} (mean, std): ({:.2e}, {:.2e})".\
                  format(tt, pp, tmp_mean, tmp_std))

    if op.out_file is not None:                    
        pickle.dump(prior_info, open(op.out_file, 'wb'))
        desc = """ """
        write_provenance_data(op.out_file, generator_args=op, desc=desc,
                              module_name='bayes_traj')
        
if __name__ == "__main__":
    main()
    

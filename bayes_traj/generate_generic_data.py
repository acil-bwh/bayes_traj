#!/usr/bin/env python

import numpy as np
import pandas as pd
import pdb
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from provenance_tools.write_provenance_data import write_provenance_data

def main():
    desc = """Generates an arbitrary number of quadratic trajectories. Useful for 
    testing purposes. The x-axis is referred to as 'age' throughout. The context
    of this script mimics a longitudinal study in which individuals are 
    recruited and then followed for a specified number of visits, spread apart by
    a specified number of time."""
    
    parser = ArgumentParser(description=desc)
    parser.add_argument('--traj_params', help='Tuple specifying the trajectory \
        shape, residual noise, and number of subjects. Can be used multiple times \
        Specify as: <intercept,age,age^2,resid_std,num>', type=str, default=None,
        action='append', nargs='+')                    
    parser.add_argument('--enrollment', help='Comma-separated tuple: min age, max \
        age. This specifies the range of randomly generated ages correpsonding to \
        a synthetically generated individuals baseline age', dest='enrollment',
        default=None)
    parser.add_argument('--visit_span', help='Num years between successive visits',
        dest='visit_span', type=float, default=None)
    parser.add_argument('--max_age', help='No subject age will be above this amount',
        dest='max_age', type=float, default=95)
    parser.add_argument('--num_visits', help='Number of longitudinal visits \
        per individual. Note that the actual number for an individual may be less \
        than this if the generated age for a visit is greater than max_age',
        dest='num_visits', type=int, default=1)
    parser.add_argument('--out_file', help='Output data file name. Columns \
        include: intercept, x, x^2, y, id, data_names, traj.', default=None)
    
    op = parser.parse_args()
    
    enrollment_min = int(op.enrollment.split(',')[0])
    enrollment_max = int(op.enrollment.split(',')[1])
    
    if len(op.traj_params) <= 10:
        cmap = plt.cm.get_cmap('tab10')
    else:
        cmap = plt.cm.get_cmap('tab20')
    
    df_out = pd.DataFrame()
    
    subj_inc = 0
    traj_inc = 0
    
    plt.figure(figsize=(8, 8))
    print("Generating trajectories...")
    for tp in op.traj_params:
        traj_inc += 1
    
        traj_dim_cos = tp[0].split('><')
        D = len(traj_dim_cos)
        num_in_traj = int(traj_dim_cos[0].strip('>').split(',')[4])
    
        for s in range(num_in_traj):
            subj_inc += 1
            df_tmp = pd.DataFrame()
            
            enrollment_age = np.random.uniform(enrollment_min, enrollment_max)
    
            # Get age range
            ages_tmp = np.linspace(enrollment_age, \
                                   enrollment_age + op.visit_span*(op.num_visits-1),
                                   op.num_visits)
            ids_tmp = ages_tmp <= op.max_age
            ages = ages_tmp[ids_tmp]
    
            
            df_tmp['intercept'] = np.ones(ages.shape[0])
            df_tmp['age'] = ages
            df_tmp['age^2'] = ages**2
            df_tmp['id'] = str(subj_inc)
            df_tmp['data_names'] = [str(subj_inc) + '_' + \
                                    str(j) for j in range(ages.shape[0])]
            df_tmp['traj_gt'] = traj_inc
    
            for i, dd in enumerate(traj_dim_cos):
                cos = np.array(dd.strip('<').strip('>').split(',')[0:3],
                               dtype=float)
                traj_resid_std = float(dd.strip('<').strip('>').split(',')[3])
                
                y = np.dot(cos, df_tmp[['intercept', 'age', 'age^2']].values.T) + \
                    traj_resid_std*np.random.randn(ages.shape[0])
    
                df_tmp['y{}'.format(i+1)] = y
    
            df_out = pd.concat([df_out, df_tmp])        
    
    if traj_inc <= 10:
        cmap = plt.cm.get_cmap('tab10')
    else:
        cmap = plt.cm.get_cmap('tab20')
    
    fig, axs = plt.subplots(1, D, figsize=(6*D, 6))
    for d in range(D):
        for tt in range(traj_inc):
            ids = df_out.traj_gt.values == tt+1
            num_in_traj = df_out[ids].groupby('id').ngroups
            if D > 1:
                axs[d].scatter(df_out[ids].age.values,
                               df_out[ids]['y{}'.format(d+1)].values, edgecolor='k',
                               color=cmap(tt), alpha=0.5,
                               label='Traj {} (N={})'.format(tt+1, num_in_traj))
                axs[d].set_xlabel('Age')
                axs[d].set_ylabel('y{}'.format(d+1))
                axs[d].legend()
            else:
                axs.scatter(df_out[ids].age.values,
                            df_out[ids]['y{}'.format(d+1)].values, edgecolor='k',
                            color=cmap(tt), alpha=0.5,
                            label='Traj {} (N={})'.format(tt+1, num_in_traj))
                axs.set_xlabel('Age')
                axs.set_ylabel('y{}'.format(d+1))
                axs.legend()
    plt.show()
    
    if op.out_file is not None:
        print("Writing to file...")
        df_out.to_csv(op.out_file, index=False)
        write_provenance_data(op.out_file, generator_args=op,
                              module_name='bayes_traj')
    
    print("DONE.")    


if __name__ == "__main__":
    main()
            

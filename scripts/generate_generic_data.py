import numpy as np
import pandas as pd
import pdb
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from provenance_tools.provenance_tracker import write_provenance_data

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

print("Generating trajectories...")
for tp in op.traj_params:
    traj_inc += 1
    traj_cos = np.array(tp[0].strip('<').strip('>').split(',')[0:3],
                        dtype=float)
    num_in_traj = int(tp[0].strip('<').strip('>').split(',')[-1])
    traj_resid_std = float(tp[0].strip('<').strip('>').split(',')[-2])

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

        y = np.dot(traj_cos, df_tmp[['intercept', 'age', 'age^2']].values.T) + \
            traj_resid_std*np.random.randn(ages.shape[0])

        df_tmp['y'] = y
        df_tmp['traj'] = traj_inc
        
        df_out = pd.concat([df_out, df_tmp])        
        

    plt.scatter(df_out[df_out.traj.values==traj_inc].age.values,
                df_out[df_out.traj.values==traj_inc].y.values,
                color=cmap(traj_inc), label='Traj {} (N={})'.\
                format(traj_inc, num_in_traj))

plt.legend()
plt.xlabel('Age')
plt.ylabel('y')
plt.show()

if op.out_file is not None:
    print("Writing to file...")
    df_out.to_csv(op.out_file, index=False)
    write_provenance_data(op.out_file, generator_args=op)

print("DONE.")    

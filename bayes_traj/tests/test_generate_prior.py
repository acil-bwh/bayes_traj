from bayes_traj.generate_prior import PriorGenerator
import numpy as np
import pandas as pd
from bayes_traj.utils import *
import pdb, os

def test_foo():
    data_file_name = os.path.split(os.path.realpath(__file__))[0] + \
        '/../resources/data/trajectory_data_1.csv'
    df = pd.read_csv(data_file_name)

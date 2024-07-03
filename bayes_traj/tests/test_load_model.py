from bayes_traj.utils import *
import pickle, torch
import numpy as np
import os
import pdb
import bayes_traj
import pandas as pd
import warnings
from bayes_traj.load_model import load_model

def test_load_model_pickle():
    model_file_name = os.path.split(os.path.realpath(__file__))[0] + \
        '/../resources/models/model_1.p'
        
    model = load_model(model_file_name)
    
    assert isinstance(model, bayes_traj.mult_dp_regression.MultDPRegression), \
        "Model not read correctly"


def test_load_model_torch():
    model_file_name = os.path.split(os.path.realpath(__file__))[0] + \
        '/../resources/models/pyro_model_1.pt'
        
    model = load_model(model_file_name)
    
    assert isinstance(model, bayes_traj.mult_pyro.MultPyro), \
        "Model not read correctly"




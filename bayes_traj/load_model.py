import pickle
import numpy as np
import torch
import pdb
import bayes_traj
from bayes_traj import *
import pandas as pd
import sys
from bayes_traj.base_model import *

def load_model(file_path):
    """
    Load a model from the given file path. Determines the file type by 
    inspecting the file header and loads using appropriate method.

    Parameters
    ----------
        file_path : str
            Path to the model file.

    Returns
    -------
        model : object
            Instance of MultDPRegression or MultPyro
        
    """
    # Try to load with pickle first
    try:
        with open(file_path, 'rb') as f:
            model = pickle.load(f)['MultDPRegression']
        print("Model loaded with pickle")
        
        return model
    except (pickle.UnpicklingError, AttributeError, EOFError, ImportError,
            IndexError):
        print("Pickle load failed. Trying torch...")

    # If pickle fails, try torch
    try:
        model = torch.load(file_path)
        print("Model loaded with torch")
        
        return model
    except Exception as e:
        print(f"Torch load failed: {e}")

    raise ValueError("Failed to load the model with both pickle and torch")
    


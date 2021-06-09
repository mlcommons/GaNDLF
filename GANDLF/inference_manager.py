
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
from torch.utils.data.dataset import Dataset
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import os
import random
import torchio
from torchio.transforms import *
from torchio import Image, Subject
from sklearn.model_selection import KFold
from shutil import copyfile
import time
import sys
import ast 
import pickle
from pathlib import Path
import subprocess


# from GANDLF.data.ImagesFromDataFrame import ImagesFromDataFrame
from GANDLF.inference_loop import *

def InferenceManager(dataframe, headers, outputDir, parameters, device):
    '''
    This function takes in a dataframe, with some other parameters and performs the inference
    '''
    # get the indeces for kfold splitting
    inferenceData_full = dataframe
    
    # # initialize parameters for inference
    if not("weights" in parameters):
        parameters["weights"] = None # no need for loss weights for inference
    
    inference_loop(
            inferenceDataFromPickle=inferenceData_full, 
            headers=headers, 
            outputDir=outputDir,
            device=device, 
            parameters=parameters
        )

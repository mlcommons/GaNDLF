
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
from GANDLF.inference_loop import inferenceLoop

def InferenceManager(dataframe, headers, outputDir, parameters, device):
    '''
    This function takes in a dataframe, with some other parameters and performs the inference
    '''
    # get the indeces for kfold splitting
    inferenceData_full = dataframe
    # inference_indeces_full = list(inferenceData_full.index.values)

    if parameters['modality'] == 'rad':
	    inferenceLoopRad(inferenceDataFromPickle=inferenceData_full, headers=headers, outputDir=outputDir,
	                     device=device, parameters=parameters)
    elif parameters['modality'] == 'path':
	    inferenceLoopPath(inferenceDataFromPickle=inferenceData_full, headers=headers, outputDir=outputDir,
	                      device=device, parameters=parameters)
    else:
    	print('Modality should be on of rad or path. Please set the correct on in the config file.')
        sys.exit(0)


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

def InferenceManager(dataframe, channelHeaders, labelHeader, outputDir, parameters, device):
    '''
    This function takes in a dataframe, with some other parameters and performs the inference
    '''
    # get the indeces for kfold splitting
    inferenceData_full = dataframe
    # inference_indeces_full = list(inferenceData_full.index.values)

    inferenceLoop(inferenceDataFromPickle = inferenceData_full, 
    channelHeaders = channelHeaders, labelHeader = labelHeader, outputDir = outputDir, device = device, parameters = parameters)
        
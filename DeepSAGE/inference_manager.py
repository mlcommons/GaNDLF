
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


# from DeepSAGE.data.ImagesFromDataFrame import ImagesFromDataFrame
from DeepSAGE.inference_loop import inferenceLoop

# This function takes in a dataframe, with some other parameters and returns the dataloader
def InferenceManager(dataframe, augmentations, psize, channelHeaders, labelHeader, model_parameters_file, outputDir,batch_size, which_loss, n_classes, base_filters, n_channels, which_model, parallel_compute_command, device):

    # get the indeces for kfold splitting
    inferenceData_full = dataframe
    inference_indeces_full = list(inferenceData_full.index.values)

    inferenceLoop(trainingDataFromPickle = trainingData, batch_size = batch_size, learning_rate = learning_rate, 
            which_loss = which_loss, opt = opt, save_best = save_best, n_classes = n_classes,
            base_filters = base_filters, n_channels = n_channels, which_model = which_model, psize = psize, 
            channelHeaders = channelHeaders, labelHeader = labelHeader, augmentations = augmentations, outputDir = currentOutputFolder, device = device)
        
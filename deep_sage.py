
#!usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function, division
import os
os.environ['TORCHIO_HIDE_CITATION_PROMPT'] = '1' # hides torchio citation request, see https://github.com/fepegar/torchio/issues/235

import argparse
import sys
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import gc
from torchsummary import summary
import ast 
from pathlib import Path
from sklearn.model_selection import KFold
import pickle
import pkg_resources
import torchio
from datetime import date

from DeepSAGE.training_manager import TrainingManager
from DeepSAGE.inference_manage import InferenceManager 

copyrightMessage = 'Contact: software@cbica.upenn.edu\n\n' + 'This program is NOT FDA/CE approved and NOT intended for clinical use.\nCopyright (c) ' + str(date.today().year) + ' University of Pennsylvania. All rights reserved.' 
parser = argparse.ArgumentParser(prog='DeepSage', formatter_class=argparse.RawTextHelpFormatter, description = "3D Image Semantic Segmentation using Deep Learning.\n\n" + copyrightMessage)
parser.add_argument('-config', type=str, help = 'The configuration file (contains all the information related to the training/inference session), this is read from \'modelDir\' during inference', required=False)
parser.add_argument('-data', type=str, help = 'Data csv file that is used for training/inference', required=True)
parser.add_argument('-output', type=str, help = 'Output directory to save intermediate files and model weights', required=True)
parser.add_argument('-train', default=1, type=int, help = '1 == training and 0 == inference; for 0, there needs to be a compatible model saved in \'-md\'', required=False)
parser.add_argument('-modelDir', type=str, help = 'The pre-trained model directory that is used for inference, needs to contain the configuration', required=False)
parser.add_argument('-device', default=0, type=str, help = 'Device to perform requested session on (\'-1\' == cpu and everything else denotes GPU ID)', required=True) 
#parser.add_argument('-v', '--version', action='version', 
#      version = format(parser.prog) + ' v' + pkg_resources.require('deep_seg')[0].version + '\n\n' + copyrightMessage, help="Show program's version number and exit.")
                            
args = parser.parse_args()

file_data_full = args.data
model_parameters = args.config
dev = args.device
model_path = args.output
mode = args.train
if '-1' in dev:
    dev = 'cpu'

if mode == 0: # inference mode
    pretrainedModelPath = args.modelDir

# safe directory creation
Path(model_path).mkdir(parents=True, exist_ok=True)

df_model = pd.read_csv(model_parameters, sep=' = ', names=['param_name', 'param_value'],
                         comment='#', skip_blank_lines=True,
                         engine='python').fillna(' ')

#Read the parameters as a dictionary so that we can access everything by the name and so when we add some extra parameters we dont have to worry about the indexing
params = {}
for j in range(df_model.shape[0]):
    params[df_model.iloc[j, 0]] = df_model.iloc[j, 1]

# Extrating the training parameters from the dictionary
num_epochs = int(params['num_epochs'])
batch_size = int(params['batch_size'])
learning_rate = int(params['learning_rate'])
which_loss = str(params['loss_function'])
opt = str(params['opt'])
save_best = int(params['save_best'])
augmentations = ast.literal_eval(str(params['data_augmentation']))

# Extracting the model parameters from the dictionary
n_classes = int(params['largestLabelToPredict'])
base_filters = int(params['base_filters'])
n_channels = int(params['numberOfInputChannels'])
# model_path = str(params['folderForOutput'])
which_model = str(params['modelName'])
kfolds = int(params['kcross_validation'])
psize = params['patch_size']
psize = ast.literal_eval(psize) 
psize = np.array(psize)
parallel_compute_command = ''
if 'parallel_compute_command' in params:
    parallel_compute_command = params['parallel_compute_command']
    parallel_compute_command = parallel_compute_command.replace('\'', '')
    parallel_compute_command = parallel_compute_command.replace('\"', '')

## read training dataset into data frame
data_full = pd.read_csv(file_data_full)
# shuffle the data - this is a useful level of randomization for the training process
data_full=data_full.sample(frac=1).reset_index(drop=True)

# find actual header locations for input channel and label
# the user might put the label first and the channels afterwards 
# or might do it completely randomly
channelHeaders = []
for col in data_full.columns: 
    # add appropriate headers to read here, as needed
    if ('Channel' in col) or ('Modality' in col) or ('Image' in col):
        channelHeaders.append(data_full.columns.get_loc(col))
    elif ('Label' in col) or ('Mask' in col) or ('Segmentation' in col):
        labelHeader = data_full.columns.get_loc(col)
    elif ('SubjectID' in col) or ('PatientName' in col):
        subjectIDHeader = data_full.columns.get_loc(col)

if mode != 0: # training mode
    TrainingManager(dataframe = data_full, augmentations = augmentations, kfolds = kfolds, psize = psize, channelHeaders = channelHeaders, labelHeader = labelHeader, model_parameters_file = model_parameters, outputDir = model_path, num_epochs = num_epochs, batch_size = batch_size, learning_rate = learning_rate, which_loss = which_loss, opt = opt, save_best = save_best, n_classes = n_classes, base_filters = base_filters, n_channels = n_channels, which_model = which_model, parallel_compute_command = parallel_compute_command, device = args.device)
else:
    InferenceManager(dataframe = data_full, augmentations = augmentations, psize = psize, channelHeaders = channelHeaders, labelHeader = labelHeader, model_parameters_file = model_parameters, outputDir = pretrainedModelPath, batch_size = batch_size, which_loss = which_loss, n_classes = n_classes, base_filters = base_filters, n_channels = n_channels, which_model = which_model, device = args.device)
    

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
from GANDLF.training_loop import trainingLoop

# This function takes in a dataframe, with some other parameters and returns the dataloader
def TrainingManager(dataframe, channelHeaders, labelHeader, outputDir, parameters, device):

    # check for single fold training
    singleFoldTraining = False
    if parameters['kfolds'] < 0: # if the user wants a single fold training
      parameters['kfolds'] = abs(parameters['kfolds'])
      singleFoldTraining = True

    kf = KFold(n_splits=parameters['kfolds']) # initialize the kfold structure

    currentFold = 0

    # get the indeces for kfold splitting
    trainingData_full = dataframe
    training_indeces_full = list(trainingData_full.index.values)

    # start the kFold train
    for train_index, test_index in kf.split(training_indeces_full):

      # the output of the current fold is only needed if multi-fold training is happening
      if singleFoldTraining:
        currentOutputFolder = outputDir
      else:
        currentOutputFolder = os.path.join(outputDir, str(currentFold))
        Path(currentOutputFolder).mkdir(parents=True, exist_ok=True)

      trainingData = trainingData_full.iloc[train_index]
      validationData = trainingData_full.iloc[test_index]

      # save the current model configuration as a sanity check
      currentModelConfigPickle = os.path.join(currentOutputFolder, 'parameters.pkl')
      with open(currentModelConfigPickle, 'wb') as handle:
        pickle.dump(parameters, handle, protocol=pickle.HIGHEST_PROTOCOL)

      parallel_compute_command = parameters['parallel_compute_command']

      if (not parallel_compute_command) or (singleFoldTraining): # parallel_compute_command is an empty string, thus no parallel computing requested
        trainingLoop(trainingDataFromPickle = trainingData, validataionDataFromPickle = validationData, 
        channelHeaders = channelHeaders, labelHeader = labelHeader, outputDir = currentOutputFolder, device = device, parameters = parameters)

      else:
        # # write parameters to pickle - this should not change for the different folds, so keeping is independent
        ## pickle/unpickle data
        # pickle the data
        currentTrainingDataPickle = os.path.join(currentOutputFolder, 'train.pkl')
        currentValidataionDataPickle = os.path.join(currentOutputFolder, 'validation.pkl')
        trainingData.to_pickle(currentTrainingDataPickle)
        validationData.to_pickle(currentValidataionDataPickle)

        channelHeaderPickle = os.path.join(currentOutputFolder,'channelHeader.pkl')
        with open(channelHeaderPickle, 'wb') as handle:
            pickle.dump(channelHeaders, handle, protocol=pickle.HIGHEST_PROTOCOL)
        labelHeaderPickle = os.path.join(currentOutputFolder,'labelHeader.pkl')
        with open(labelHeaderPickle, 'wb') as handle:
            pickle.dump(labelHeader, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # call qsub here
        parallel_compute_command_actual = parallel_compute_command.replace('${outputDir}', currentOutputFolder)
        
        if not('python' in parallel_compute_command_actual):
          sys.exit('The \'parallel_compute_command_actual\' needs to have the python from the virtual environment, which is usually \'${GANDLF_dir}/venv/bin/python\'')

        command = parallel_compute_command_actual + \
            ' -m GANDLF.training_loop -train_loader_pickle ' + currentTrainingDataPickle + \
            ' -val_loader_pickle ' + currentValidataionDataPickle + \
            ' -num_epochs ' + str(num_epochs) + ' -batch_size ' + str(batch_size) + \
            ' -learning_rate ' + str(learning_rate) + ' -which_loss ' + which_loss + \
            ' -n_classes ' + str(n_classes) + ' -base_filters ' + str(base_filters) + \
            ' -n_channels ' + str(n_channels) + ' -which_model ' + which_model + \
            ' -channel_header_pickle ' + channelHeaderPickle + ' -label_header_pickle ' + labelHeaderPickle + \
            ' -augmentations_pickle ' + augmentationsPickle + ' -psize_pickle ' + psizePickle + ' -device ' + str(device) + ' -outputDir ' + currentOutputFolder + ' -q_verbose ' + q_verbose_string + ' -q_max_length ' + str(q_max_length) + ' -q_samples_per_volume ' + str(q_samples_per_volume) + ' -q_num_workers ' +  str(q_num_workers)
        
        subprocess.Popen(command, shell=True).wait()

      if singleFoldTraining:
        break
      currentFold = currentFold + 1 # increment the fold

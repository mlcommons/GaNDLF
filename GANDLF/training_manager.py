
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
def TrainingManager(dataframe, headers, outputDir, parameters, device):

    # check for single fold training
    singleFoldValidation = False
    singleFoldHoldout = False
    noHoldoutData = False

    if parameters['nested_training']['holdout'] < 0: # if the user wants a single fold training
        parameters['nested_training']['holdout'] = abs(parameters['nested_training']['holdout'])
        singleFoldHoldout = True

    if parameters['nested_training']['validation'] < 0: # if the user wants a single fold training
        parameters['nested_training']['validation'] = abs(parameters['nested_training']['validation'])
        singleFoldValidation = True

    # this is the condition where holdout data is not to be kept
    if parameters['nested_training']['holdout'] == 1:
        noHoldoutData = True
        singleFoldHoldout = True
        parameters['nested_training']['holdout'] = 2 # put 2 just so that the first for-loop does not fail

    # initialize the kfold structures
    kf_holdout = KFold(n_splits=parameters['nested_training']['holdout']) 
    kf_validation = KFold(n_splits=parameters['nested_training']['validation'])

    currentHoldoutFold = 0
    currentValidationFold = 0

    # get the indeces for kfold splitting
    trainingData_full = dataframe
    training_indeces_full = list(trainingData_full.index.values)

    # start the kFold train for holdout
    for trainAndVal_index, holdout_index in kf_holdout.split(training_indeces_full): # perform holdout split

        # get the current training and holdout data
        if noHoldoutData:
            trainingAndValidationData = training_indeces_full # don't consider the split indeces for this case
        else:
            trainingAndValidationData = trainingData_full.iloc[trainAndVal_index]
            holdoutData = trainingData_full.iloc[holdout_index]

        # the output of the current fold is only needed if multi-fold training is happening
        if singleFoldHoldout:
            currentOutputFolder = outputDir
        else:
            currentOutputFolder = os.path.join(outputDir, 'holdout_' + str(currentHoldoutFold))
            Path(currentOutputFolder).mkdir(parents=True, exist_ok=True)

        # save the current model configuration as a sanity check
        currentModelConfigPickle = os.path.join(currentOutputFolder, 'parameters.pkl')
        with open(currentModelConfigPickle, 'wb') as handle:
            pickle.dump(parameters, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # save the current training+validation and holdout datasets 
        if noHoldoutData:
            print('!!! WARNING !!!')
            print('!!! Holdout data is empty, which will result in scientifically incorrect results; use at your own risk !!!')
            print('!!! WARNING !!!')
        else:
            currentTrainingAndValidataionDataPickle = os.path.join(currentOutputFolder, 'trainAndVal.pkl')
            currentHoldoutDataPickle = os.path.join(currentOutputFolder, 'holdout.pkl')
            trainingAndValidationData.to_pickle(currentTrainingAndValidataionDataPickle)
            holdoutData.to_pickle(currentHoldoutDataPickle)

        current_training_indeces_full = list(trainingAndValidationData.index.values) # using the new indeces for validation training

        # start the kFold train for validation
        for train_index, test_index in kf_validation.split(current_training_indeces_full):

            # the output of the current fold is only needed if multi-fold training is happening
            if singleFoldValidation:
                currentValOutputFolder = currentOutputFolder
            else:
                currentValOutputFolder = os.path.join(outputDir, str(currentValidationFold))
                Path(currentValOutputFolder).mkdir(parents=True, exist_ok=True)

            trainingData = trainingAndValidationData.iloc[train_index]
            validationData = trainingAndValidationData.iloc[test_index]

            parallel_compute_command = parameters['parallel_compute_command']

            if (not parallel_compute_command) or (singleFoldValidation): # parallel_compute_command is an empty string, thus no parallel computing requested
                trainingLoop(trainingDataFromPickle=trainingData, validataionDataFromPickle=validationData,
                            headers = headers, outputDir=currentValOutputFolder,
                            device=device, parameters=parameters)

            else:
                # # write parameters to pickle - this should not change for the different folds, so keeping is independent
                ## pickle/unpickle data
                # pickle the data
                currentTrainingDataPickle = os.path.join(currentValOutputFolder, 'train.pkl')
                currentValidataionDataPickle = os.path.join(currentValOutputFolder, 'validation.pkl')
                trainingData.to_pickle(currentTrainingDataPickle)
                validationData.to_pickle(currentValidataionDataPickle)

                headersPickle = os.path.join(currentValOutputFolder,'headers.pkl')
                with open(headersPickle, 'wb') as handle:
                    pickle.dump(headers, handle, protocol=pickle.HIGHEST_PROTOCOL)

                # call qsub here
                parallel_compute_command_actual = parallel_compute_command.replace('${outputDir}', currentValOutputFolder)
                
                if not('python' in parallel_compute_command_actual):
                    sys.exit('The \'parallel_compute_command_actual\' needs to have the python from the virtual environment, which is usually \'${GANDLF_dir}/venv/bin/python\'')

                command = parallel_compute_command_actual + \
                    ' -m GANDLF.training_loop -train_loader_pickle ' + currentTrainingDataPickle + \
                    ' -val_loader_pickle ' + currentValidataionDataPickle + \
                    ' -parameter_pickle ' + currentModelConfigPickle + \
                    ' -headers_pickle ' + headersPickle + \
                    ' -device ' + str(device) + ' -outputDir ' + currentValOutputFolder
                
                subprocess.Popen(command, shell=True).wait()

            if singleFoldValidation:
                break
            currentValidationFold = currentValidationFold + 1 # increment the fold

        if singleFoldHoldout:
            break
        currentHoldoutFold = currentHoldoutFold + 1 # increment the fold

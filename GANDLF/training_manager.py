
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
    singleFoldTesting = False
    noTestingData = False

    if parameters['nested_training']['testing'] < 0: # if the user wants a single fold training
        parameters['nested_training']['testing'] = abs(parameters['nested_training']['testing'])
        singleFoldTesting = True

    if parameters['nested_training']['validation'] < 0: # if the user wants a single fold training
        parameters['nested_training']['validation'] = abs(parameters['nested_training']['validation'])
        singleFoldValidation = True

    # this is the condition where testing data is not to be kept
    if parameters['nested_training']['testing'] == 1:
        noTestingData = True
        singleFoldTesting = True
        parameters['nested_training']['testing'] = 2 # put 2 just so that the first for-loop does not fail

    # initialize the kfold structures
    kf_testing = KFold(n_splits=parameters['nested_training']['testing']) 
    kf_validation = KFold(n_splits=parameters['nested_training']['validation'])

    currentTestingFold = 0

    # split across subjects
    subjectIDs_full = dataframe[dataframe.columns[headers['subjectIDHeader']]].unique().tolist()

    # get the indeces for kfold splitting
    trainingData_full = dataframe

    # start the kFold train for testing
    for trainAndVal_index, testing_index in kf_testing.split(subjectIDs_full): # perform testing split

        currentValidationFold = 0 # ensure the validation fold is initialized per-testing split
        
        trainingAndValidationData = pd.DataFrame() # initialize the variable
        testingData = pd.DataFrame() # initialize the variable
        # get the current training and testing data
        if noTestingData:
            trainingAndValidationData = trainingData_full # don't consider the split indeces for this case
            testingData = None
        else:
            # loop over all trainAndVal_index and construct new dataframe
            for subject_idx in trainAndVal_index:                 
                trainingAndValidationData = trainingAndValidationData.append(trainingData_full[trainingData_full[trainingData_full.columns[headers['subjectIDHeader']]] == subjectIDs_full[subject_idx]])

            # loop over all testing_index and construct new dataframe
            for subject_idx in testing_index:                 
                testingData = testingData.append(trainingData_full[trainingData_full[trainingData_full.columns[headers['subjectIDHeader']]] == subjectIDs_full[subject_idx]])


        # the output of the current fold is only needed if multi-fold training is happening
        if singleFoldTesting:
            currentOutputFolder = outputDir
        else:
            currentOutputFolder = os.path.join(outputDir, 'testing_' + str(currentTestingFold))
            Path(currentOutputFolder).mkdir(parents=True, exist_ok=True)

        # save the current model configuration as a sanity check
        currentModelConfigPickle = os.path.join(currentOutputFolder, 'parameters.pkl')
        if not os.path.exists(currentModelConfigPickle):
            with open(currentModelConfigPickle, 'wb') as handle:
                pickle.dump(parameters, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # save the current training+validation and testing datasets 
        if noTestingData:
            print('!!! WARNING !!!')
            print('!!! Testing data is empty, which will result in scientifically incorrect results; use at your own risk !!!')
            print('!!! WARNING !!!')
            current_training_subject_indeces_full = subjectIDs_full
        else:
            currentTrainingAndValidationDataPickle = os.path.join(currentOutputFolder, 'trainAndVal.pkl')
            currentTestingDataPickle = os.path.join(currentOutputFolder, 'testing.pkl')
            
            if not os.path.exists(currentTestingDataPickle):
                testingData.to_pickle(currentTestingDataPickle)
            if not os.path.exists(currentTrainingAndValidationDataPickle):
                trainingAndValidationData.to_pickle(currentTrainingAndValidationDataPickle)
            
            current_training_subject_indeces_full = trainingAndValidationData[trainingAndValidationData.columns[headers['subjectIDHeader']]].unique().tolist()

        # start the kFold train for validation
        for train_index, val_index in kf_validation.split(current_training_subject_indeces_full):

            # the output of the current fold is only needed if multi-fold training is happening
            if singleFoldValidation:
                currentValOutputFolder = currentOutputFolder
            else:
                currentValOutputFolder = os.path.join(currentOutputFolder, str(currentValidationFold))
                Path(currentValOutputFolder).mkdir(parents=True, exist_ok=True)

            trainingData = pd.DataFrame() # initialize the variable
            validationData = pd.DataFrame() # initialize the variable

            # loop over all train_index and construct new dataframe
            for subject_idx in train_index:                 
                trainingData = trainingData.append(trainingData_full[trainingData_full[trainingData_full.columns[headers['subjectIDHeader']]] == subjectIDs_full[subject_idx]])

            # loop over all val_index and construct new dataframe
            for subject_idx in val_index:                 
                validationData = validationData.append(trainingData_full[trainingData_full[trainingData_full.columns[headers['subjectIDHeader']]] == subjectIDs_full[subject_idx]])


            if (not parameters['parallel_compute_command']) or (singleFoldValidation): # parallel_compute_command is an empty string, thus no parallel computing requested
                trainingLoop(trainingDataFromPickle=trainingData, validationDataFromPickle=validationData, headers = headers, outputDir=currentValOutputFolder,
                            device=device, parameters=parameters, testingDataFromPickle=testingData)

            else:
                # # write parameters to pickle - this should not change for the different folds, so keeping is independent
                ## pickle/unpickle data
                # pickle the data
                currentTrainingDataPickle = os.path.join(currentValOutputFolder, 'train.pkl')
                currentValidationDataPickle = os.path.join(currentValOutputFolder, 'validation.pkl')
                if not os.path.exists(currentTrainingDataPickle):
                    trainingData.to_pickle(currentTrainingDataPickle)
                if not os.path.exists(currentValidationDataPickle):
                    validationData.to_pickle(currentValidationDataPickle)

                headersPickle = os.path.join(currentValOutputFolder,'headers.pkl')
                if not os.path.exists(headersPickle):
                    with open(headersPickle, 'wb') as handle:
                        pickle.dump(headers, handle, protocol=pickle.HIGHEST_PROTOCOL)

                # call qsub here
                parallel_compute_command_actual = parameters['parallel_compute_command'].replace('${outputDir}', currentValOutputFolder)
                
                if not('python' in parallel_compute_command_actual):
                    sys.exit('The \'parallel_compute_command_actual\' needs to have the python from the virtual environment, which is usually \'${GANDLF_dir}/venv/bin/python\'')

                command = parallel_compute_command_actual + \
                    ' -m GANDLF.training_loop -train_loader_pickle ' + currentTrainingDataPickle + \
                    ' -val_loader_pickle ' + currentValidationDataPickle + \
                    ' -parameter_pickle ' + currentModelConfigPickle + \
                    ' -headers_pickle ' + headersPickle + \
                    ' -device ' + str(device) + ' -outputDir ' + currentValOutputFolder + ' -testing_loader_pickle '
                
                if noTestingData:
                    command = command + 'None'
                else:
                    command = command + currentTestingDataPickle
                
                subprocess.Popen(command, shell=True).wait()

            if singleFoldValidation:
                break
            currentValidationFold = currentValidationFold + 1 # increment the fold

        if singleFoldTesting:
            break
        currentTestingFold = currentTestingFold + 1 # increment the fold


from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
from torch.utils.data.dataset import Dataset
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import os
import random
# import scipy
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


from DeepSAGE.data.ImagesFromDataFrame import ImagesFromDataFrame
from DeepSAGE.training_loop import trainingLoop

# This function takes in a dataframe, with some other parameters and returns the dataloader
def Trainer(dataframe, augmentations, kfolds, psize, channelHeaders, labelHeader, model_parameters_file, outputDir,
    num_epochs, batch_size, learning_rate, which_loss, opt, save_best, 
    n_classes, base_filters, n_channels, which_model, parallel_compute_command):

    # kfolds = int(parameters['kcross_validation'])
    # check for single fold training
    singleFoldTraining = False
    if kfolds < 0: # if the user wants a single fold training
      kfolds = abs(kfolds)
      singleFoldTraining = True

    kf = KFold(n_splits=kfolds) # initialize the kfold structure

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
      # parametersFilePickle = os.path.join(currentOutputFolder,'model.cfg')
      copyfile(model_parameters_file, os.path.join(currentOutputFolder,'model.cfg'))

      ## pickle/unpickle data
      # pickle the data
      currentTrainingDataPickle = os.path.join(currentOutputFolder, 'train.pkl')
      currentValidataionDataPickle = os.path.join(currentOutputFolder, 'validation.pkl')
      trainingData.to_pickle(currentTrainingDataPickle)
      validationData.to_pickle(currentValidataionDataPickle)

      if not parallel_compute_command: # parallel_compute_command is an empty string, thus no parallel computing requested
        trainingLoop(train_loader_pickle = currentTrainingDataPickle, val_loader_pickle = currentValidataionDataPickle, 
        num_epochs = num_epochs, batch_size = batch_size, learning_rate = learning_rate, 
        which_loss = which_loss, opt = opt, save_best = save_best, n_classes = n_classes,
        base_filters = base_filters, n_channels = n_channels, which_model = which_model, psize = psize, 
        channelHeaders = channelHeaders, labelHeader = labelHeader, augmentations = augmentations)

      else:
        # # write parameters to pickle - this should not change for the different folds, so keeping is independent
        channelHeaderPickle = os.path.join(outputDir,'channelHeader.pkl')
        with open(channelHeaderPickle, 'wb') as handle:
            pickle.dump(channelHeaders, handle, protocol=pickle.HIGHEST_PROTOCOL)
        labelHeaderPickle = os.path.join(outputDir,'labelHeader.pkl')
        with open(labelHeaderPickle, 'wb') as handle:
            pickle.dump(labelHeader, handle, protocol=pickle.HIGHEST_PROTOCOL)
        augmentationsPickle = os.path.join(outputDir,'labelHeader.pkl')
        with open(augmentationsPickle, 'wb') as handle:
            pickle.dump(augmentations, handle, protocol=pickle.HIGHEST_PROTOCOL)
        psizePickle = os.path.join(outputDir,'psize.pkl')
        with open(psizePickle, 'wb') as handle:
            pickle.dump(psize, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # call qsub here
        parallel_compute_command = parallel_compute_command.replace('${outputDir}', outputDir)
        # todo: how to ensure that the correct python is picked up???
        command = parallel_compute_command + \
            ' python -m DeepSAGE.training_loop -train_loader_pickle ' + currentTrainingDataPickle + \
            ' -val_loader_pickle ' + currentValidataionDataPickle + \
            ' -num_epochs ' + str(num_epochs) + ' -batch_size ' + str(batch_size) + \
            ' -learning_rate ' + str(learning_rate) + ' -which_loss ' + which_loss + \
            ' -opt ' + opt + ' -save_best ' + str(save_best) + \
            ' -n_classes ' + str(n_classes) + ' -base_filters ' + str(base_filters) + \
            ' -n_channels ' + str(n_channels) + ' -which_model ' + which_model + \
            ' -channel_header_pickle ' + channelHeaderPickle + ' -label_header_pickle ' + labelHeaderPickle + \
            ' -augmentations_pickle ' + augmentationsPickle + ' -psize_pickle ' + psizePickle
            
        subprocess.Popen(command, shell=True).wait()

      currentFold = currentFold + 1 # increment the fold

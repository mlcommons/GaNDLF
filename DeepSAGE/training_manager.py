
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


from DeepSAGE.data.ImagesFromDataFrame import ImagesFromDataFrame
from DeepSAGE.training_loop import trainingLoop

# This function takes in a dataframe, with some other parameters and returns the dataloader
def Trainer(dataframe, augmentations, kfolds, psize, channelHeaders, labelHeader, model_parameters_file, outputDir,
  num_epochs, batch_size, learning_rate, which_loss, opt, save_best, 
  n_classes, base_filters, n_channels, which_model):

  # kfolds = int(parameters['kcross_validation'])
  # check for single fold training
  singleFoldTraining = False
  if kfolds < 0: # if the user wants a single fold training
      kfolds = abs(kfolds)
      singleFoldTraining = True

  kf = KFold(n_splits=kfolds) # initialize the kfold structure

  currentFold = 0

  # # write parameters to pickle - this should not change for the different folds, so keeping is independent
  # paramtersPickle = os.path.join(outputDir,'params.pkl')
  # with open(paramtersPickle, 'wb') as handle:
  #     pickle.dump(params, handle, protocol=pickle.HIGHEST_PROTOCOL)

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

      ## inside the training function
      ## for efficient processing, this can be passed off to sge as independant processes
      trainingDataFromPickle = pd.read_pickle(currentTrainingDataPickle)
      validataionDataFromPickle = pd.read_pickle(currentValidataionDataPickle)
      # paramsPickle = pd.read_pickle(parametersFilePickle)
      # with open('/path/to/params.pkl', 'rb') as handle:
      #     params = pickle.load(handle)
      ## pickle/unpickle data

      trainingDataForTorch = ImagesFromDataFrame(trainingDataFromPickle, psize, channelHeaders, labelHeader, augmentations)
      validationDataForTorch = ImagesFromDataFrame(validataionDataFromPickle, psize, channelHeaders, labelHeader, augmentations) # may or may not need to add augmentations here

      train_loader = DataLoader(trainingDataForTorch, batch_size=batch_size)
      val_loader = DataLoader(validationDataForTorch, batch_size=1)

      trainingLoop(train_loader = train_loader, val_loader = val_loader, 
        num_epochs = num_epochs, batch_size = batch_size, learning_rate = learning_rate, 
        which_loss = which_loss, opt = opt, save_best = save_best, n_classes = n_classes,
        base_filters = base_filters, n_channels = n_channels, which_model = which_model)

      currentFold = currentFold + 1 # increment the fold

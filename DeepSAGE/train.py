
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
from data.ImagesFromDataFrame import ImagesFromDataFrame
from shutil import copyfile
import time
import sys
import ast 
import pickle
from pathlib import Path


from schd import *
from models.fcn import fcn
from models.unet import unet
from models.resunet import resunet
from models.uinc import uinc
from losses import *
from utils import *

def trainingLoop(train_loader, val_loader, 
  num_epochs, batch_size, learning_rate, which_loss, opt, save_best, 
  n_classes, base_filters, n_channels, which_model, psize):
  
  # Defining our model here according to parameters mentioned in the configuration file : 
  if which_model == 'resunet':
      model = resunet(n_channels,n_classes,base_filters)
  elif which_model == 'unet':
      model = unet(n_channels,n_classes,base_filters)
  elif which_model == 'fcn':
      model = fcn(n_channels,n_classes,base_filters)
  elif which_model == 'uinc':
      model = uinc(n_channels,n_classes,base_filters)
  else:
      print('WARNING: Could not find the requested model \'' + which_model + '\' in the impementation, using ResUNet, instead', file = sys.stderr)
      which_model = 'resunet'
      model = resunet(n_channels,n_classes,base_filters)

  # setting optimizer
  if opt == 'sgd':
      optimizer = optim.SGD(model.parameters(),
                                lr= learning_rate,
                                momentum = 0.9)
  elif opt == 'adam':    
      optimizer = optim.Adam(model.parameters(), lr = learning_rate, betas = (0.9,0.999), weight_decay = 0.00005)
  else:
      print('WARNING: Could not find the requested optimizer \'' + opt + '\' in the impementation, using sgd, instead', file = sys.stderr)
      opt = 'sgd'
      optimizer = optim.SGD(model.parameters(),
                                lr= learning_rate,
                                momentum = 0.9)
  # setting the loss function
  if which_loss == 'dc':
      loss_fn  = MCD_loss
  elif which_loss == 'dcce':
      loss_fn  = DCCE
  elif which_loss == 'ce':
      loss_fn = CE
  elif which_loss == 'mse':
      loss_fn = MCD_MSE_loss
  else:
      print('WARNING: Could not find the requested loss function \'' + which_loss + '\' in the impementation, using dc, instead', file = sys.stderr)
      which_loss = 'dc'
      loss_fn  = MCD_loss

  training_start_time = time.asctime()
  startstamp = time.time()
  print("\nHostname   :" + str(os.getenv("HOSTNAME")))
  sys.stdout.flush()

  # get the channel keys
  batch = next(iter(train_loader))
  channel_keys = list(batch.keys())
  channel_keys.remove('index_ini')
  channel_keys.remove('label')  

  print("Training Data Samples: ", len(train_loader.dataset))
  sys.stdout.flush()
  device = torch.device(dev)
  print("Current Device : ", torch.cuda.current_device())
  print("Device Count on Machine : ", torch.cuda.device_count())
  print("Device Name : ", torch.cuda.get_device_name(device))
  print("Cuda Availibility : ", torch.cuda.is_available())
  print('Using device:', device)
  if device.type == 'cuda':
      print('Memory Usage:')
      print('  Allocated:', round(torch.cuda.memory_allocated(0)/1024**3, 1),'GB')
      print('  Cached: ', round(torch.cuda.memory_cached(0)/1024**3, 1), 'GB')

  sys.stdout.flush()
  model = model.to(device)

  step_size = 4*batch_size*len(train_loader.dataset)
  clr = cyclical_lr(step_size, min_lr = 0.000001, max_lr = 0.001)
  scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, [clr])
  print("Starting Learning rate is:",clr(2*step_size))
  sys.stdout.flush()
  ############## STORING THE HISTORY OF THE LOSSES #################
  avg_val_loss = 0
  total_val_loss = 0
  best_val_loss = 2000
  best_tr_loss = 2000
  total_loss = 0
  total_dice = 0
  best_idx = 0
  best_n_val_list = []
  val_avg_loss_list = []

  test = 1

# This function takes in a dataframe, with some other parameters and returns the dataloader
def Trainer(dataframe, augmentations, kfolds, psize, channelHeaders, labelHeader, model_parameters_file, outputDir):

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

      trainingLoop(train_loader, val_loader, parameters)
  
  test = 1
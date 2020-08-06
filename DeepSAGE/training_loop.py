import os
os.environ['TORCHIO_HIDE_CITATION_PROMPT'] = '1' # hides torchio citation request, see https://github.com/fepegar/torchio/issues/235

from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
from torch.utils.data.dataset import Dataset
import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
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
import argparse
import datetime
import GPUtil
from DeepSAGE.data.ImagesFromDataFrame import ImagesFromDataFrame
from DeepSAGE.schd import *
from DeepSAGE.models.fcn import fcn
from DeepSAGE.models.unet import unet
from DeepSAGE.models.resunet import resunet
from DeepSAGE.models.uinc import uinc
from DeepSAGE.losses import *
from DeepSAGE.utils import *


def trainingLoop(trainingDataFromPickle, validataionDataFromPickle, 
  num_epochs, batch_size, learning_rate, which_loss, opt, save_best, 
  n_classes, base_filters, n_channels, which_model, psize, channelHeaders, labelHeader, augmentations, outputDir, device):
  '''
  This is the main training loop
  '''
  trainingDataForTorch = ImagesFromDataFrame(trainingDataFromPickle, psize, channelHeaders, labelHeader, augmentations)
  validationDataForTorch = ImagesFromDataFrame(validataionDataFromPickle, psize, channelHeaders, labelHeader, augmentations) # may or may not need to add augmentations here

  train_loader = DataLoader(trainingDataForTorch, batch_size=batch_size)
  val_loader = DataLoader(validationDataForTorch, batch_size=1)

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

  # training_start_time = time.asctime()
  # startstamp = time.time()
  print("\nHostname   :" + str(os.getenv("HOSTNAME")))
  sys.stdout.flush()

  # get the channel keys
  batch = next(iter(train_loader))
  channel_keys = list(batch.keys())
  channel_keys.remove('index_ini')
  channel_keys.remove('label')  

  print("Training Data Samples: ", len(train_loader.dataset))
  sys.stdout.flush()
  dev = device
  
  ###
  # https://discuss.pytorch.org/t/cuda-visible-devices-make-gpu-disappear/21439/17?u=sarthakpati
  ###
  # if GPU has been requested, ensure that the correct free GPU is found and used
  if 'cuda' in dev: # this does not work correctly for windows
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    DEVICE_ID_LIST = GPUtil.getAvailable(order = 'first', limit=10)
    print('GPU devices: ', DEVICE_ID_LIST)
    environment_variable = ''
    if 'cuda-multi' in dev:
      for ids in DEVICE_ID_LIST:
        environment_variable = environment_variable + str(ids) + ','
      
      environment_variable = environment_variable[:-1] # delete last comma
      dev = 'cuda' # remove the 'multi'
      model = nn.DataParallel(model, DEVICE_ID_LIST)
    elif 'CUDA_VISIBLE_DEVICES' not in os.environ:
      environment_variable = str(DEVICE_ID_LIST[0])
    
    # only set the environment variable if there is something to set 
    if environment_variable != '':
      os.environ["CUDA_VISIBLE_DEVICES"] = environment_variable
  
  print("CUDA_VISIBLE_DEVICES: ", os.environ["CUDA_VISIBLE_DEVICES"])
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

  model = model.to(dev)

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

  batch = next(iter(train_loader))
  channel_keys = list(batch.keys())
  channel_keys.remove('index_ini')
  channel_keys.remove('label')  
  
  ################ TRAINING THE MODEL##############
  for ep in range(num_epochs):
      start = time.time()
      print("\n")
      print("Epoch Started at:", datetime.datetime.now())
      print("Epoch # : ",ep)
      print("Learning rate:", optimizer.param_groups[0]['lr'])
      model.train
    #   batch_iterator_train = iter(train_loader)
      for batch_idx, (subject) in enumerate(train_loader):
          # Load the subject and its ground truth
          # read and concat the images
          image = torch.cat([subject[key][torchio.DATA] for key in channel_keys], dim=1) # concatenate channels 
          # read the mask
          mask = subject['label'][torchio.DATA] # get the label image
          mask = one_hot(mask.float().numpy(), n_classes)
          # print("mask.shape:", mask.shape)
          # print("mask.dtype:", mask.dtype)
          # print("image.shape:", image.float().numpy().shape)
          # print("image.dtype:", image.float().numpy().dtype)
          mask = torch.from_numpy(mask)
          # Loading images into the GPU and ignoring the affine
          image, mask = image.float().to(device), mask.to(device)
          #Variable class is deprecated - parameteters to be given are the tensor, whether it requires grad and the function that created it   
          image, mask = Variable(image, requires_grad = True), Variable(mask, requires_grad = True)
          # Making sure that the optimizer has been reset
          optimizer.zero_grad()
          # Forward Propagation to get the output from the models
          torch.cuda.empty_cache()
          output = model(image.float())
          # Computing the loss
          loss = loss_fn(output.double(), mask.double(),n_classes)
          # Back Propagation for model to learn
          loss.backward()
          #Updating the weight values
          optimizer.step()
          #Pushing the dice to the cpu and only taking its value
          curr_loss = loss.cpu().data.item()
          #train_loss_list.append(loss.cpu().data.item())
          total_loss+=curr_loss
          # Computing the average loss
          average_loss = total_loss/(batch_idx + 1)
          #Computing the dice score 
          curr_dice = 1 - curr_loss
          #Computing the total dice
          total_dice+= curr_dice
          #Computing the average dice
          average_dice = total_dice/(batch_idx + 1)
          scheduler.step()
          torch.cuda.empty_cache()
      print("Epoch Training dice:" , average_dice)      
      if average_dice > 1-best_tr_loss:
          best_tr_idx = ep
          best_tr_loss = 1 - average_dice
      total_dice = 0
      total_loss = 0     
      print("Best Training Dice:", 1-best_tr_loss)
      print("Best Training Epoch:", best_tr_idx)
      print("Average Loss:", average_loss)
      # Now we enter the evaluation/validation part of the epoch    
      model.eval        
    #   batch_iterator_val = iter(val_loader)
      for batch_idx, (subject) in enumerate(val_loader):
          with torch.no_grad():        
              image = torch.cat([subject[key][torchio.DATA] for key in channel_keys], dim=1) # concatenate channels 
              mask = subject['label'][torchio.DATA] # get the label image
              image, mask = image.to(device), mask.to(device)
              output = model(image.float())
              curr_loss = loss_fn(output.double(), mask.double(),n_classes).cpu().data.item()
              total_loss+=curr_loss
              # Computing the average loss
              average_loss = total_loss/(batch_idx + 1)
              #Computing the dice score 
              curr_dice = 1 - curr_loss
              #Computing the total dice
              total_dice+= curr_dice
              #Computing the average dice
              average_dice = total_dice/(batch_idx + 1)

      print("Epoch Validation Dice: ", average_dice)
      print("Average Loss:", average_loss)
      torch.save(model, os.path.join(outputDir, which_model  + str(ep) + ".pt"))
      if ep > save_best:
          keep_list = np.argsort(np.array(val_avg_loss_list))
          keep_list = keep_list[0:save_best]
          for j in range(ep):
              if j not in keep_list:
                  if os.path.isfile(os.path.join(outputDir + which_model  + str(j) + ".pt")):
                      os.remove(os.path.join(outputDir + which_model  + str(j) + ".pt"))
          
          print("Best ",save_best," validation epochs:", keep_list)

      total_dice = 0
      total_loss = 0
      stop = time.time()   
      val_avg_loss_list.append(1-average_dice)  
      print("Time for epoch:",(stop - start)/60,"mins")    
      sys.stdout.flush()


if __name__ == "__main__":

    # parse the cli arguments here
    parser = argparse.ArgumentParser(description = "Training Loop of DeepSAGE")
    parser.add_argument('-train_loader_pickle', type=str, help = 'Train loader pickle', required=True)
    parser.add_argument('-val_loader_pickle', type=str, help = 'Validation loader pickle', required=True)
    parser.add_argument('-num_epochs', type=int, help = 'Number of epochs', required=True)
    parser.add_argument('-batch_size', type=int, help = 'Batch size', required=True)
    parser.add_argument('-learning_rate', type=float, help = 'Learning rate', required=True)
    parser.add_argument('-which_loss', type=str, help = 'Loss type', required=True)
    parser.add_argument('-opt', type=str, help = 'Optimizer type', required=True)
    parser.add_argument('-save_best', type=int, help = 'Number of best models to save', required=True)
    parser.add_argument('-n_classes', type=int, help = 'Number of output classes', required=True)
    parser.add_argument('-base_filters', type=int, help = 'Number of base filters', required=True)
    parser.add_argument('-n_channels', type=int, help = 'Number of input channels', required=True)
    parser.add_argument('-which_model', type=str, help = 'Model type', required=True)
    parser.add_argument('-channel_header_pickle', type=str, help = 'Channel header pickle', required=True)
    parser.add_argument('-label_header_pickle', type=str, help = 'Label header pickle', required=True)
    parser.add_argument('-augmentations_pickle', type=str, help = 'Augmentations pickle', required=True)
    parser.add_argument('-psize_pickle', type=str, help = 'psize pickle', required=True)
    parser.add_argument('-outputDir', type=str, help = 'Output directory', required=True)
    parser.add_argument('-device', type=str, help = 'Device to train on', required=True)
    
    args = parser.parse_args()

    # # write parameters to pickle - this should not change for the different folds, so keeping is independent
    psize = pickle.load(open(args.psize_pickle,"rb"))
    channel_header = pickle.load(open(args.channel_header_pickle,"rb"))
    label_header = pickle.load(open(args.label_header_pickle,"rb"))
    augmentations = pickle.load(open(args.augmentations_pickle,"rb"))
    trainingDataFromPickle = pd.read_pickle(args.train_loader_pickle)
    validataionDataFromPickle = pd.read_pickle(args.val_loader_pickle)


    trainingLoop(trainingDataFromPickle = trainingDataFromPickle, 
        validataionDataFromPickle = validataionDataFromPickle, 
        num_epochs = args.num_epochs, 
        batch_size = args.batch_size, 
        learning_rate = args.learning_rate, 
        which_loss = args.which_loss, 
        opt = args.opt, 
        save_best = args.save_best, 
        n_classes = args.n_classes,
        base_filters = args.base_filters, 
        n_channels = args.n_channels, 
        which_model = args.which_model, 
        psize = psize, 
        channelHeaders = channel_header, 
        labelHeader = label_header, 
        augmentations = augmentations,
        outputDir = args.outputDir,
        device = args.device)
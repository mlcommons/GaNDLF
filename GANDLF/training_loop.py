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
from GANDLF.data.ImagesFromDataFrame import ImagesFromDataFrame
from GANDLF.schd import *
from GANDLF.models.fcn import fcn
from GANDLF.models.unet import unet
from GANDLF.models.resunet import resunet
from GANDLF.models.uinc import uinc
from GANDLF.losses import *
from GANDLF.utils import *


def trainingLoop(trainingDataFromPickle, validataionDataFromPickle, 
  num_epochs, batch_size, learning_rate, scheduler, which_loss, opt,
  class_list, base_filters, n_channels, which_model, psize, channelHeaders, labelHeader, augmentations, outputDir, device, q_max_length, q_samples_per_volume, q_num_workers, q_verbose):
  '''
  This is the main training loop
  '''
  trainingDataForTorch = ImagesFromDataFrame(trainingDataFromPickle, psize, channelHeaders, labelHeader, q_max_length, q_samples_per_volume, q_num_workers, q_verbose, train = True, augmentations = augmentations)
  validationDataForTorch = ImagesFromDataFrame(validataionDataFromPickle, psize, channelHeaders, labelHeader, q_max_length, q_samples_per_volume, q_num_workers, q_verbose, train = True, augmentations = augmentations) # may or may not need to add augmentations here

  train_loader = DataLoader(trainingDataForTorch, batch_size=batch_size, shuffle=True)
  val_loader = DataLoader(validationDataForTorch, batch_size=1)

  # Defining our model here according to parameters mentioned in the configuration file : 
  if which_model == 'resunet':
    model = resunet(n_channels,len(class_list),base_filters)
  elif which_model == 'unet':
    model = unet(n_channels,len(class_list),base_filters)
  elif which_model == 'fcn':
    model = fcn(n_channels,len(class_list),base_filters)
  elif which_model == 'uinc':
    model = uinc(n_channels,len(class_list),base_filters)
  else:
    print('WARNING: Could not find the requested model \'' + which_model + '\' in the impementation, using ResUNet, instead', file = sys.stderr)
    which_model = 'resunet'
    model = resunet(n_channels,len(class_list),base_filters)

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


  print("Training Data Samples: ", len(train_loader.dataset))
  sys.stdout.flush()
  dev = device
  
  # ###
  # # https://discuss.pytorch.org/t/cuda-visible-devices-make-gpu-disappear/21439/17?u=sarthakpati
  # ###
  # # if GPU has been requested, ensure that the correct free GPU is found and used
  # if 'cuda' in dev: # this does not work correctly for windows
  #   os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
  #   DEVICE_ID_LIST = GPUtil.getAvailable(order = 'first', limit=10)
  #   print('GPU devices: ', DEVICE_ID_LIST)
  #   environment_variable = ''
  #   if 'cuda-multi' in dev:
  #     for ids in DEVICE_ID_LIST:
  #       environment_variable = environment_variable + str(ids) + ','
      
  #     environment_variable = environment_variable[:-1] # delete last comma
  #     dev = 'cuda' # remove the 'multi'
  #     model = nn.DataParallel(model, DEVICE_ID_LIST)
  #   elif ('CUDA_VISIBLE_DEVICES' not in os.environ) or (os.environ["CUDA_VISIBLE_DEVICES"] == ''):
  #     environment_variable = str(DEVICE_ID_LIST[0])
    
  #   # only set the environment variable if there is something to set 
  #   if environment_variable != '':
  #     print('Setting \'CUDA_VISIBLE_DEVICES\' to: ', environment_variable)
  #     os.environ["CUDA_VISIBLE_DEVICES"] = environment_variable
  
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
      print('  Cached: ', round(torch.cuda.memory_reserved(0)/1024**3, 1), 'GB')

  sys.stdout.flush()

  model = model.to(dev)
  # Checking for the learning rate scheduler
  if scheduler == "triangle":
    step_size = 4*batch_size*len(train_loader.dataset)
    clr = cyclical_lr(step_size, min_lr = 10**-3, max_lr=1)
    scheduler_lr = torch.optim.lr_scheduler.LambdaLR(optimizer, [clr])
    print("Starting Learning rate is:",clr(2*step_size))
  elif scheduler == "exp":
    scheduler_lr = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.1, last_epoch=-1)
  elif scheduler == "step":
    scheduler_lr = torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma=0.1, last_epoch=-1)
  elif scheduler == "reduce-on-plateau":
    scheduler_lr = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=False)
  elif scheduler == "triangular":
    scheduler_lr = torch.optim.lr_scheduler.CyclicLR(optimizer, learning_rate * 0.001, learning_rate, step_size_up=4*batch_size*len(train_loader.dataset), step_size_down=None, mode='triangular', gamma=1.0, scale_fn=None, scale_mode='cycle', cycle_momentum=True, base_momentum=0.8, max_momentum=0.9, last_epoch=-1)
  else:
    print('WARNING: Could not find the requested Learning Rate scheduler \'' + scheduler + '\' in the impementation, using exp, instead', file = sys.stderr)
    scheduler_lr = scheduler_lr = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.1, last_epoch=-1)

  print(scheduler_lr)
  sys.stdout.flush()
  ############## STORING THE HISTORY OF THE LOSSES #################
  best_val_dice = -1
  best_tr_dice = -1
  total_loss = 0
  total_dice = 0
  best_idx = 0  
  # Getting the channels for training and removing all the non numeric entries from the channels
  batch = next(iter(train_loader))
  channel_keys = list(batch.keys())
  channel_keys_new = []

  # automatic mixed precision - https://pytorch.org/docs/stable/amp.html
  scaler = torch.cuda.amp.GradScaler() 

  for item in channel_keys:
    if item.isnumeric():
      channel_keys_new.append(item)
  channel_keys = channel_keys_new
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
          mask = one_hot(mask.cpu().float().numpy(), class_list)
          mask = torch.from_numpy(mask)
          # Loading images into the GPU and ignoring the affine
          image, mask = image.float().to(device), mask.to(device)
          # Making sure that the optimizer has been reset
          optimizer.zero_grad()
          # Forward Propagation to get the output from the models
          torch.cuda.empty_cache()
          # Casts operations to mixed precision 
          with torch.cuda.amp.autocast(): 
              output = model(image.float())
              # Computing the loss
              mask = mask.unsqueeze(0)
              loss = loss_fn(output.double(), mask.double(),len(class_list))
          # Back Propagation for model to learn
          scaler.scale(loss).backward() 
          ### gradient clipping
          # # Unscales the gradients of optimizer's assigned params in-place
          # scaler.unscale_(optimizer)
          # # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
          # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
          ### gradient clipping
          #Updating the weight values
          scaler.step(optimizer) 
          #Pushing the dice to the cpu and only taking its value
          curr_loss = loss.cpu().data.item()
          #train_loss_list.append(loss.cpu().data.item())
          total_loss+=curr_loss        
          #Computing the dice score 
          curr_dice = 1 - curr_loss
          #Computing the total dice
          total_dice+= curr_dice
          # update scale for next iteration
          scaler.update() 
          torch.cuda.empty_cache()
          if scheduler == "triangular":
            scheduler_lr.step()
          print("something")

      average_dice = total_dice/(batch_idx + 1)
      average_loss = total_loss/(batch_idx + 1)
    
      if average_dice > best_tr_dice:
          best_tr_idx = ep
          best_tr_dice = average_dice

      print("Epoch Training dice:" , average_dice) 
      print("Best Training Dice:", best_tr_dice)
      print("Average Training Loss:", average_loss)
      print("Best Training Epoch: ",best_tr_idx)
      total_dice = 0
      total_loss = 0  
      # Now we enter the evaluation/validation part of the epoch    
      model.eval        
    #   batch_iterator_val = iter(val_loader)
      for batch_idx, (subject) in enumerate(val_loader):
          with torch.no_grad():        
              image = torch.cat([subject[key][torchio.DATA] for key in channel_keys], dim=1) # concatenate channels 
              mask = subject['label'][torchio.DATA] # get the label image
              image, mask = image.to(device), mask.to(device)
              output = model(image.float())
              # one hot encoding the mask 
              mask = one_hot(mask.cpu().float().numpy(), class_list)
              mask = torch.from_numpy(mask)
              mask = mask.unsqueeze(0)
              # making sure that the output and mask are on the same device
              output, mask = output.to(device), mask.to(device)
              curr_loss = loss_fn(output.double(), mask.double(),len(class_list)).cpu().data.item()
              total_loss+=curr_loss
              #Computing the dice score 
              curr_dice = 1 - curr_loss
              #Computing the total dice
              total_dice+= curr_dice

      #Computing the average dice
      average_dice = total_dice/(batch_idx + 1)
      # Computing the average loss
      average_loss = total_loss/(batch_idx + 1)

      if average_dice > best_val_dice:
          best_val_idx = ep
          best_val_dice = average_dice
          torch.save(model.state_dict(), os.path.join(outputDir, which_model + "_best.pt"))
  
      print("Epoch Validation dice:" , average_dice) 
      print("Best Validation Dice:", best_val_dice)
      print("Average Validation Loss:", average_loss)
      print("Best Validation Epoch: ",best_val_idx)

      # Updating the learning rate accoring to some conditions - reduce lr on plateau needs out loss to be monitored and scedules the LR accordingly. Others change irrespective of loss.
      if not scheduler == "triangular":
        if scheduler == "reduce-on-plateau":
          scheduler_lr.step(average_loss)
        else:
          scheduler_lr.step()

      total_dice = 0
      total_loss = 0
      stop = time.time()   
      print("Time for epoch:",(stop - start)/60,"mins")    
      sys.stdout.flush()


if __name__ == "__main__":

    # parse the cli arguments here
    parser = argparse.ArgumentParser(description = "Training Loop of GANDLF")
    parser.add_argument('-train_loader_pickle', type=str, help = 'Train loader pickle', required=True)
    parser.add_argument('-val_loader_pickle', type=str, help = 'Validation loader pickle', required=True)
    parser.add_argument('-num_epochs', type=int, help = 'Number of epochs', required=True)
    parser.add_argument('-batch_size', type=int, help = 'Batch size', required=True)
    parser.add_argument('-learning_rate', type=float, help = 'Learning rate', required=True)
    parser.add_argument('-which_loss', type=str, help = 'Loss type', required=True)
    parser.add_argument('-opt', type=str, help = 'Optimizer type', required=True)
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
    parser.add_argument('-q_max_length', type=int, help = '[Queue] Max length', required=True)
    parser.add_argument('-q_samples_per_volume', type=int, help = '[Queue] Samples per volume', required=True)
    parser.add_argument('-q_num_workers', type=int, help = '[Queue] Number of workers', required=True)
    parser.add_argument('-q_verbose', type=str, help = '[Queue] Verbose debugging', required=True)
    
    args = parser.parse_args()

    # # write parameters to pickle - this should not change for the different folds, so keeping is independent
    psize = pickle.load(open(args.psize_pickle,"rb"))
    channel_header = pickle.load(open(args.channel_header_pickle,"rb"))
    label_header = pickle.load(open(args.label_header_pickle,"rb"))
    augmentations = pickle.load(open(args.augmentations_pickle,"rb"))
    trainingDataFromPickle = pd.read_pickle(args.train_loader_pickle)
    validataionDataFromPickle = pd.read_pickle(args.val_loader_pickle)

    q_verbose = False
    if args.q_verbose == 'True':
        q_verbose = True

    trainingLoop(trainingDataFromPickle = trainingDataFromPickle, 
        validataionDataFromPickle = validataionDataFromPickle, 
        num_epochs = args.num_epochs, 
        batch_size = args.batch_size, 
        learning_rate = args.learning_rate, 
        which_loss = args.which_loss, 
        opt = args.opt, 
        n_classes = args.n_classes,
        base_filters = args.base_filters, 
        n_channels = args.n_channels, 
        which_model = args.which_model, 
        psize = psize, 
        channelHeaders = channel_header, 
        labelHeader = label_header, 
        augmentations = augmentations,
        outputDir = args.outputDir,
        device = args.device,
        q_verbose = q_verbose,
        q_max_length = args.q_max_length,
        q_samples_per_volume = args.q_samples_per_volume,
        q_num_workers = args.q_num_workers)
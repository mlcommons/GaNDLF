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
import torchio
from torchio import Image, Subject
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


def inferenceLoop(inferenceDataFromPickle,batch_size, which_loss,n_classes, base_filters, n_channels, which_model, psize, channelHeaders, labelHeader, outputDir, device):
  '''
  This is the main inference loop
  '''
  # Setting up the inference loader
  inferenceDataForTorch = ImagesFromDataFrame(inferenceDataFromPickle, psize, channelHeaders, labelHeader, train = False)
  inference_loader = DataLoader(inferenceDataForTorch, batch_size=batch_size)
  
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

  print("\nHostname   :" + str(os.getenv("HOSTNAME")))
  sys.stdout.flush()

  # get the channel keys
  batch = next(iter(inference_loader))
  channel_keys = list(batch.keys())
  channel_keys.remove('label')  

  print("Training Data Samples: ", len(inference_loader.dataset))
  sys.stdout.flush()
  dev = device
  
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
      model = nn.DataParallel(model)
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

  batch = next(iter(inference_loader))
  channel_keys = list(batch.keys())
  channel_keys.remove('label')  
  
  model.eval
  #   batch_iterator_train = iter(train_loader)
  with torch.no_grad():
    for batch_idx, subject in enumerate(inferenceDataForTorch):
        # Load the subject and its ground truth
        grid_sampler = torchio.inference.GridSampler(subject , psize, 4)
        patch_loader = torch.utils.data.DataLoader(grid_sampler, batch_size=batch_size)
        aggregator = torchio.inference.GridAggregator(grid_sampler)
        for patches_batch in patch_loader:
            image = torch.cat([patches_batch[key][torchio.DATA] for key in channel_keys], dim=1).to(device)
            locations = patches_batch[torchio.LOCATION]
            logits = model(image)
            aggregator.add_batch(labels, locations)

        foreground = aggregator.get_output_tensor()
        print(foreground.shape())

        # read the mask
        locations = subject[torchio.LOCATION]
        mask = subject['label'][torchio.DATA] # get the label image
        mask = one_hot(mask.float().numpy(), n_classes)
        mask = torch.from_numpy(mask)
        # Loading images into the GPU and ignoring the affine
        image, mask = image.float().to(device), mask.to(device)
        #Variable class is deprecated - parameteters to be given are the tensor, whether it requires grad and the funct ion that created it   
        image, mask = Variable(image, requires_grad = True), Variable(mask, requires_grad = True)
        # Making sure that the optimizer has been reset
        # Forward Propagation to get the output from the models
        torch.cuda.empty_cache()
        output = model(image.float())
        # Aggregarting the patches
        aggregator.add_batch(output, locations)
        # Computing the loss
        loss = loss_fn(output.double(), mask.double(),n_classes)
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
        torch.cuda.empty_cache()


if __name__ == "__main__":

    # parse the cli arguments here
    parser = argparse.ArgumentParser(description = "Inference Loop of DeepSAGE")
    parser.add_argument('-inference_loader_pickle', type=str, help = 'Inference loader pickle', required=True)
    parser.add_argument('-which_loss', type=str, help = 'Loss type', required=True)
    parser.add_argument('-n_classes', type=int, help = 'Number of output classes', required=True)
    parser.add_argument('-base_filters', type=int, help = 'Number of base filters', required=True)
    parser.add_argument('-n_channels', type=int, help = 'Number of input channels', required=True)
    parser.add_argument('-which_model', type=str, help = 'Model type', required=True)
    parser.add_argument('-channel_header_pickle', type=str, help = 'Channel header pickle', required=True)
    parser.add_argument('-label_header_pickle', type=str, help = 'Label header pickle', required=True)
    parser.add_argument('-psize_pickle', type=str, help = 'psize pickle', required=True)
    parser.add_argument('-outputDir', type=str, help = 'Output directory', required=True)
    parser.add_argument('-device', type=str, help = 'Device to train on', required=True)
    
    args = parser.parse_args()

    # # write parameters to pickle - this should not change for the different folds, so keeping is independent
    psize = pickle.load(open(args.psize_pickle,"rb"))
    channel_header = pickle.load(open(args.channel_header_pickle,"rb"))
    label_header = pickle.load(open(args.label_header_pickle,"rb"))
    inferenceDataFromPickle = pd.read_pickle(inference_loader_pickle)


    inferenceLoop(inference_loader_pickle = inferenceDataFromPickle, 
        batch_size = args.batch_size, 
        which_loss = args.which_loss, 
        n_classes = args.n_classes,
        base_filters = args.base_filters, 
        n_channels = args.n_channels, 
        which_model = args.which_model, 
        psize = psize, 
        channelHeaders = channel_header, 
        labelHeader = label_header, 
        outputDir = args.outputDir,
        device = args.device)

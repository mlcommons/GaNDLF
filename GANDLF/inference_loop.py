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
import SimpleITK as sitk
from GANDLF.data.ImagesFromDataFrame import ImagesFromDataFrame
from GANDLF.schd import *
from GANDLF.models.fcn import fcn
from GANDLF.models.unet import unet
from GANDLF.models.resunet import resunet
from GANDLF.models.uinc import uinc
from GANDLF.losses import *
from GANDLF.utils import *

def inferenceLoop(inferenceDataFromPickle, headers, device, parameters, outputDir):
  '''
  This is the main inference loop
  '''
  # extract variables form parameters dict
  psize = parameters['psize']
  q_max_length = parameters['q_max_length']
  q_samples_per_volume = parameters['q_samples_per_volume']
  q_num_workers = parameters['q_num_workers']
  q_verbose = parameters['q_verbose']
  augmentations = parameters['data_augmentation']
  which_model = parameters['model']['architecture']
  class_list = parameters['class_list']
  base_filters = parameters['base_filters']
  batch_size = parameters['batch_size']
  
  n_channels = len(headers['channelHeaders'])
  n_classList = len(class_list)

  if len(psize) == 2:
      psize.append(1) # ensuring same size during torchio processing

  # Setting up the inference loader
  inferenceDataForTorch = ImagesFromDataFrame(inferenceDataFromPickle, psize, headers, q_max_length, q_samples_per_volume, q_num_workers, q_verbose, train = False, augmentations = augmentations, resize = parameters['resize'])
  inference_loader = DataLoader(inferenceDataForTorch, batch_size=batch_size)
  
  # Defining our model here according to parameters mentioned in the configuration file : 
  if which_model == 'resunet':
    model = resunet(parameters['dimension'], n_channels, n_classList, base_filters, final_convolution_layer = parameters['model']['final_layer'])
    if psize[-1] == 1:
        checkPatchDivisibility(psize[:-1]) # for 2D, don't check divisibility of last dimension
    else:
        checkPatchDivisibility(psize)
  elif which_model == 'unet':
    model = unet(parameters['dimension'], n_channels, n_classList, base_filters, final_convolution_layer = parameters['model']['final_layer'])
    if psize[-1] == 1:
        checkPatchDivisibility(psize[:-1]) # for 2D, don't check divisibility of last dimension
    else:
        checkPatchDivisibility(psize)
  elif which_model == 'fcn':
    model = fcn(parameters['dimension'], n_channels, n_classList, base_filters, final_convolution_layer = parameters['model']['final_layer'])
  elif which_model == 'uinc':
    model = uinc(parameters['dimension'], n_channels, n_classList, base_filters, final_convolution_layer = parameters['model']['final_layer'])
  else:
    print('WARNING: Could not find the requested model \'' + which_model + '\' in the implementation, using ResUNet, instead', file = sys.stderr)
    which_model = 'resunet'
    model = resunet(parameters['dimension'], n_channels, n_classList, base_filters, final_convolution_layer = parameters['model']['final_layer'])

  # Loading the weights into the model
  main_dict = torch.load(os.path.join(outputDir,str(which_model) + "_best.pth.tar"))
  model.load_state_dict(main_dict['model_state_dict'])
  
  print("\nHostname   :" + str(os.getenv("HOSTNAME")))
  sys.stdout.flush()

  # get the channel keys for concatenation later (exclude non numeric channel keys)
  batch = next(iter(inference_loader))
  channel_keys = list(batch.keys())
  channel_keys_new = []
  for item in channel_keys:
    if item.isnumeric():
      channel_keys_new.append(item)
  channel_keys = channel_keys_new

  print("Data Samples: ", len(inference_loader.dataset))
  sys.stdout.flush()
  if device != 'cpu':
      if os.environ.get('CUDA_VISIBLE_DEVICES') is None:
          sys.exit('Please set the environment variable \'CUDA_VISIBLE_DEVICES\' correctly before trying to run GANDLF on GPU')
      
      dev = os.environ.get('CUDA_VISIBLE_DEVICES')
      # multi-gpu support:  https://discuss.pytorch.org/t/cuda-visible-devices-make-gpu-disappear/21439/17?u=sarthakpati
      if ',' in dev:
          device = torch.device('cuda')
          model = nn.DataParallel(model, '[' + dev + ']')
      else:
          print('Device requested via CUDA_VISIBLE_DEVICES: ', dev)
          if (torch.cuda.device_count() == 1) and (int(dev) == 1): # this should be properly fixed
              dev = '0'
          print('Device finally used: ', dev)
          device = torch.device('cuda:' + dev)
          model = model.to(int(dev))
          print('Memory Total : ', round(torch.cuda.get_device_properties(int(dev)).total_memory/1024**3, 1), 'GB')
          print('Memory Usage : ')
          print('Allocated : ', round(torch.cuda.memory_allocated(int(dev))/1024**3, 1),'GB')
          print('Cached: ', round(torch.cuda.memory_reserved(int(dev))/1024**3, 1), 'GB')
      
      print("Device - Current: %s Count: %d Name: %s Availability: %s"%(torch.cuda.current_device(), torch.cuda.device_count(), torch.cuda.get_device_name(device), torch.cuda.is_available()))
    
  else:
      dev = -1
      device = torch.device('cpu')
      model.cpu()
      amp = False
      print("Since Device is CPU, Mixed Precision Training is set to False")
  
  
  # multi-gpu support: https://discuss.pytorch.org/t/cuda-visible-devices-make-gpu-disappear/21439/17?u=sarthakpati
  if os.environ.get('CUDA_VISIBLE_DEVICES') is not None:
      if ',' in os.environ.get('CUDA_VISIBLE_DEVICES'):
          environment_cuda_visible = os.environ["CUDA_VISIBLE_DEVICES"]
          model = nn.DataParallel(model, '[' + environment_cuda_visible + ']')
  
  # print stats
  print('Using device:', device)
  sys.stdout.flush()

  model = model.to(dev)

  sys.stdout.flush()
  model.eval()
  average_dice, average_loss = get_stats(model, inference_loader, psize, channel_keys, class_list, loss_fn, weights = None, save_mask = True)
  print(average_dice, average_loss)

if __name__ == "__main__":

    # parse the cli arguments here
    parser = argparse.ArgumentParser(description = "Inference Loop of GANDLF")
    parser.add_argument('-inference_loader_pickle', type=str, help = 'Inference loader pickle', required=True)
    parser.add_argument('-parameter_pickle', type=str, help = 'Parameters pickle', required=True)
    parser.add_argument('-headers_pickle', type=str, help = 'Header pickle', required=True)
    parser.add_argument('-outputDir', type=str, help = 'Output directory', required=True)
    parser.add_argument('-device', type=str, help = 'Device to train on', required=True)
    
    args = parser.parse_args()

    # # write parameters to pickle - this should not change for the different folds, so keeping is independent
    psize = pickle.load(open(args.psize_pickle,"rb"))
    headers = pickle.load(open(args.headers_pickle,"rb"))
    label_header = pickle.load(open(args.label_header_pickle,"rb"))
    parameters = pickle.load(open(args.parameter_pickle,"rb"))
    inferenceDataFromPickle = pd.read_pickle(args.inference_loader_pickle)

    inferenceLoop(inference_loader_pickle = inferenceDataFromPickle, 
        headers = headers, 
        parameters = parameters,
        outputDir = args.outputDir,
        device = args.device,)

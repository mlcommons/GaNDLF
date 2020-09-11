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
  opt = parameters['opt']
  loss_function = parameters['loss_function']
  scheduler = parameters['scheduler']
  class_list = parameters['class_list']
  base_filters = parameters['base_filters']
  base_filters = parameters['base_filters']
  base_filters = parameters['base_filters']
  batch_size = parameters['batch_size']
  learning_rate = parameters['learning_rate']
  num_epochs = parameters['num_epochs']
  
  n_channels = len(headers['channelHeaders'])
  n_classList = len(class_list)

  # Setting up the inference loader
  inferenceDataForTorch = ImagesFromDataFrame(inferenceDataFromPickle, psize, headers, q_max_length, q_samples_per_volume, q_num_workers, q_verbose, train = False, augmentations = augmentations, resize = parameters['resize'])
  inference_loader = DataLoader(inferenceDataForTorch, batch_size=batch_size)
  
  # Defining our model here according to parameters mentioned in the configuration file : 
  if which_model == 'resunet':
    model = resunet(n_channels, n_classList, base_filters, final_convolution_layer = parameters['model']['final_layer'])
  elif which_model == 'unet':
    model = unet(n_channels, n_classList, base_filters, final_convolution_layer = parameters['model']['final_layer'])
  elif which_model == 'fcn':
    model = fcn(n_channels, n_classList, base_filters, final_convolution_layer = parameters['model']['final_layer'])
  elif which_model == 'uinc':
    model = uinc(n_channels, n_classList, base_filters, final_convolution_layer = parameters['model']['final_layer'])
  else:
    print('WARNING: Could not find the requested model \'' + which_model + '\' in the implementation, using ResUNet, instead', file = sys.stderr)
    which_model = 'resunet'
    model = resunet(n_channels, n_classList, base_filters, final_convolution_layer = parameters['model']['final_layer'])

  # Loading the weights into the model
  model.load_state_dict(torch.load(os.path.join(outputDir,str(which_model) + "_best.pt")))
  # setting the loss function
  if loss_function == 'dc':
    loss_fn  = MCD_loss
  elif loss_function == 'dcce':
    loss_fn  = DCCE
  elif loss_function == 'ce':
    loss_fn = CE
  elif loss_function == 'mse':
    loss_fn = MCD_MSE_loss
  else:
    print('WARNING: Could not find the requested loss function \'' + loss_fn + '\' in the implementation, using dc, instead', file = sys.stderr)
    loss_function = 'dc'
    loss_fn  = MCD_loss

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
      dev = int(device)
      device = torch.device(dev)
      print("Current Device : ", torch.cuda.current_device())
      print("Device Count on Machine : ", torch.cuda.device_count())
      print("Device Name : ", torch.cuda.get_device_name(device))
      print("Cuda Availability : ", torch.cuda.is_available())
  else:
      dev = -1
      device = torch.device('cpu')
  
  
  # multi-gpu support
  # ###
  # # https://discuss.pytorch.org/t/cuda-visible-devices-make-gpu-disappear/21439/17?u=sarthakpati
  # ###
  if os.environ.get('CUDA_VISIBLE_DEVICES') is not None:
      if ',' in os.environ.get('CUDA_VISIBLE_DEVICES'):
          environment_cuda_visible = os.environ["CUDA_VISIBLE_DEVICES"]
          model = nn.DataParallel(model, '[' + environment_cuda_visible + ']')
  
  # print stats
  print('Using device:', device)
  if device.type == 'cuda':
      print("Current Device : ", torch.cuda.current_device())
      print("Device Count on Machine : ", torch.cuda.device_count())
      print("Device Name : ", torch.cuda.get_device_name(device))
      print("Cuda Availibility : ", torch.cuda.is_available())
      print('Memory Usage : ')
      print('Allocated : ', round(torch.cuda.memory_allocated(0)/1024**3, 1),'GB')
      print('Cached: ', round(torch.cuda.memory_reserved(0)/1024**3, 1), 'GB')

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

  model.eval()
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
            pred_mask = model(image)
            aggregator.add_batch(pred_mask, locations)

        pred_mask = aggregator.get_output_tensor()
        pred_mask = pred_mask.unsqueeze(0)
        # read the ground truth mask
        if not subject['label'] == "NA":
          mask = subject['label'][torchio.DATA] # get the label image
          mask = mask.unsqueeze(0) # increasing the number of dimension of the mask
          mask = one_hot(mask.float().numpy(), class_list)
          mask = torch.from_numpy(mask)
          torch.cuda.empty_cache()
          # Computing the loss
          #mask = torch.nn.functional.one_hot(mask, num_classes=-1)
          mask = mask.unsqueeze(0)
          loss = loss_fn(pred_mask.double(), mask.double(),len(class_list))
          #Pushing the dice to the cpu and only taking its value
          curr_loss = loss.cpu().data.item()
          #train_loss_list.append(loss.cpu().data.item())
          total_loss+=curr_loss
          # Computing the average loss
          average_loss = total_loss/(batch_idx + 1)
          #Computing the dice score 
          curr_dice = MCD(pred_mask.double(), mask.double(), n_classList)
          #Computing the total dice
          total_dice+= curr_dice
          torch.cuda.empty_cache()
          print("Current Dice is: ", curr_dice)
        else:
          print("Ground Truth Mask not found. Generating the Segmentation based one the METADATA of one of the modalities, The Segmentation will be named accordingly")

        # Saving the mask to disk in the output directory using the same metadata as from the 
        inputImage = sitk.ReadImage(subject['path_to_metadata'])
        pred_mask = pred_mask.cpu().numpy()
        # works since batch size is always one in inference time  
        pred_mask = reverse_one_hot(pred_mask[0],class_list)
        
        result_image = sitk.GetImageFromArray(np.swapaxes(pred_mask,0,2))
        result_image.CopyInformation(inputImage)
        # resize
        if parameters['resize'] is not None:
            result_image = resize_image(resize_image, parameters['resize'], sitk.sitkNearestNeighbor)
        
        result_image.CopyInformation(inputImage)
        patient_name = os.path.basename(subject['path_to_metadata'])
        if not os.path.isdir(os.path.join(outputDir,"generated_masks")):
          os.mkdir(os.path.join(outputDir,"generated_masks"))
        sitk.WriteImage(result_image, os.path.join(outputDir,"generated_masks","pred_mask_" + patient_name))
  #Computing the average dice
  if not subject['label'] == "NA":
    average_dice = total_dice/(batch_idx + 1)
    print(average_dice)

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

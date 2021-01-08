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
from GANDLF.losses import *
from GANDLF.utils import *
from .parameterParsing import *

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
    preprocessing = parameters['data_preprocessing']
    which_model = parameters['model']['architecture']
    if not('n_channels' in parameters['model']):
        n_channels = len(headers['channelHeaders'])
    else:
        n_channels = parameters['model']['n_channels']

    base_filters = parameters['model']['base_filters']
    batch_size = parameters['batch_size']
    loss_function = parameters['loss_function']
    scaling_factor = parameters['scaling_factor']
    amp = parameters['amp']
    
    class_list = parameters['model']['class_list']
    n_classList = len(class_list)

    # Defining our model here according to parameters mentioned in the configuration file
    model = get_model(which_model, parameters['model']['dimension'], n_channels, n_classList, base_filters, final_convolution_layer = parameters['model']['final_layer'], psize = psize, batch_size = 1)
    # initialize problem type    
    is_regression, is_classification, is_segmentation = find_problem_type(headers, model.final_convolution_layer)

    # Defining our model here according to parameters mentioned in the configuration file
    model = get_model(which_model, parameters['dimension'], n_channels, n_classList, base_filters, final_convolution_layer = parameters['model']['final_layer'], psize = psize)

    # initialize problem type    
    is_regression, is_classification, is_segmentation = find_problem_type(headers, model.final_convolution_layer)

    if is_regression or is_classification:
        n_classList = len(headers['predictionHeaders']) # ensure the output class list is correctly populated

    # Setting up the inference loader
    inferenceDataForTorch = ImagesFromDataFrame(inferenceDataFromPickle, psize, headers, q_max_length, q_samples_per_volume, q_num_workers, q_verbose, sampler = parameters['patch_sampler'], train = False, augmentations = augmentations, preprocessing = preprocessing)
    inference_loader = DataLoader(inferenceDataForTorch, batch_size=batch_size)

    # Loading the weights into the model
    main_dict = torch.load(os.path.join(outputDir,str(which_model) + "_best_val.pth.tar"))
    model.load_state_dict(main_dict['model_state_dict'])
    
    if not(os.environ.get('HOSTNAME') is None):
        print("\nHostname     :" + str(os.environ.get('HOSTNAME')))
        sys.stdout.flush()

    # get the channel keys for concatenation later (exclude non numeric channel keys)
    batch = next(iter(inference_loader))
    all_keys = list(batch.keys())
    channel_keys = []
    value_keys = []

    for item in all_keys:
        if item.isnumeric():
            channel_keys.append(item)
        elif 'value' in item:
            value_keys.append(item)

    print("Data Samples: ", len(inference_loader.dataset))
    sys.stdout.flush()
    model, amp, device = send_model_to_device(model, amp, device, optimizer=None)
    
    # print stats
    print('Using device:', device)
    sys.stdout.flush()

    # get loss function
    loss_fn, MSE_requested = get_loss(loss_function)

    average_dice, average_loss = get_metrics_save_mask(model, device, inference_loader, psize, channel_keys, value_keys, class_list, loss_fn, is_segmentation, scaling_factor = scaling_factor, weights = None, save_mask = True, outputDir = outputDir, with_roi = True)
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

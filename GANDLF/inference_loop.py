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
from tqdm import tqdm
import time
import sys
import ast 
import pickle
from pathlib import Path
import argparse
import datetime
import SimpleITK as sitk
from torch.cuda.amp import autocast
from GANDLF.data.ImagesFromDataFrame import ImagesFromDataFrame
from GANDLF.schd import *
from GANDLF.losses import *
from GANDLF.utils import *
from .parameterParsing import *

def inferenceLoopRad(inferenceDataFromPickle, headers, device, parameters, outputDir):
    '''
    This is the main inference loop
    '''
    # extract variables form parameters dict
    patch_size = parameters['patch_size']
    q_max_length = parameters['q_max_length']
    q_samples_per_volume = parameters['q_samples_per_volume']
    q_num_workers = parameters['q_num_workers']
    q_verbose = parameters['q_verbose']
    augmentations = parameters['data_augmentation']
    preprocessing = parameters['data_preprocessing']
    which_model = parameters['model']['architecture']
    if not('num_channels' in parameters['model']):
        num_channels = len(headers['channelHeaders'])
    else:
        num_channels = parameters['model']['num_channels']

    base_filters = parameters['model']['base_filters']
    amp = parameters['model']['amp']
    class_list = parameters['model']['class_list']
    
    batch_size = parameters['batch_size']
    loss_function = parameters['loss_function']

    scaling_factor = parameters['scaling_factor']
    n_classList = len(class_list)

    # Defining our model here according to parameters mentioned in the configuration file
    print("Number of dims     : ", parameters['model']['dimension'])
    if 'num_channels' in parameters['model']:
        print("Number of channels : ", parameters['model']['num_channels'])
    print("Number of classes  : ", n_classList)
    model = get_model(which_model, parameters['model']['dimension'], num_channels, n_classList, base_filters, final_convolution_layer = parameters['model']['final_layer'], patch_size = patch_size, batch_size = 1)
    # initialize problem type    
    is_regression, is_classification, is_segmentation = find_problem_type(headers, model.final_convolution_layer)

    # initialize problem type    
    is_regression, is_classification, is_segmentation = find_problem_type(headers, model.final_convolution_layer)

    if is_regression or is_classification:
        n_classList = len(headers['predictionHeaders']) # ensure the output class list is correctly populated

    # Setting up the inference loader
    inferenceDataForTorch = ImagesFromDataFrame(inferenceDataFromPickle, patch_size, headers, q_max_length, q_samples_per_volume, q_num_workers, q_verbose, sampler = parameters['patch_sampler'], train = False, augmentations = augmentations, preprocessing = preprocessing)
    inference_loader = DataLoader(inferenceDataForTorch, batch_size=batch_size)

    # Loading the weights into the model
    main_dict = outputDir
    if os.path.isdir(outputDir):
        file_to_check = os.path.join(outputDir,str(which_model) + "_best_val.pth.tar")
        if not os.path.isfile(file_to_check):
            file_to_check = os.path.join(outputDir,str(which_model) + "_best_train.pth.tar")
    else:
        file_to_check = outputDir
    main_dict = torch.load(file_to_check)
    model.load_state_dict(main_dict['model_state_dict'])
    
    if not(os.environ.get('HOSTNAME') is None):
        print("\nHostname     :" + str(os.environ.get('HOSTNAME')), flush=True)

    # get the channel keys for concatenation later (exclude non numeric channel keys)
    batch = next(iter(inference_loader))
    channel_keys = list(batch.keys())
    channel_keys_new = []
    value_keys = []
    for item in channel_keys:
        if item.isnumeric():
            channel_keys_new.append(item)
        elif 'value' in item:
            value_keys.append(item)
    channel_keys = channel_keys_new

    print("Data Samples: ", len(inference_loader.dataset), flush=True)
    model, amp, device = send_model_to_device(model, amp, device, optimizer=None)
    
    # print stats
    print('Using device:', device, flush=True)

    # get loss function
    loss_fn, MSE_requested = get_loss(loss_function)

    model.eval()
    average_dice, average_loss = get_metrics_save_mask(model, device, inference_loader, patch_size, channel_keys, value_keys, class_list, loss_fn, is_segmentation, scaling_factor=scaling_factor, weights=None, save_mask=True, outputDir=outputDir)
    print(average_dice, average_loss)

if os.name != 'nt':
    '''
    path inference is Linux-only because openslide for Windows works only for Python-3.8  whereas pickle5 works only for 3.6 and 3.7
    ''' 
    def inferenceLoopPath(inferenceDataFromPickle, headers, device, parameters, outputDir):
        from GANDLF.inference_dataloader_histopath import InferTumorSegDataset
        from openslide import OpenSlide
        '''
        This is the main inference loop for histopathology
        '''
        # extract variables form parameters dict
        patch_size = parameters['patch_size']
        stride = parameters['stride_size']
        augmentations = parameters['data_augmentation']
        preprocessing = parameters['data_preprocessing']
        which_model = parameters['model']['architecture']
        class_list = parameters['model']['class_list']
        base_filters = parameters['model']['base_filters']
        amp = parameters['model']['amp']
        batch_size = parameters['batch_size']
        num_channels = len(headers['channelHeaders'])
        if not('num_channels' in parameters['model']):
            num_channels = len(headers['channelHeaders'])
        else:
            num_channels = parameters['model']['num_channels']
        n_classList = len(class_list)
        # Report the time stamp
        training_start_time = time.asctime()
        startstamp = time.time()
        print("\nHostname   :" + str(os.getenv("HOSTNAME")))
        print("\nStart Time :" + str(training_start_time))
        print("\nStart Stamp:" + str(startstamp))
        sys.stdout.flush()

        # PRINT PARSED ARGS
        print("\n\n")
        print("Output directory        :", outputDir)
        if 'num_channels' in parameters['model']:
            print("Number of channels      :", parameters['model']['num_channels'])
        print("Modalities              :", parameters['modality'])
        #print("Number of classes       :", parameters['modality']['num_classes']
        print("Batch Size              :", parameters['batch_size'])
        print("Patch Size              :", parameters['patch_size'])
        print("Sampling Stride         :", parameters['stride_size'])
        print("Base Filters            :", parameters['model']['base_filters'])
        #print("Load Weights            :", parameters['load_weights'])
        sys.stdout.flush()
        # We generate CSV for training if not provided
        print("Reading CSV Files")
        n_classList = len(class_list)
        #test_csv = parameters['test_csv']

        # Defining our model here according to parameters mentioned in the configuration file
        print("Number of dims     : ", parameters['model']['dimension'])
        if 'num_channels' in parameters['model']:
            print("Number of channels : ", parameters['model']['num_channels'])
        print("Number of classes  : ", n_classList)
        model = get_model(which_model, num_dimensions=parameters['model']['dimension'], num_channels=num_channels, num_classes=n_classList,
                          base_filters=base_filters, final_convolution_layer=parameters['model']['final_layer'], patch_size=patch_size, batch_size=batch_size)

        # Loading the weights into the model
        main_dict = torch.load(os.path.join(outputDir, str(which_model) + "_best_val.pth.tar"))
        model.load_state_dict(main_dict['model_state_dict'])
        print('Loaded Weights successfully.')
        sys.stdout.flush()

        model, amp, device = send_model_to_device(model, amp, device, optimizer=None)

        model.eval()
        # print stats
        print('Using device:', device)
        sys.stdout.flush()

        test_df = inferenceDataFromPickle 
        # Patch blocks

        for index, row in test_df.iterrows():
            subject_name = row[headers['subjectIDHeader']]
            print("Patient Slide       : ", row[headers['subjectIDHeader']])
            print("Patient Location    : ", row[headers['channelHeaders']])
            print(row[headers['channelHeaders']].values[0])
            os_image = OpenSlide(row[headers['channelHeaders']].values[0])
            level_width, level_height = os_image.level_dimensions[int(parameters['slide_level'])]
            subject_dest_dir = os.path.join(outputDir, subject_name)
            os.makedirs(subject_dest_dir, exist_ok=True)

            probs_map = np.zeros((level_height, level_width), dtype=np.float16)
            count_map = np.zeros((level_height, level_width), dtype=np.uint8)

            patient_dataset_obj = InferTumorSegDataset(row[headers['channelHeaders']].values[0],
                                                    patch_size=patch_size,
                                                    stride_size=stride,
                                                    selected_level=parameters['slide_level'],
                                                    mask_level=4)

            dataloader = DataLoader(patient_dataset_obj,
                                    batch_size=int(parameters['batch_size']),
                                    shuffle=False, num_workers=2)
            for image_patches, (x_coords, y_coords) in tqdm(dataloader):
                x_coords, y_coords = y_coords.numpy(), x_coords.numpy()
                if params['amp']:
                    with autocast():
                        output = model(image_patches.half().cuda())
                else:
                    output = model(image_patches.half().cuda())
                output = output.cpu().detach().numpy()
                for i in range(int(output.shape[0])):
                    count_map[x_coords[i]:x_coords[i]+patch_size[0],
                              y_coords[i]:y_coords[i]+patch_size[1]] += 1
                    probs_map[x_coords[i]:x_coords[i]+patch_size[0],
                              y_coords[i]:y_coords[i]+patch_size[1]] += output[i][0]
            probs_map = probs_map/count_map
            count_map = (count_map/count_map.max())
            out = count_map*probs_map
            count_map = np.array(count_map*255, dtype=np.uint16)
            out_thresh = np.array((out > 0.5)*255, dtype=np.uint16)
            imsave(os.path.join(subject_dest_dir, row[headers['subjectIDHeader']]+'_prob.png'), out)
            imsave(os.path.join(subject_dest_dir, row[headers['subjectIDHeader']]+'_seg.png'), out_thresh)
            imsave(os.path.join(subject_dest_dir, row[headers['subjectIDHeader']]+'_count.png'), count_map)
        #average_dice, average_loss = get_metrics_save_mask(model, device, inference_loader, patch_size, channel_keys, value_keys, class_list, loss_fn, is_segmentation, scaling_factor = scaling_factor, weights = None, save_mask = True, outputDir = outputDir, with_roi = True)
        #print('Average dice: ', average_dice, '; Average loss: ', average_loss, flush=True)

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
    patch_size = pickle.load(open(args.patch_size_pickle,"rb"))
    headers = pickle.load(open(args.headers_pickle,"rb"))
    label_header = pickle.load(open(args.label_header_pickle,"rb"))
    parameters = pickle.load(open(args.parameter_pickle,"rb"))
    inferenceDataFromPickle = pd.read_pickle(args.inference_loader_pickle)

    if parameters['modalities'] == 'rad':
        inferenceLoopRad(inference_loader_pickle=inferenceDataFromPickle, 
                         headers=headers, 
                         parameters=parameters,
                         outputDir=args.outputDir,
                         device=args.device)
    elif parameters['modalities'] == 'path':
        if os.name != 'nt':
            inferenceLoopPath(inference_loader_pickle=inferenceDataFromPickle, 
                              headers=headers, 
                              parameters=parameters,
                              outputDir=args.outputDir,
                              device=args.device)
    else:
        print('Please Select a modality between rad and path. Correct the option in the config file.')
        sys.exit(0)

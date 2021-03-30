import os, sys
from datetime import datetime
os.environ['TORCHIO_HIDE_CITATION_PROMPT'] = '1' # hides torchio citation request, see https://github.com/fepegar/torchio/issues/235
import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
import torch.nn as nn
import torchio
from GANDLF.losses import *
from torch.utils.data import DataLoader
from pathlib import Path


def one_hot(segmask_array, class_list):
    '''
    This function creates a one-hot-encoded mask from the segmentation mask array and specified class list
    '''
    batch_size = segmask_array.shape[0]
    batch_stack = []
    for b in range(batch_size):
        one_hot_stack = []
        segmask_array_iter = segmask_array[b,0]
        bin_mask = (segmask_array_iter == 0) # initialize bin_mask
        for _class in class_list: # this implementation allows users to combine logical operands 
            if isinstance(_class, str):
                if '||' in _class: # special case
                    class_split = _class.split('||')
                    bin_mask = (segmask_array_iter == int(class_split[0]))
                    for i in range(1,len(class_split)):
                        bin_mask = bin_mask | (segmask_array_iter == int(class_split[i]))
                elif '|' in _class: # special case
                    class_split = _class.split('|')
                    bin_mask = (segmask_array_iter == int(class_split[0]))
                    for i in range(1,len(class_split)):
                        bin_mask = bin_mask | (segmask_array_iter == int(class_split[i]))
                else:
                    # assume that it is a simple int
                    bin_mask = (segmask_array_iter == int(_class)) 
            else:
                bin_mask = (segmask_array_iter == int(_class))
            one_hot_stack.append(bin_mask)
        one_hot_stack = torch.stack(one_hot_stack)
        batch_stack.append(one_hot_stack)
    batch_stack = torch.stack(batch_stack)    
    return batch_stack

def reverse_one_hot(predmask_array,class_list):
    '''
    This function creates a full segmentation mask array from a one-hot-encoded mask and specified class list
    '''
    idx_argmax  = np.argmax(predmask_array,axis=0)
    final_mask = 0
    special_cases_to_check = ['||'] 
    special_case_detected = False
    max = 0
    
    for _class in class_list:
        for case in special_cases_to_check:
            if isinstance(_class, str):
                if case in _class: # check if any of the special cases are present
                    special_case_detected = True
                    class_split = _class.split(case) # if present, then split the sub-class
                    for i in class_split: # find the max for computation later on
                        if int(i) > max:
                            max = int(i)
    
    if special_case_detected:
        start_idx = 0
        if (class_list[0] == 0) or (class_list[0] == '0'):
            start_idx = 1
        
        final_mask = np.asarray(predmask_array[start_idx,:,:,:], dtype=int) # predmask_array[0,:,:,:].long()
        start_idx += 1
        for i in range(start_idx,len(class_list)):
            final_mask += np.asarray(predmask_array[0,:,:,:], dtype=int) # predmask_array[i,:,:,:].long()
            # temp_sum = torch.sum(output)
        # output_2 = (max - torch.sum(output)) % max 
        # test_2 = 1
    else:        
        for idx, _class in enumerate(class_list):
            final_mask = final_mask +  (idx_argmax == idx)*_class
    return final_mask


def checkPatchDivisibility(patch_size, number = 16):
    '''
    This function checks the divisibility of a numpy array or integer for architectural integrity
    '''
    if isinstance(patch_size, int):
        patch_size_to_check = np.array(patch_size)
    else:
        patch_size_to_check = patch_size
    if patch_size_to_check[-1] == 1: # for 2D, don't check divisibility of last dimension
        patch_size_to_check = patch_size_to_check[:-1]
    if np.count_nonzero(np.remainder(patch_size_to_check, number)) > 0:
        return False
    return True


def send_model_to_device(model, amp, device, optimizer):
    '''
    This function reads the environment variable(s) and send model to correct device
    '''
    if device != 'cpu':
        if os.environ.get('CUDA_VISIBLE_DEVICES') is None:
            sys.exit('Please set the environment variable \'CUDA_VISIBLE_DEVICES\' correctly before trying to run GANDLF on GPU')
        
        dev = os.environ.get('CUDA_VISIBLE_DEVICES')
        # multi-gpu support
        # ###
        # # https://discuss.pytorch.org/t/cuda-visible-devices-make-gpu-disappear/21439/17?u=sarthakpati
        # ###
        if ',' in dev:
            device = torch.device('cuda')
            model = nn.DataParallel(model, '[' + dev + ']')
        else:
            print('Device requested via CUDA_VISIBLE_DEVICES: ', dev)
            print('Total number of CUDA devices: ', torch.cuda.device_count())

            # if only a single visible device, it will be indexed as '0'
            if torch.cuda.device_count() == 1:
                dev = '0'

            dev_int = int(dev)
            print('Device finally used: ', dev)
            # device = torch.device('cuda:' + dev)
            device = torch.device('cuda')
            print('Sending model to aforementioned device')
            model = model.to(device)
            print('Memory Total : ', round(torch.cuda.get_device_properties(dev_int).total_memory/1024**3, 1), 'GB, Allocated: ', round(torch.cuda.memory_allocated(dev_int)/1024**3, 1),'GB, Cached: ',round(torch.cuda.memory_reserved(dev_int)/1024**3, 1), 'GB' )
        
        print("Device - Current: %s Count: %d Name: %s Availability: %s"%(torch.cuda.current_device(), torch.cuda.device_count(), torch.cuda.get_device_name(device), torch.cuda.is_available()))
     
        if not(optimizer is None):
            # ensuring optimizer is in correct device - https://github.com/pytorch/pytorch/issues/8741
            optimizer.load_state_dict(optimizer.state_dict())

    else:
        dev = -1
        device = torch.device('cpu')
        model.cpu()
        amp = False
        print("Since Device is CPU, Mixed Precision Training is set to False")

    return model, amp, device

def resize_image(input_image, output_size, interpolator = sitk.sitkLinear):
    '''
    This function resizes the input image based on the output size and interpolator
    '''
    inputSize = input_image.GetSize()
    inputSpacing = np.array(input_image.GetSpacing())
    outputSpacing = np.array(inputSpacing)

    if (len(output_size) != len(inputSpacing)):
        sys.exit('The output size dimension is inconsistent with the input dataset, please check parameters.')

    for i in range(len(output_size)):
        outputSpacing[i] = inputSpacing[i] * (inputSize[i] / output_size[i])
    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(interpolator)
    resampler.SetSize(output_size)
    resampler.SetOutputSpacing(outputSpacing)
    resampler.SetOutputOrigin(input_image.GetOrigin())
    resampler.SetOutputDirection(input_image.GetDirection())
    resampler.SetDefaultPixelValue(0)
    return resampler.Execute(input_image)

def get_metrics_save_mask(model, device, loader, psize, channel_keys, value_keys, class_list, loss_fn, is_segmentation, scaling_factor = 1, weights = None, save_mask = False, outputDir = None, with_roi = False, ignore_label_validation = None):
    '''
    This function gets various statistics from the specified model and data loader
    '''
    # # if no weights are specified, use 1
    # if weights is None:
    #     weights = [1]
    #     for i in range(len(class_list) - 1):
    #         weights.append(1)
    Path(outputDir).mkdir(parents=True, exist_ok=True)
    outputToWrite = 'SubjectID,PredictedValue\n'
    model.eval()
    with torch.no_grad():
        total_loss = total_dice = 0
        for batch_idx, (subject) in enumerate(loader):
            # constructing a new dict because torchio.GridSampler requires torchio.Subject, which requires torchio.Image to be present in initial dict, which the loader does not provide
            subject_dict = {}
            if ('label' in subject):
                if (subject['label'] != ['NA']):
                    subject_dict['label'] = torchio.Image(subject['label']['path'], type = torchio.LABEL)
            
            for key in value_keys: # for regression/classification
                subject_dict['value_' + key] = subject[key]

            for key in channel_keys:
                subject_dict[key] = torchio.Image(subject[key]['path'], type=torchio.INTENSITY)
            grid_sampler = torchio.inference.GridSampler(torchio.Subject(subject_dict), psize)
            patch_loader = torch.utils.data.DataLoader(grid_sampler, batch_size=1)
            aggregator = torchio.inference.GridAggregator(grid_sampler)

            pred_output = 0 # this is used for regression
            for patches_batch in patch_loader:
                image = torch.cat([patches_batch[key][torchio.DATA] for key in channel_keys], dim=1)
                if len(value_keys) > 0:
                    valuesToPredict = torch.cat([patches_batch['value_' + key] for key in value_keys], dim=0)
                locations = patches_batch[torchio.LOCATION]
                image = image.float().to(device)
                ## special case for 2D            
                if image.shape[-1] == 1:
                    model_2d = True
                    image = torch.squeeze(image, -1)
                    locations = torch.squeeze(locations, -1)
                else:
                    model_2d = False
                
                if is_segmentation: # for segmentation, get the predicted mask
                    pred_mask = model(image)
                    if model_2d:
                        pred_mask = pred_mask.unsqueeze(-1)
                else: # for regression/classification, get the predicted output and add it together to average later on
                    pred_output += model(image)
                
                if is_segmentation: # aggregate the predicted mask
                    aggregator.add_batch(pred_mask, locations)
            
            if is_segmentation:
                pred_mask = aggregator.get_output_tensor()
                pred_mask = pred_mask.cpu() # the validation is done on CPU
                pred_mask = pred_mask.unsqueeze(0) # increasing the number of dimension of the mask
            else:
                pred_output = pred_output / len(locations) # average the predicted output across patches
                pred_output = pred_output.cpu()
                # loss = loss_fn(pred_output.double(), valuesToPredict.double(), len(class_list), weights).cpu().data.item() # this would need to be customized for regression/classification
                loss = torch.nn.MSELoss()(pred_output.double(), valuesToPredict.double()).cpu().data.item() # this needs to be revisited for multi-class output
                total_loss += loss
            
            first = next(iter(subject['label']))
            if is_segmentation:
                if first == 'NA':
                    print("Ground Truth Mask not found. Generating the Segmentation based one the METADATA of one of the modalities, The Segmentation will be named accordingly")
                mask = subject_dict['label'][torchio.DATA] # get the label image
                if mask.dim() == 4:
                    mask = mask.unsqueeze(0) # increasing the number of dimension of the mask
                mask = one_hot(mask, class_list)        
                loss = loss_fn(pred_mask.double(), mask.double(), len(class_list), weights).cpu().data.item() # this would need to be customized for regression/classification
                total_loss += loss
                #Computing the dice score 
                curr_dice = MCD(pred_mask.double(), mask.double(), len(class_list), ignore_class=ignore_label_validation).cpu().data.item()
                #Computing the total dice
                total_dice += curr_dice
                
            if save_mask:
                patient_name = subject['subject_id'][0]

                if is_segmentation:
                    path_to_metadata = subject['path_to_metadata'][0]
                    inputImage = sitk.ReadImage(path_to_metadata)
                    _, ext = os.path.splitext(path_to_metadata)
                    if ext == '.gz':
                        ext = '.nii.gz'
                    pred_mask = pred_mask.numpy()
                    pred_mask = reverse_one_hot(pred_mask[0],class_list)
                    if not(model_2d):
                        result_image = sitk.GetImageFromArray(np.swapaxes(pred_mask,0,2))
                    else:
                        result_image = pred_mask
                    result_image.CopyInformation(inputImage)
                    # if parameters['resize'] is not None:
                    #     result_image = resize_image(resize_image, inputImage.GetSize(), sitk.sitkNearestNeighbor) # change this for resample
                    sitk.WriteImage(result_image, os.path.join(outputDir, patient_name + '_seg' + ext))
                elif len(value_keys) > 0:
                    outputToWrite += patient_name + ',' + str(pred_output / scaling_factor) + '\n'
        
        if len(value_keys) > 0:
            file = open(os.path.join(outputDir,"output_predictions.csv"), 'w')
            file.write(outputToWrite)
            file.close()

        # calculate average loss and dice
        avg_loss = total_loss/len(loader.dataset)
        if is_segmentation:
            avg_dice = total_dice/len(loader.dataset)
        else:
            avg_dice = 1 # we don't care about this for regression/classification
        return avg_dice, avg_loss

def fix_paths(cwd):
    '''
    This function takes the current working directory of the script (which is required for VIPS) and sets up all the paths correctly
    '''
    if os.name == 'nt': # proceed for windows
        vipshome = os.path.join(cwd, 'vips/vips-dev-8.10/bin')
        os.environ['PATH'] = vipshome + ';' + os.environ['PATH']

def find_problem_type(headersFromCSV, model_final_layer):
    '''
    This function determines the type of problem at hand - regression, classification or segmentation
    '''    
    # initialize problem type    
    is_regression = False
    is_classification = False
    is_segmentation = False

    # check if regression/classification has been requested
    if len(headersFromCSV['predictionHeaders']) > 0:
        if model_final_layer is None:
            is_regression = True
        else:
            is_classification = True
    else:
        is_segmentation = True
    
    return is_regression, is_classification, is_segmentation

def writeTrainingCSV(inputDir, channelsID, labelID, outputFile):
    '''
    This function writes the CSV file based on the input directory, channelsID + labelsID strings
    '''
    channelsID_list = channelsID.split(',') # split into list
    
    outputToWrite = 'SubjectID,'
    for i in range(len(channelsID_list)):
        outputToWrite = outputToWrite + 'Channel_' + str(i) + ','
    outputToWrite = outputToWrite + 'Label'
    outputToWrite = outputToWrite + '\n'
    
    # iterate over all subject directories
    for dirs in os.listdir(inputDir):
        currentSubjectDir = os.path.join(inputDir, dirs)
        if os.path.isdir(currentSubjectDir):
            outputToWrite = outputToWrite + dirs + ','
            if os.path.isdir(currentSubjectDir): # only consider folders
                filesInDir = os.listdir(currentSubjectDir) # get all files in each directory
                maskFile = ''
                allImageFiles = ''
                for channel in channelsID_list:
                    for i in range(len(filesInDir)):
                        currentFile = os.path.abspath(os.path.join(currentSubjectDir, filesInDir[i]))
                        currentFile = currentFile.replace('\\', '/')
                        if channel in filesInDir[i]:
                            allImageFiles += currentFile + ','            
                        elif labelID in filesInDir[i]:
                            maskFile = currentFile 
                outputToWrite += allImageFiles + maskFile + '\n'

    file = open(outputFile, 'w')
    file.write(outputToWrite)
    file.close()

def parseTrainingCSV(inputTrainingCSVFile):
    '''
    This function parses the input training CSV and returns a dictionary of headers and the full (randomized) data frame
    '''
    ## read training dataset into data frame
    data_full = pd.read_csv(inputTrainingCSVFile)
    # shuffle the data - this is a useful level of randomization for the training process
    data_full=data_full.sample(frac=1).reset_index(drop=True)

    # find actual header locations for input channel and label
    # the user might put the label first and the channels afterwards 
    # or might do it completely randomly
    headers = {}
    headers['channelHeaders'] = []
    headers['predictionHeaders'] = []
    headers['labelHeader'] = None
    headers['subjectIDHeader'] = None

    for col in data_full.columns: 
        # add appropriate headers to read here, as needed
        col_lower = col.lower()
        currentHeaderLoc = data_full.columns.get_loc(col)
        if ('channel' in col_lower) or ('modality' in col_lower) or ('image' in col_lower):
            headers['channelHeaders'].append(currentHeaderLoc)
        elif ('valuetopredict' in col_lower):
            headers['predictionHeaders'].append(currentHeaderLoc)
        elif ('subject' in col_lower) or ('patient' in col_lower) or ('pid' in col_lower):
            headers['subjectIDHeader'] = currentHeaderLoc
        elif ('label' in col_lower) or ('mask' in col_lower) or ('segmentation' in col_lower) or ('ground_truth' in col_lower) or ('groundtruth' in col_lower):
            if (headers['labelHeader'] == None):
                headers['labelHeader'] = currentHeaderLoc
            else:
                print('WARNING: Multiple label headers found in training CSV, only the first one will be used', file = sys.stderr)
    return data_full, headers
    

def get_date_time():
    now = datetime.now()
    date_string = now.strftime("%d/%m/%Y %H:%M:%S")
    return now

# def get_class_imbalance_weights(trainingDataFromPickle, parameters, headers, is_regression, classList):
#     '''
#     This function calculates the penalty that is used for validation loss in multi-class problems
#     '''
#     dice_weights_dict = {} # average for "weighted averaging"
#     dice_penalty_dict = {} # penalty for misclassification
#     for i in range(1, classList):
#         dice_weights_dict[i] = 0
#         dice_penalty_dict[i] = 0
#     # define a seaparate data loader for penalty calculations
#     penaltyData = ImagesFromDataFrame(trainingDataFromPickle, parameters['psize'], headers, train=False) 
#     penalty_loader = DataLoader(penaltyData, batch_size=1)
    
#     # get the weights for use for dice loss
#     total_nonZeroVoxels = 0
    
#     # For regression dice penalty need not be taken account
#     # For classification this should be calculated on the basis of predicted labels and mask
#     if not is_regression:
#         for batch_idx, (subject) in enumerate(penalty_loader): # iterate through full training data
#             # accumulate dice weights for each label
#             mask = subject['label'][torchio.DATA]
#             one_hot_mask = one_hot(mask, classList)
#             for i in range(1, len(classList)):
#                 currentNumber = torch.nonzero(one_hot_mask[:,i,:,:,:], as_tuple=False).size(0)
#                 dice_weights_dict[i] = dice_weights_dict[i] + currentNumber # class-specific non-zero voxels
#                 total_nonZeroVoxels = total_nonZeroVoxels + currentNumber # total number of non-zero voxels to be considered
            
#             # get the penalty values - dice_weights contains the overall number for each class in the training data
#         for i in range(1, len(classList)):
#             penalty = total_nonZeroVoxels # start with the assumption that all the non-zero voxels make up the penalty
#             for j in range(1, len(classList)):
#                 if i != j: # for differing classes, subtract the number
#                     penalty = penalty - dice_penalty_dict[j]
            
#             dice_penalty_dict[i] = penalty / total_nonZeroVoxels # this is to be used to weight the loss function
#         # dice_weights_dict[i] = 1 - dice_weights_dict[i]# this can be used for weighted averaging
#     return dice_penalty_dict
        

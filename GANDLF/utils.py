import os
os.environ['TORCHIO_HIDE_CITATION_PROMPT'] = '1' # hides torchio citation request, see https://github.com/fepegar/torchio/issues/235
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn as nn
import torchio
from GANDLF.losses import *
import sys
import os

def one_hot(segmask_array, class_list):
    batch_size = segmask_array.shape[0]
    batch_stack = []
    for b in range(batch_size):
        one_hot_stack = []
        segmask_array_iter = segmask_array[b,0]
        for class_ in class_list:
            bin_mask = (segmask_array_iter == int(class_))
            one_hot_stack.append(bin_mask)
        one_hot_stack = torch.stack(one_hot_stack)
        batch_stack.append(one_hot_stack)
    batch_stack = torch.stack(batch_stack)    
    return batch_stack

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

def reverse_one_hot(predmask_array,class_list):
    idx_argmax  = np.argmax(predmask_array,axis=0)
    final_mask = 0
    for idx, class_ in enumerate(class_list):
        final_mask = final_mask +  (idx_argmax == idx)*class_
    return final_mask

def send_model_to_device(model, ampInput, device, optimizer):
    '''
    This function reads the environment variable(s) and send model to correct device
    '''
    amp = ampInput
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
            if (torch.cuda.device_count() == 1) and (int(dev) == 1): # this should be properly fixed
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
    resampler.SetSize(output_size)
    resampler.SetOutputSpacing(outputSpacing)
    resampler.SetOutputOrigin(input_image.GetOrigin())
    resampler.SetOutputDirection(input_image.GetDirection())
    resampler.SetDefaultPixelValue(0)
    return resampler.Execute(input_image)

def get_metrics_save_mask(model, device, loader, psize, channel_keys, value_keys, class_list, loss_fn, weights = None, save_mask = False, outputDir = None, is_segmentation = True):
    '''
    This function gets various statistics from the specified model and data loader
    '''
    # if no weights are specified, use 1
    if weights is None:
        weights = [1]
        for i in range(len(class_list) - 1):
            weights.append(1)

    model.eval()
    with torch.no_grad():
        total_loss = total_dice = 0
        for batch_idx, (subject) in enumerate(loader):
            subject_dict = {}
            if ('label' in subject) and (subject_dict['label'] != 'NA'):
                subject_dict['label'] = torchio.Image(subject['label']['path'], type = torchio.LABEL)
            
            for key in value_keys: # for regression/classification
                subject_dict['value_' + key] = torchio.Image(subject[key])

            for key in channel_keys:
                subject_dict[key] = torchio.Image(subject[key]['path'], type=torchio.INTENSITY)
            grid_sampler = torchio.inference.GridSampler(torchio.Subject(subject_dict), psize)
            patch_loader = torch.utils.data.DataLoader(grid_sampler, batch_size=1)
            aggregator = torchio.inference.GridAggregator(grid_sampler)

            pred_output = 0 # this is used for regression
            for patches_batch in patch_loader:
                image = torch.cat([patches_batch[key][torchio.DATA] for key in channel_keys], dim=1)
                valuesToPredict = torch.cat([patches_batch[key] for key in value_keys], dim=0)
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
                pred_mask.cpu() # the validation is done on CPU, see https://github.com/FETS-AI/GANDLF/issues/270
                pred_mask = pred_mask.unsqueeze(0) # increasing the number of dimension of the mask
            else:
                pred_output = pred_output / len(locations) # average the predicted output across patches
                loss = loss_fn(pred_output.double(), valuesToPredict.double(), len(class_list), weights).cpu().data.item() # this would need to be customized for regression/classification
                total_loss += loss

            if not subject['label'] == "NA":
                mask = subject_dict['label'][torchio.DATA] # get the label image
                if mask.dim() == 4:
                    mask = mask.unsqueeze(0) # increasing the number of dimension of the mask
                mask = one_hot(mask, class_list)        
                loss = loss_fn(pred_mask.double(), mask.double(), len(class_list), weights).cpu().data.item() # this would need to be customized for regression/classification
                total_loss += loss
                #Computing the dice score 
                curr_dice = MCD(pred_mask.double(), mask.double(), len(class_list)).cpu().data.item()
                #Computing the total dice
                total_dice += curr_dice
            else:
                if not (is_segmentation):
                    avg_dice = 1 # we don't care about this for regression/classification
                    avg_loss = total_loss/len(loader.dataset)
                    return avg_dice, avg_loss
                else:
                    print("Ground Truth Mask not found. Generating the Segmentation based one the METADATA of one of the modalities, The Segmentation will be named accordingly")
            if save_mask:
                inputImage = sitk.ReadImage(subject['path_to_metadata'])
                pred_mask = pred_mask.numpy()
                pred_mask = reverse_one_hot(pred_mask[0],class_list)
                result_image = sitk.GetImageFromArray(np.swapaxes(pred_mask,0,2))
                result_image.CopyInformation(inputImage)
                # if parameters['resize'] is not None:
                #     originalSize = inputImage.GetSize()
                #     result_image = resize_image(resize_image, originalSize, sitk.sitkNearestNeighbor)
        
                patient_name = os.path.basename(subject['path_to_metadata'])
                if not os.path.isdir(os.path.join(outputDir,"generated_masks")):
                    os.mkdir(os.path.join(outputDir,"generated_masks"))
                sitk.WriteImage(result_image, os.path.join(outputDir,"generated_masks","pred_mask_" + patient_name))
        if (subject['label'] != "NA"):
            avg_dice, avg_loss = total_dice/len(loader.dataset), total_loss/len(loader.dataset)
            return avg_dice, avg_loss
        else:
            print("WARNING: No Ground Truth Label provided, returning metrics as NONE")
            return None, None

def fix_paths(cwd):
    '''
    This function takes the current working directory of the script (which is required for VIPS) and sets up all the paths correctly
    '''
    if os.name == 'nt': # proceed for windows
        vipshome = os.path.join(cwd, 'vips/vips-dev-8.10/bin')
        os.environ['PATH'] = vipshome + ';' + os.environ['PATH']

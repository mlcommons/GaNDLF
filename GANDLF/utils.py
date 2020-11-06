import numpy as np
import SimpleITK as sitk
import torch
import torchio
from GANDLF.losses import *
import sys

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
    if np.count_nonzero(np.remainder(patch_size, number)) > 0:
        sys.exit('The \'patch_size\' should be divisible by ' + str(number) + ' for unet-like models')

def reverse_one_hot(predmask_array,class_list):
    idx_argmax  = np.argmax(predmask_array,axis=0)
    final_mask = 0
    for idx, class_ in enumerate(class_list):
        final_mask = final_mask +  (idx_argmax == idx)*class_
    return final_mask


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

def get_metrics_save_mask(model, loader, psize, channel_keys, class_list, loss_fn, weights = None, save_mask = False):
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
            grid_sampler = torchio.inference.GridSampler(subject , psize)
            patch_loader = torch.utils.data.DataLoader(grid_sampler, batch_size=1)
            aggregator = torchio.inference.GridAggregator(grid_sampler)
            for patches_batch in patch_loader:
                image = torch.cat([patches_batch[key][torchio.DATA] for key in channel_keys], dim=1).cuda()
                locations = patches_batch[torchio.LOCATION]
                ## special case for 2D            
                if image.shape[-1] == 1:
                    model_2d = True
                    image = torch.squeeze(image, -1)
                    locations = torch.squeeze(locations, -1)
                pred_mask = model(image)
                if model_2d:
                    pred_mask = pred_mask.unsqueeze(-1)
                aggregator.add_batch(pred_mask, locations)
            pred_mask = aggregator.get_output_tensor()
            pred_mask = pred_mask.unsqueeze(0) # increasing the number of dimension of the mask
            if not subject['label'] == "NA":
                mask = subject['label'][torchio.DATA] # get the label image
                mask = mask.unsqueeze(0) # increasing the number of dimension of the mask
                mask = one_hot(mask, class_list)
                # making sure that the output and mask are on the same device
                pred_mask, mask = pred_mask.cuda(), mask.cuda()
                loss = loss_fn(pred_mask.double(), mask.double(), len(class_list), weights).cpu().data.item() # this would need to be customized for regression/classification
                total_loss += loss
                #Computing the dice score 
                curr_dice = MCD(pred_mask.double(), mask.double(), len(class_list)).cpu().data.item()
                #Computing the total dice
                total_dice += curr_dice
            else:
                print("Ground Truth Mask not found. Generating the Segmentation based one the METADATA of one of the modalities, The Segmentation will be named accordingly")
            if save_mask:
                inputImage = sitk.ReadImage(subject['path_to_metadata'])
                pred_mask = pred_mask.cpu().numpy()
                pred_mask = reverse_one_hot(pred_mask[0],class_list)
                result_image = sitk.GetImageFromArray(np.swapaxes(pred_mask,0,2))
                result_image.CopyInformation(inputImage)
                if parameters['resize'] is not None:
                    originalSize = inputImage.GetSize()
                    result_image = resize_image(resize_image, originalSize, sitk.sitkNearestNeighbor)
        
                patient_name = os.path.basename(subject['path_to_metadata'])
                if not os.path.isdir(os.path.join(outputDir,"generated_masks")):
                    os.mkdir(os.path.join(outputDir,"generated_masks"))
                sitk.WriteImage(result_image, os.path.join(outputDir,"generated_masks","pred_mask_" + patient_name))
        if (subject['label'] != "NA"):
            avg_dice, avg_loss = total_dice/len(loader.dataset), total_loss/len(loader.dataset)
            return avg_dice, avg_loss
        else:
            print("WARNING: No Ground Truth Label provided, returning metris as NONE")
            return None, None


import numpy as np
import SimpleITK as sitk
import torch
import torchio
from GANDLF.losses import *

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
    for i in range(len(output_size)):
        outputSpacing[i] = inputSpacing[i] * (inputSize[i] / output_size[i])
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(output_size)
    resampler.SetOutputSpacing(outputSpacing)
    resampler.SetOutputOrigin(input_image.GetOrigin())
    resampler.SetOutputDirection(input_image.GetDirection())
    resampler.SetDefaultPixelValue(0)
    return resampler.Execute(input_image)

def get_stats(model, loader, psize, channel_keys, class_list, loss_fn, weights = None):
    '''
    This function gets various statistics from the specified model and data loader
    '''
    # if no weights are specified, use 1
    if weights is None:
        weights = [1]
        for i in range(class_list - 1):
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
                pred_mask = model(image)
                #print(image.shape)
                #print(pred_mask.shape)
                aggregator.add_batch(pred_mask, locations)
            pred_mask = aggregator.get_output_tensor()
            pred_mask = pred_mask.unsqueeze(0)
            mask = subject['label'][torchio.DATA] # get the label image
            mask = mask.unsqueeze(0) # increasing the number of dimension of the mask
            mask = one_hot(mask, class_list)
            # making sure that the output and mask are on the same device
            pred_mask, mask = pred_mask.cuda(), mask.cuda()
            loss = loss_fn(pred_mask.double(), mask.double(), len(class_list), weights).cpu().data.item() # this would need to be customized for regression/classification
            total_loss += loss
            #Computing the dice score 
            curr_dice = MCD(pred_mask.double(), mask.double(), len(class_list), weights).cpu().data.item()
            #Computing the total dice
            total_dice+= curr_dice
            #print("Current Dice is: ", curr_dice)
        return total_dice, total_loss


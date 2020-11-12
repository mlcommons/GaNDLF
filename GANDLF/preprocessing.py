import torch
import numpy as np
import SimpleITK as sitk

def threshold_intensities(input_tensor, min_val, max_val):
    '''
    This function takes an input tensor and 2 thresholds, lower & upper and thresholds between them, basically making intensity values outside this range '0'
    '''
    C = torch.zeros(input_tensor.size())
    l1_tensor = torch.where(input_tensor < max_val, input_tensor, C)
    l2_tensor = torch.where(l1_tensor > min_val, l1_tensor, C)
    return l2_tensor

def clip_intensities(input_tensor, min_val, max_val):
    '''
    This function takes an input tensor and 2 thresholds, lower and upper and clips between them, basically making the lowest value as 'min_val' and largest values as 'max_val'
    '''
    return torch.clamp(input_tensor, min_val, max_val)

def resize_image_resolution(input_image, output_size):
    '''
    This function resizes the input image based on the output size and interpolator
    '''
    inputSize = input_image.GetSize()
    outputSpacing = np.array(input_image.GetSpacing())
    for i in range(len(output_size)):
        outputSpacing[i] = outputSpacing[i] * (inputSize[i] / output_size[i])
    return outputSpacing

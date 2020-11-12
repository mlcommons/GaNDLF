import torch
import numpy as np
import SimpleITK as sitk

def threshold_intensities(input_tensor, min_val, max_val):
    '''
    This function takes an input tensor and 2 thresholds, lower & upper and thresholds between them, basically making intensity values outside this range '0'
    '''
    l1_tensor = torch.where(input_tensor < float(min_val), input_tensor, 0)
    l2_tensor = torch.where(l1_tensor > float(max_val), l1_tensor, 0)
    return l2_tensor

def clip_intensities(input_tensor, min_val, max_val):
    '''
    This function takes an input tensor and 2 thresholds, lower and upper and clips between them, basically making the lowest value as 'min_val' and largest values as 'max_val'
    '''
    l1_tensor = torch.where(input_tensor < float(min_val), input_tensor, float(min_val))
    l2_tensor = torch.where(l1_tensor > float(max_val), l1_tensor, float(max_val))
    return l2_tensor

def resize_image_resolution(input_image, output_size):
    '''
    This function resizes the input image based on the output size and interpolator
    '''
    inputSize = input_image.GetSize()
    outputSpacing = np.array(input_image.GetSpacing())
    for i in range(len(output_size)):
        outputSpacing[i] = outputSpacing[i] * (inputSize[i] / output_size[i])
    return outputSpacing

import torch
import numpy as np
import SimpleITK as sitk
import torchvision.transforms as transforms

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


def normalize_by_val(input_tensor, mean, std):
    """
    This function returns the tensor normalized by these particular values
    """
    return transforms.Normalize(mean, std)

def normalize_div_by_255(input_tensor):
    """
    This function divides all values of the input tensor by 255 on all channels
    """
    return normalize_by_val(input_tensor,
                            mean=[0., 0., 0.],
                            std=[1., 1., 1.])

def normalize_standardize(input_tensor):
    """
    This function
    """
    return normalize_by_val(input_tensor,
                            mean=[0.5, 0.5, 0.5],
                            std=[0.5, 0.5, 0.5])

def normalize_imagenet(input_tensor):
    return normalize_by_val(input_tensor,
                            mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])

def resize_image_resolution(input_image, output_size):
    '''
    This function resizes the input image based on the output size and interpolator
    '''
    inputSize = input_image.GetSize()
    outputSpacing = np.array(input_image.GetSpacing())
    for i in range(len(output_size)):
        outputSpacing[i] = outputSpacing[i] * (inputSize[i] / output_size[i])
    return outputSpacing

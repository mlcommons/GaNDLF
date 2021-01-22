import numpy as np
import sys

import torch
import torchio
import SimpleITK as sitk
import nibabel as nib

from torchio.transforms.spatial_transform import SpatialTransform

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


# adapted from https://codereview.stackexchange.com/questions/132914/crop-black-border-of-image-using-numpy/132933#132933
def crop_image_outside_zeros(array, patch_size):
    dimensions = len(array.shape)
    if dimensions != 4:
        raise ValueError("Array expected to be 4D but got {} dimensions.".format(dimensions))
    if (len(patch_size) != 4) or (patch_size[0] != 1):
        raise ValueError("The length of patch_size is expected as 4 with a 1 at index 0 as we do not crop the channels.") 
    
    # collapse to single channel and get the mask of non-zero voxels
    mask = array.sum(axis=0) > 0
    patch_size = patch_size[1:]

    # get the small and large corners
  
    m0 = mask.any(1).any(1)
    m1 = mask.any(0)
    m2 = m1.any(0)
    m1 = m1.any(1)
    
    small = [m0.argmax(), m1.argmax(), m2.argmax()]
    large = [m0[::-1].argmax(), m1[::-1].argmax(), m2[::-1].argmax()]
    large = [m - l for m, l in zip(mask.shape, large)]
    
    # ensure we have a full patch
    # for each axis
    for i in range(3):
        # if less than patch size, extend the small corner out
        if large[i] - small[i] < patch_size[i]:
            small[i] = large[i] - patch_size[i]

        # if bottom fell off array, extend the large corner and set small to 0
        if small[i] < 0:
            small[i] = 0
            large[i] = patch_size[i]

    # calculate pixel location of new bounding box corner (will use to update the reference of the image to physical space)
    new_corner_idxs = np.array([small[0], small[1], small[2]])
    # Get the contents of the bounding box from the array
    new_array = array[:,
                      small[0]:large[0],
                      small[1]:large[1],
                      small[2]:large[2]]
    
    return new_corner_idxs, new_array

# adapted from https://github.com/fepegar/torchio/blob/master/torchio/transforms/preprocessing/spatial/crop.py
class  CropExternalZeroplanes(SpatialTransform):
    """
    Transformation class to enable taking the whole stack (inluding all scan
    modalities as well as a channel for the segmentation label) and removing all 
    external zero planes (zero across all channels).

    Args:
        psize: patch size (has a channel axis dimension which should be made 1) used to ensure we do not crop to smaller size than this)
        **kwargs: See :class:`~torchio.transforms.Transform` for additional
            keyword arguments.
    """

    def __init__(self, patch_size, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
    
    def apply_transform(self, subject):
        subject.check_consistent_spatial_shape()
        # create stack of all images
        images_dict = subject.get_images_dict(intensity_only=False)
        numpy_stack_list = []
        names_list = []

        for name, image in images_dict.items():
            numpy_stack_list.append(image.data.numpy().copy())
            names_list.append(name)
        numpy_stack = np.concatenate(numpy_stack_list, axis=0)

        # crop away the zero planes on the whole stack
        new_corner_idxs, new_stack = crop_image_outside_zeros(array=numpy_stack, patch_size=self.patch_size)

        # recompute origin of affine matrix using first image in dictionary
        # first line in this method checked that all are the same
        example_image_affine = list(images_dict.values())[0].affine
        new_origin = nib.affines.apply_affine(example_image_affine, new_corner_idxs)
        new_affine = example_image_affine.copy()
        new_affine[:3, 3] = new_origin

        # repopulate the subject data with the stack slices
        for idx, array in enumerate(new_stack):

            images_dict[names_list[idx]]['data'] = torch.tensor(np.expand_dims(array, axis=0))
            images_dict[names_list[idx]]['affine'] = new_affine

        return subject

    def is_invertible(self):
        return False
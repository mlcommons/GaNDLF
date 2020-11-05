import torch
import numpy as np
import torchio
from torchio.transforms import (OneOf, RandomMotion, RandomGhosting, RandomSpike,
                                RandomAffine, RandomElasticDeformation,
                                RandomBiasField, RandomBlur,
                                RandomNoise, RandomSwap, ZNormalization,
                                Resample, Compose, Lambda)
from torchio import Image, Subject
import SimpleITK as sitk
from GANDLF.utils import resize_image

import copy

def threshold_intensities(input_tensor, min_val, max_val):
    '''
    This function takes an input tensor and 2 thresholds, lower & upper and thresholds between them, basically making intensity values outside this range '0'
    '''
    l1_tensor = torch.where(input_tensor < min_val, input_tensor, 0)
    l2_tensor = torch.where(l1_tensor > max_val, l1_tensor, 0)
    return l2_tensor

def clip_intensities(input_tensor, min_val, max_val):
    '''
    This function takes an input tensor and 2 thresholds, lower and upper and clips between them, basically making the lowest value as 'min_val' and largest values as 'max_val'
    '''
    l1_tensor = torch.where(input_tensor < min_val, input_tensor, min_val)
    l2_tensor = torch.where(l1_tensor > max_val, l1_tensor, max_val)
    return l2_tensor

## todo: ability to change interpolation type from config file
## todo: ability to change the dimensionality according to the config file
# define individual functions/lambdas for augmentations to handle properties
def mri_artifact(patch_size = None, p = 1):
    return OneOf({RandomMotion(): 0.34, RandomGhosting(): 0.33, RandomSpike(): 0.33}, p=p)

def spatial_transform(patch_size = None, p=1):
    if patch_size is not None:
        num_controls = patch_size
        max_displacement = np.divide(patch_size, 10)
        if patch_size[-1] == 1:
            max_displacement[-1] = 0.1 # ensure maximum displacement is never grater than patch size
    else:
        # use defaults defined in torchio
        num_controls = 7 
        max_displacement = 7.5
    return OneOf({RandomAffine(): 0.8, RandomElasticDeformation(max_displacement = max_displacement): 0.2}, p=p)

def threshold_transform(min, max, p=1):
    return Lambda(lambda x: threshold_intensities(x, min, max))

def clip_transform(min, max, p=1):
    return Lambda(lambda x: clip_intensities(x, min, max))

def bias(patch_size = None, p=1):
    return RandomBiasField(coefficients=0.5, order=3, p=p, seed=None)

def blur(patch_size = None, p=1):
    return RandomBlur(std=(0., 4.), p=p, seed=None)

def noise(patch_size = None, p=1):
    return RandomNoise(mean=0, std=(0, 0.25), p=p, seed=None)

def swap(patch_size = 15, p=1):
    return RandomSwap(patch_size=patch_size, num_iterations=100, p=p, seed=None)

# Defining a dictionary - key is the string and the value is the augmentation object
global_augs_dict = {
    'threshold' : threshold_transform,
    'clip' : clip_transform,
    'normalize' : ZNormalization(),
    'spatial' : spatial_transform,
    'kspace' : mri_artifact,
    'bias' : bias,
    'blur' : blur,
    'noise' : noise,
    'swap': swap
}

# This function takes in a dataframe, with some other parameters and returns the dataloader
def ImagesFromDataFrame(dataframe, psize, headers, q_max_length, q_samples_per_volume,
                        q_num_workers, q_verbose, train=True, augmentations=None, resize=None):
    # Finding the dimension of the dataframe for computational purposes later
    num_row, num_col = dataframe.shape
    # num_channels = num_col - 1 # for non-segmentation tasks, this might be different
    # changing the column indices to make it easier
    dataframe.columns = range(0,num_col)
    dataframe.index = range(0,num_row)
    # This list will later contain the list of subjects
    subjects_list = []

    channelHeaders = headers['channelHeaders']
    labelHeader = headers['labelHeader']

    # define the control points and swap axes for augmentation
    augmentation_patchAxesPoints = copy.deepcopy(psize)
    for i in range(len(augmentation_patchAxesPoints)):
        augmentation_patchAxesPoints[i] = max(round(augmentation_patchAxesPoints[i] / 10), 1) # always at least have 1

    # iterating through the dataframe
    for patient in range(num_row):
        # We need this dict for storing the meta data for each subject
        # such as different image modalities, labels, any other data
        subject_dict = {}

        # iterating through the channels/modalities/timepoints of the subject
        for channel in channelHeaders:
            # assigning the dict key to the channel
            subject_dict[str(channel)] = Image(str(dataframe[channel][patient]),
                                               type=torchio.INTENSITY)

            if resize is not None:
                image_resized = resize_image(subject_dict[str(channel)].as_sitk(), resize)
                image_masked_tensor = torch.from_numpy(np.swapaxes(sitk.GetArrayFromImage(image_resized), 0, 2))
                # overwrite previous image data with new masked data
                subject_dict[str(channel)] = Image(tensor=image_masked_tensor,
                                                   type=torchio.INTENSITY)

        if labelHeader is not None:
            subject_dict['label'] = Image(str(dataframe[labelHeader][patient]), type=torchio.LABEL)

            if resize is not None:
                image_resized = resize_image(subject_dict['label'].as_sitk(),
                                             resize,
                                             sitk.sitkNearestNeighbor)
                image_masked_tensor = torch.from_numpy(np.swapaxes(sitk.GetArrayFromImage(image_resized), 0, 2))
                # overwrite previous image data with new masked data
                subject_dict['label'] = Image(tensor = image_masked_tensor, type=torchio.INTENSITY)

            if not train:
                subject_dict['path_to_metadata'] = str(dataframe[labelHeader][patient])
        else:
            subject_dict['label'] = "NA"
            if not train:
                subject_dict['path_to_metadata'] = str(dataframe[channel][patient])
        # Initializing the subject object using the dict
        subject = Subject(subject_dict)
        # Appending this subject to the list of subjects
        subjects_list.append(subject)

    augmentation_list = []

    # first, we want to do thresholding, followed by clipping, if it is present - required for inference as well
    for key in ['threshold','clip']:
        augmentation_list.append(global_augs_dict[key](min=augmentations[key]['min'], max=augmentations[key]['max']))
        
    # first, we want to do the resampling, if it is present - required for inference as well
    if 'resample' in augmentations:
        if 'resolution' in augmentations['resample']:
            # resample_split = str(aug).split(':')
            resample_values = tuple(np.array(augmentations['resample']['resolution']).astype(np.float))
            augmentation_list.append(Resample(resample_values))

    # next, we want to do the intensity normalize - required for inference as well
    if 'normalize' in augmentations:
        augmentation_list.append(global_augs_dict['normalize'])

    # other augmentations should only happen for training - and also setting the probabilities
    # for the augmentations
    if train:
        for aug in augmentations:
            # resample and normalize should always have probability=1
            if (aug != 'normalize') and (aug != 'resample'):
                actual_function = global_augs_dict[aug](patch_size=augmentation_patchAxesPoints, p=augmentations[aug]['probability'])
                augmentation_list.append(actual_function)

    transform = Compose(augmentation_list)

    subjects_dataset = torchio.SubjectsDataset(subjects_list, transform=transform)

    if not train:
        return subjects_dataset

    sampler = torchio.data.UniformSampler(psize)
    # all of these need to be read from model.yaml
    patches_queue = torchio.Queue(subjects_dataset, max_length=q_max_length,
                                  samples_per_volume=q_samples_per_volume,
                                  sampler=sampler, num_workers=q_num_workers,
                                  shuffle_subjects=False, shuffle_patches=True, verbose=q_verbose)
    return patches_queue

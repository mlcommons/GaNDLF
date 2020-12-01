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
from GANDLF.preprocessing import *

import copy

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

def bias(patch_size = None, p=1):
    return RandomBiasField(coefficients=0.5, order=3, p=p, seed=None)

def blur(patch_size = None, p=1):
    return RandomBlur(std=(0., 4.), p=p, seed=None)

def noise(patch_size = None, p=1):
    return RandomNoise(mean=0, std=(0, 0.25), p=p, seed=None)

def swap(patch_size = 15, p=1):
    return RandomSwap(patch_size=patch_size, num_iterations=100, p=p, seed=None)

## lambdas for pre-processing
def threshold_transform(min, max, p=1):
    return Lambda(lambda x: threshold_intensities(x, min, max))

def clip_transform(min, max, p=1):
    return Lambda(lambda x: clip_intensities(x, min, max))

# defining dict for pre-processing - key is the string and the value is the transform object
global_preprocessing_dict = {
    'threshold' : threshold_transform,
    'clip' : clip_transform,
    'normalize' : ZNormalization()
}

# Defining a dictionary for augmentations - key is the string and the value is the augmentation object
global_augs_dict = {
    'spatial' : spatial_transform,
    'kspace' : mri_artifact,
    'bias' : bias,
    'blur' : blur,
    'noise' : noise,
    'swap': swap
}

global_sampler_dict = {
    'uniform': torchio.data.UniformSampler,
    'uniformsampler': torchio.data.UniformSampler,
    'uniformsample': torchio.data.UniformSampler,
    'label': torchio.data.LabelSampler,
    'labelsampler': torchio.data.LabelSampler,
    'labelsample': torchio.data.LabelSampler,
}

# This function takes in a dataframe, with some other parameters and returns the dataloader
def ImagesFromDataFrame(dataframe, psize, headers, q_max_length, q_samples_per_volume,
                        q_num_workers, q_verbose, sampler = 'label', train=True, augmentations=None, preprocessing=None):
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
    predictionHeaders = headers['predictionHeaders']

    # define the control points and swap axes for augmentation
    augmentation_patchAxesPoints = copy.deepcopy(psize)
    for i in range(len(augmentation_patchAxesPoints)):
        augmentation_patchAxesPoints[i] = max(round(augmentation_patchAxesPoints[i] / 10), 1) # always at least have 1

    # iterating through the dataframe
    resizeCheck = False
    for patient in range(num_row):
        # We need this dict for storing the meta data for each subject
        # such as different image modalities, labels, any other data
        subject_dict = {}

        # iterating through the channels/modalities/timepoints of the subject
        for channel in channelHeaders:
            # assigning the dict key to the channel
            subject_dict[str(channel)] = Image(str(dataframe[channel][patient]),
                                               type=torchio.INTENSITY)

            # if resize has been defined but resample is not (or is none)
            if not resizeCheck:
                if not(preprocessing is None) and ('resize' in preprocessing):
                    if (preprocessing['resize'] is not None):
                        resizeCheck = True
                        if not('resample' in preprocessing):
                            preprocessing['resample'] = {}

                            if not('resolution' in preprocessing['resample']):
                                preprocessing['resample']['resolution'] = resize_image_resolution(subject_dict[str(channel)].as_sitk(), preprocessing['resize'])
                        else:
                            print('WARNING: \'resize\' is ignored as \'resample\' is defined under \'data_processing\', this will be skipped', file = sys.stderr)
                else:
                    resizeCheck = True
        
        # # for regression
        # if predictionHeaders:
        #     # get the mask
        #     if (subject_dict['label'] is None) and (class_list is not None):
        #         sys.exit('The \'class_list\' parameter has been defined but a label file is not present for patient: ', patient)

        if labelHeader is not None:
            subject_dict['label'] = Image(str(dataframe[labelHeader][patient]), type=torchio.LABEL)

            if not train:
                subject_dict['path_to_metadata'] = str(dataframe[labelHeader][patient])
        else:
            subject_dict['label'] = "NA"
            if not train:
                subject_dict['path_to_metadata'] = str(dataframe[channel][patient])
        
        # iterating through the values to predict of the subject
        valueCounter = 0
        for values in predictionHeaders:
            # assigning the dict key to the channel
            subject_dict['value_' + str(valueCounter)] = np.array(dataframe[values][patient])
            valueCounter = valueCounter + 1
        
        # Initializing the subject object using the dict
        subject = Subject(subject_dict)
        # Appending this subject to the list of subjects
        subjects_list.append(subject)

    augmentation_list = []

    # first, we want to do thresholding, followed by clipping, if it is present - required for inference as well
    if not(preprocessing is None):
        for key in ['threshold','clip']:
            if key in preprocessing:
                augmentation_list.append(global_preprocessing_dict[key](min=preprocessing[key]['min'], max=preprocessing[key]['max']))
        
        # first, we want to do the resampling, if it is present - required for inference as well
        if 'resample' in preprocessing:
            if 'resolution' in preprocessing['resample']:
                # resample_split = str(aug).split(':')
                resample_values = tuple(np.array(preprocessing['resample']['resolution']).astype(np.float))
                if len(resample_values) == 2:
                    resample_values = tuple(np.append(resample_values,1))
                augmentation_list.append(Resample(resample_values))

        # next, we want to do the intensity normalize - required for inference as well
        if 'normalize' in preprocessing:
            augmentation_list.append(global_preprocessing_dict['normalize'])

    # other augmentations should only happen for training - and also setting the probabilities
    # for the augmentations
    if train and not(augmentations == None):
        for aug in augmentations:
            actual_function = global_augs_dict[aug](patch_size=augmentation_patchAxesPoints, p=augmentations[aug]['probability'])
            augmentation_list.append(actual_function)

    transform = Compose(augmentation_list)

    subjects_dataset = torchio.SubjectsDataset(subjects_list, transform=transform)

    if not train:
        return subjects_dataset

    sampler = sampler.lower() # for easier parsing
    sampler = global_sampler_dict[sampler](psize)
    # all of these need to be read from model.yaml
    patches_queue = torchio.Queue(subjects_dataset, max_length=q_max_length,
                                  samples_per_volume=q_samples_per_volume,
                                  sampler=sampler, num_workers=q_num_workers,
                                  shuffle_subjects=False, shuffle_patches=True, verbose=q_verbose)
    return patches_queue

import torch
from torch.utils.data.dataset import Dataset
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import os
import random
import random
import scipy
import torchio
from torchio.transforms import *
from torchio import Image, Subject

# Defining a dictionary - key is the string and the value is the augmentation object
## todo: ability to change interpolation type from config file
global_augs_dict = {
    'normalize':ZNormalization(),
    'affine':RandomAffine(image_interpolation = 'linear'), 
    'elastic': RandomElasticDeformation(num_control_points=(7, 7, 7),locked_borders=2),
    'motion': RandomMotion(degrees=10, translation = 10, num_transforms= 2, image_interpolation = 'linear', p = 1., seed = None), 
    'ghosting': RandomGhosting(num_ghosts = (4, 10), axes = (0, 1, 2), intensity = (0.5, 1), restore = 0.02, p = 1., seed = None),
    'bias': RandomBiasField(coefficients = 0.5, order= 3, p= 1., seed = None), 
    'blur': RandomBlur(std = (0., 4.), p = 1, seed = None), 
    'noise':RandomNoise(mean = 0, std = (0, 0.25), p = 1., seed = None) , 
    'swap':RandomSwap(patch_size = 15, num_iterations = 100, p = 1, seed = None) 
}

# This function takes in a dataframe, with some other parameters and returns the dataloader
def ImagesFromDataFrame(dataframe, psize, channelHeaders, labelHeader, train = True, augmentations = None):
    # Finding the dimension of the dataframe for computational purposes later
    num_row, num_col = dataframe.shape
    # num_channels = num_col - 1 # for non-segmentation tasks, this might be different

    # changing the column indices to make it easier
    dataframe.columns = range(0,num_col)
    dataframe.index = range(0,num_row)
    # This list will later contain the list of subjects 
    subjects_list = []

    # iterating through the dataframe
    for patient in range(num_row):
        # We need this dict for storing the meta data for each subject such as different image modalities, labels, any other data
        subject_dict = {}
        # iterating through the channels/modalities/timepoints of the subject
        for channel in channelHeaders:
            # assigining the dict key to the channel
            subject_dict[str(channel)] = Image(str(dataframe[channel][patient]),type = torchio.INTENSITY)
        subject_dict['label'] = Image(str(dataframe[labelHeader][patient]),type = torchio.LABEL)
        # Initializing the subject object using the dict
        subject = Subject(subject_dict) 
        # Appending this subject to the list of subjects
        subjects_list.append(subject)
    
    augmentation_list = []
    
    # first, we want to do the resampling, if it is present
    for aug in augmentations:
        if 'resample' in str(aug):
            resample_split = str(aug).split(':')
            resample_values = tuple(np.array(resample_split[1].split(',')).astype(np.float))
            augmentation_list.append(Resample(resample_values))
    
    # next, we want to do the intensity normalize
    if 'normalize' in augmentations:
        augmentation_list.append(global_augs_dict['normalize'])
    
    # other augmentations should only happen for training
    if train:
        for aug in augmentations:
            if (str(aug) != 'normalize') and not('resample' in str(aug)):
                augmentation_list.append(global_augs_dict[str(aug)])
        
    transform = Compose(augmentation_list)

    
    # If mode is not training (i.e if mode is inference then do not perform any augmentations)
    if train:
        subjects_dataset = torchio.ImagesDataset(subjects_list, transform=transform)
    if not train:
        subjects_dataset = torchio.ImagesDataset(subjects_list, transform=None)
        return subjects_dataset

    sampler = torchio.data.UniformSampler(psize) 
    # all of these need to be read from model.cfg
    patches_queue = torchio.Queue(subjects_dataset,max_length = 1,samples_per_volume  = 1,sampler = sampler,num_workers=4,shuffle_subjects=False, shuffle_patches=True) 
    
    return patches_queue


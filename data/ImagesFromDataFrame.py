import nibabel as nib
import torch
from torch.utils.data.dataset import Dataset
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import os
import random
from all_augmentations import *
from utils import *
import random
import scipy
import torchio
from torchio.transforms import RandomAffine, RandomElasticDeformation, Compose
from torchio import Image, Subject

# Defining a dictionary - key is the string and the value is the augmentation object
global_augs_dict = {}



# This function takes in a dataframe, with some other parameters and returns the dataloader
def ImagesFromDataFrame(dataframe, psize, augmentations = None):
    # Finding the dimension of the dataframe for computational purposes later
    num_row, num_col = dataframe.shape
    num_channels = num_col - 1 # for non-segmentation tasks, this might be different
    # changing the column indices to make it easier
    dataframe.columns = range(0,num_col)
    dataframe.index = range(0,num_row)
    # This list will later contain the list of subjects 
    subjects_list = []

    # iterating through the dataframe
    for patient in range(num_row):
        # We need this dict for storing the meta data for eac subject such as different image modalities, labels, any other data
        subject_dict = {}
        # iterating through the channels/modalities/timepoint of the image
        for channel in range(num_channels):
            channel_path = str(dataframe[channel][patient])
            # assigining the dict key to the channel
            subject_dict[channel_path] = Image(channel_path,type = torchio.INTENSITY)
        subject_dict['mask'] = Image(str(dataframe[num_channels][patient]),type = torchio.LABEL)
        # Initializing the subject object using the dict
        subject = Subject(subject_dict) 
        # Appending this subject to the list of subjects
        subjects_list.append(subject)
    
    return subjects_list


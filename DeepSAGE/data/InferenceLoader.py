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


# This function takes in a dataframe, with some other parameters and returns the dataloader
def InferenceLoader(dataframe, psize, channelHeaders, labelHeader):
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
    
    subjects_dataset = torchio.ImagesDataset(subjects_list)
    # Using the grid sampler for inference since somtetimes the entire image can't fit in the GPU
    print(subjects_dataset)
    grid_sampler = torchio.inference.GridSampler(subjects_dataset, psize, patch_overlap = 4)
    
    return grid_sampler


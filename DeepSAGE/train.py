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
from torchio.transforms import *
from torchio import Image, Subject
from sklearn.model_selection import KFold
from data.ImagesFromDataFrame import ImagesFromDataFrame

# Defining a dictionary - key is the string and the value is the augmentation object
## todo: ability to change interpolation type from config file
global_augs_dict = {
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
def Trainer(dataframe, parameters):

  kfolds = int(parameters['kcross_validation'])
  # check for single fold training
  singleFoldTraining = False
  if kfolds < 0: # if the user wants a single fold training
      kfolds = abs(kfolds)
      singleFoldTraining = True

  kf = KFold(n_splits=kfolds) # initialize the kfold structure

  currentFold = 0

  # write parameters to pickle - this should not change for the different folds, so keeping is independent
  paramtersPickle = os.path.join(model_path,'params.pkl')
  with open(paramtersPickle, 'wb') as handle:
      pickle.dump(params, handle, protocol=pickle.HIGHEST_PROTOCOL)

  # get the indeces for kfold splitting
  trainingData_full = dataframe
  training_indeces_full = list(trainingData_full.index.values)

  # start the kFold train
  for train_index, test_index in kf.split(training_indeces_full):

      # the output of the current fold is only needed if multi-fold training is happening
      if singleFoldTraining:
          currentOutputFolder = model_path
      else:
          currentOutputFolder = os.path.join(model_path, str(currentFold))
          Path(currentOutputFolder).mkdir(parents=True, exist_ok=True)

      trainingData = trainingData_full.iloc[train_index]
      validationData = trainingData_full.iloc[test_index]

      # save the current model configuration as a sanity check
      copyfile(model_parameters, os.path.join(currentOutputFolder,'model.cfg'))

  test = 1

import os
import ast 
import sys
import numpy as np
import yaml

def parseConfig(config_file_path):
  '''
  This function parses the configuration file and returns a dictionary of parameters
  '''
  with open(config_file_path) as f:
    params = yaml.load(f, Loader=yaml.FullLoader)
  
  # require parameters - this should error out if not present
  if not('class_list' in params):
    sys.exit('The \'class_list\' parameter needs to be present in the configuration file')

  if 'patch_size' in params:
    params['psize'] = params['patch_size'] 
  else:
    sys.exit('The \'patch_size\' parameter needs to be present in the configuration file')

  # Extrating the training parameters from the dictionary
  if 'num_epochs' in params:
    num_epochs = int(params['num_epochs'])
  else:
    num_epochs = 100
    print('Using default num_epochs: ', num_epochs)
  params['num_epochs'] = num_epochs

  if 'batch_size' in params:
    batch_size = int(params['batch_size'])
  else:
    batch_size = 1
    print('Using default batch_size: ', batch_size)
  params['batch_size'] = batch_size

  if 'learning_rate' in params:
    learning_rate = float(params['learning_rate'])
  else:
    learning_rate = 0.001
    print('Using default learning_rate: ', learning_rate)
  params['learning_rate'] = learning_rate

  if 'loss_function' in params:
    defineDefaultLoss = False
    # check if user has passed a dict 
    if isinstance(params['loss_function'], dict): # if this is a dict
      if len(params['loss_function']) > 0: # only proceed if something is defined
        for key in params['loss_function']: # iterate through all keys
          if key == 'mse_torch':
            if (params['loss_function'][key] == None) or not('reduction' in params['loss_function'][key]):
              params['loss_function'][key] = {}
              params['loss_function'][key]['reduction'] = 'mean'
          else:
            params['loss_function'] = key # use simple string for other functions - can be extended with parameters, if needed
      else:
        defineDefaultLoss = True
    else:      
      # check if user has passed a single string
      if params['loss_function'] == 'mse_torch':
        params['loss_function'] = {}
        params['loss_function']['mse_torch'] = {}
        params['loss_function']['mse_torch']['reduction'] = 'mean'
  else:
    defineDefaultLoss = True
  if defineDefaultLoss == True:
    loss_function = 'dc'
    print('Using default loss_function: ', loss_function)
  else:
    loss_function = params['loss_function']
  params['loss_function'] = loss_function

  if 'opt' in params:
    opt = str(params['opt'])
  else:
    opt = 'adam'
    print('Using default opt: ', opt)
  params['opt'] = opt
  
  # this is NOT a required parameter - a user should be able to train with NO augmentations
  if len(params['data_augmentation']) > 0: # only when augmentations are defined
    for key in params['data_augmentation']: # iterate through all keys
      if (key != 'normalize') and (key != 'resample'): # no need to check probabilities for these: they should ALWAYS be added
        if (params['data_augmentation'][key] == None) or not('probability' in params['data_augmentation'][key]): # when probability is not present for an augmentation, default to '1'
            params['data_augmentation'][key] = {}
            params['data_augmentation'][key]['probability'] = 1

  # Extracting the model parameters from the dictionary
  if 'base_filters' in params:
    base_filters = int(params['base_filters'])
  else:
    base_filters = 30
    print('Using default base_filters: ', base_filters)
  params['base_filters'] = base_filters

  if 'which_model' in params:
    which_model = str(params['modelName'])
  else:
    which_model = 'resunet'
    print('Using default which_model: ', which_model)
  params['which_model'] = which_model

  if 'kcross_validation' in params:
    kfolds = int(params['kcross_validation'])
  else:
    kfolds = -10
    print('Using default kcross_validation: ', kfolds)
  params['kfolds'] = kfolds

  # Setting default values to the params
  if 'scheduler' in params:
      scheduler = str(params['scheduler'])
  else:
      scheduler = 'triangle'
  params['scheduler'] = scheduler

  if 'q_max_length' in params:
      q_max_length = int(params['q_max_length'])
  else:
      q_max_length = 100
  params['q_max_length'] = q_max_length

  if 'q_samples_per_volume' in params:
      q_samples_per_volume = int(params['q_samples_per_volume'])
  else:
      q_samples_per_volume = 10
  params['q_samples_per_volume'] = q_samples_per_volume

  if 'q_num_workers' in params:
      q_num_workers = int(params['q_num_workers'])
  else:
      q_num_workers = 4
  params['q_num_workers'] = q_num_workers

  q_verbose = False
  if 'q_verbose' in params:
      if params['q_verbose'] == 'True':
          q_verbose = True  
  params['q_verbose'] = q_verbose

  parallel_compute_command = ''
  if 'parallel_compute_command' in params:
      parallel_compute_command = params['parallel_compute_command']
      parallel_compute_command = parallel_compute_command.replace('\'', '')
      parallel_compute_command = parallel_compute_command.replace('\"', '')
      params['parallel_compute_command'] = parallel_compute_command

  return params
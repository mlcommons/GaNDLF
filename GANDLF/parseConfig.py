
import os
import ast 
import sys
import numpy as np
import yaml
import pkg_resources

def parse_version(version_string):
  '''
  Parses version string, discards last identifier (NR/alpha/beta) and returns an integer for comparison
  '''
  version_string_split = version_string.split('.')
  if len(version_string_split) > 3:
    del version_string_split[-1]
  return int(''.join(version_string_split))

def initialize_key(parameters, key):
  '''
  This function will initialize the key in the parameters dict to 'None' if it is absent or length is zero
  '''
  if key in parameters: 
    if len(parameters[key]) == 0: # if key is present but not defined
      parameters[key] = None
  else:
    parameters[key] = None # if key is absent

  return parameters

def parseConfig(config_file_path, version_check = True):
  '''
  This function parses the configuration file and returns a dictionary of parameters
  '''
  with open(config_file_path) as f:
    params = yaml.load(f, Loader=yaml.FullLoader)
  
  if version_check: # this is only to be used for testing
    if not('version' in params):
      sys.exit('The \'version\' key needs to be defined in config with \'minimum\' and \'maximum\' fields to determine the compatibility of configuration with code base')
    else:
      gandlf_version = pkg_resources.require('GANDLF')[0].version
      gandlf_version_int = parse_version(gandlf_version)
      min = parse_version(params['version']['minimum'])
      max = parse_version(params['version']['maximum'])
      if (min > gandlf_version_int) or (max < gandlf_version_int):
        sys.exit('Incompatible version of GANDLF detected (' + gandlf_version + ')')
      
  if 'patch_size' in params:
    params['psize'] = params['patch_size'] 
  else:
    sys.exit('The \'patch_size\' parameter needs to be present in the configuration file')
  
  if not('patch_sampler' in params):
    params['patch_sampler'] = 'label'

  if 'resize' in params:
    print('WARNING: \'resize\' should be defined under \'data_processing\', this will be skipped', file = sys.stderr)

  # Extrating the training parameters from the dictionary
  if 'num_epochs' in params:
    num_epochs = int(params['num_epochs'])
  else:
    num_epochs = 100
    print('Using default num_epochs: ', num_epochs)
  params['num_epochs'] = num_epochs
  
  if 'patience' in params:
    patience = int(params['patience'])
  else:
    print("Patience not given, train for full number of epochs")
    patience = num_epochs
  params['patience'] = patience

  if 'batch_size' in params:
    batch_size = int(params['batch_size'])
  else:
    batch_size = 1
    print('Using default batch_size: ', batch_size)
  params['batch_size'] = batch_size
  
  if 'amp' in params:
    amp = bool(params['amp'])
  else:
    amp = False
    print("NOT using Mixed Precision Training")
  params['amp'] = amp

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
          if key == 'mse':
            if (params['loss_function'][key] == None) or not('reduction' in params['loss_function'][key]):
              params['loss_function'][key] = {}
              params['loss_function'][key]['reduction'] = 'mean'
          else:
            params['loss_function'] = key # use simple string for other functions - can be extended with parameters, if needed
      else:
        defineDefaultLoss = True
    else:      
      # check if user has passed a single string
      if params['loss_function'] == 'mse':
        params['loss_function'] = {}
        params['loss_function']['mse'] = {}
        params['loss_function']['mse']['reduction'] = 'mean'
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
  params = initialize_key(params, 'data_augmentation')
  if not(params['data_augmentation'] == None):
    if len(params['data_augmentation']) > 0: # only when augmentations are defined
      
      if 'spatial' in params['data_augmentation']:
          if not('affine' in params['data_augmentation']) or not('elastic' in params['data_augmentation']):
              print('WARNING: \'spatial\' is now deprecated in favor of split \'affine\' and/or \'elastic\'', file = sys.stderr)
              params['data_augmentation']['affine'] = {}
              params['data_augmentation']['elastic'] = {}
              del params['data_augmentation']['spatial']

      if 'swap' in params['data_augmentation']:
          if not(isinstance(params['data_augmentation']['swap'], dict)):
              params['data_augmentation']['swap'] = {}
          if not('patch_size' in params['data_augmentation']['swap']):
              params['data_augmentation']['swap']['patch_size'] = 15 # default
      
      for key in params['data_augmentation']:
          if (params['data_augmentation'][key] == None) or not('probability' in params['data_augmentation'][key]): # when probability is not present for an augmentation, default to '1'
              if not isinstance(params['data_augmentation'][key], dict):
                params['data_augmentation'][key] = {}
              params['data_augmentation'][key]['probability'] = 1

  # this is NOT a required parameter - a user should be able to train with NO built-in pre-processing 
  params = initialize_key(params, 'data_preprocessing')
  if not(params['data_preprocessing'] == None):
    if len(params['data_preprocessing']) < 0: # perform this only when pre-processing is defined
      thresholdOrClip = False
      thresholdOrClipDict = ['threshold', 'clip'] # this can be extended, as required
      keysForWarning = ['resize'] # properties for which the user will see a warning

      # iterate through all keys
      for key in params['data_preprocessing']: # iterate through all keys
        # for threshold or clip, ensure min and max are defined
        if not thresholdOrClip:
          if (key in thresholdOrClipDict):
            thresholdOrClip = True # we only allow one of threshold or clip to occur and not both
            if not(isinstance(params['data_preprocessing'][key], dict)): # initialize if nothing is present
              params['data_preprocessing'][key] = {}
            
            # if one of the required parameters is not present, initialize with lowest/highest possible values
            # this ensures the absence of a field doesn't affect processing
            if not 'min' in params['data_preprocessing'][key]: 
              params['data_preprocessing'][key]['min'] = sys.float_info.min
            if not 'max' in params['data_preprocessing'][key]:
              params['data_preprocessing'][key]['max'] = sys.float_info.max
        else:
          sys.exit('Use only \'threshold\' or \'clip\', not both')

        # give a warning for resize
        if key in keysForWarning:
          print('WARNING: \'' + key + '\' is generally not recommended, as it changes image properties in unexpected ways.', file = sys.stderr)

  if 'modelName' in params:
    defineDefaultModel = False
    print('This option has been superceded by \'model\'', file=sys.stderr)
    which_model = str(params['modelName'])
  elif 'which_model' in params:
    defineDefaultModel = False
    print('This option has been superceded by \'model\'', file=sys.stderr)
    which_model = str(params['which_model'])
  else: # default case
    defineDefaultModel = True
  if defineDefaultModel == True:
    which_model = 'resunet'
    # print('Using default model: ', which_model)
  params['which_model'] = which_model

  if 'model' in params:

    if not(isinstance(params['model'], dict)):
      sys.exit('The \'model\' parameter needs to be populated as a dictionary')
    elif len(params['model']) == 0: # only proceed if something is defined
      sys.exit('The \'model\' parameter needs to be populated as a dictionary and should have all properties present')

    if not('architecture' in params['model']):
      sys.exit('The \'model\' parameter needs \'architecture\' key to be defined')
    if not('final_layer' in params['model']):
      sys.exit('The \'model\' parameter needs \'final_layer\' key to be defined')
    if not('dimension' in params['model']):
      sys.exit('The \'model\' parameter needs \'dimension\' key to be defined, which should either 2 or 3')
    if not('base_filters' in params['model']):
      base_filters = 32
      params['model']['base_filters'] = base_filters
      print('Using default \'base_filters\' in \'model\': ', base_filters)
      # sys.exit('The \'model\' parameter needs \'base_filters\' key to be defined') # uncomment if we need this to be passed by user
    # if not('n_channels' in params['model']):
    #   n_channels = 32
    #   params['model']['n_channels'] = n_channels
    #   print('Using default \'n_channels\' in \'model\': ', n_channels)
    #   # sys.exit('The \'model\' parameter needs \'n_channels\' key to be defined') # uncomment if we need this to be passed by user

  else:
    sys.exit('The \'model\' parameter needs to be populated as a dictionary')

  if 'kcross_validation' in params:
    sys.exit('\'kcross_validation\' is no longer used, please use \'nested_training\' instead')

  if not('nested_training' in params):
    sys.exit('The parameter \'nested_training\' needs to be defined')
  if not('testing' in params['nested_training']):
    if not('holdout' in params['nested_training']):
      kfolds = -5
      print('Using default folds for testing split: ', kfolds)
    else:
      print('WARNING: \'holdout\' should not be defined under \'nested_training\', please use \'testing\' instead;', file = sys.stderr)
      kfolds = params['nested_training']['holdout']
    params['nested_training']['testing'] = kfolds
  if not('validation' in params['nested_training']):
    kfolds = -5
    print('Using default folds for validation split: ', kfolds)
    params['nested_training']['validation'] = kfolds

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

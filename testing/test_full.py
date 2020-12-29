import math
import sys
from pathlib import Path
import requests, zipfile, io, os

from GANDLF.utils import *
from GANDLF.parseConfig import parseConfig
from GANDLF.training_manager import TrainingManager
from GANDLF.inference_manager import InferenceManager 

## global defines
# all_models_segmentation = ['unet', 'resunet', 'fcn', 'uinc'] # pre-defined segmentation model types for testing
all_models_segmentation = ['unet', 'fcn', 'uinc'] # pre-defined segmentation model types for testing
# all_models_regression = ['densenet121', 'densenet161', 'densenet169', 'densenet201', 'vgg16'] # populate once it becomes available

inputDir = os.path.abspath(os.path.normpath('./testing/data'))
outputDir = os.path.abspath(os.path.normpath('./testing/data_output'))
Path(outputDir).mkdir(parents=True, exist_ok=True)

'''
steps to follow to write tests:
[x] download sample data
[x] construct the training csv
[x] for each dir (application type) and sub-dir (image dimension), run training for a single epoch on cpu
  [x] separate tests for 2D and 3D segmentation
  [x] read default parameters from yaml config
  [x] for each type, iterate through all available segmentation model archs
  [x] call training manager with default parameters + current segmentation model arch
[ ] for each dir (application type) and sub-dir (image dimension), run inference for a single trained model per testing/validation split for a single subject on cpu
4. hopefully the various sys.exit messages throughout the code will catch issues
'''

def test_download_data():
  '''
  This function downloads the sample data, which is the first step towards getting everything ready
  '''
  urlToDownload = 'https://github.com/sarthakpati/tempDownloads/raw/main/data.zip'
  # do not download data again
  if not Path(os.getcwd() + '/testing/data/test/3d_rad_segmentation/001/image.nii.gz').exists():
    print('Downloading and extracting sample data')
    r = requests.get(urlToDownload)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall('./testing')

def test_constructTrainingCSV():
  '''
  This function constructs training csv
  '''
  # inputDir = os.path.normpath('./testing/data')
  # delete previous csv files
  files = os.listdir(inputDir)
  for item in files:
    if item.endswith(".csv"):
      os.remove(os.path.join(inputDir, item))

  for application_data in os.listdir(inputDir):
    currentApplicationDir = os.path.join(inputDir, application_data)

    if '2d_rad_segmentation' in application_data:
      channelsID = '_blue.png,_red.png,_green.png'
      labelID = 'mask.png'
    elif '3d_rad_segmentation' in application_data:
      channelsID = 'image'
      labelID = 'mask'
    writeTrainingCSV(currentApplicationDir, channelsID, labelID, inputDir + '/train_' + application_data + '.csv')


def test_train_segmentation_rad_2d():
  print('Starting 2D Rad segmentation tests')
  application_data = '2d_rad_segmentation'
  parameters = parseConfig(inputDir + '/' + application_data + '/sample_training.yaml', version_check = False)
  training_data, headers = parseTrainingCSV(inputDir + '/train_' + application_data + '.csv')
  for model in all_models_segmentation:
    parameters['model']['architecture'] = model 
    currentOutputDir = os.path.join(outputDir, application_data + '_' + model)
    Path(currentOutputDir).mkdir(parents=True, exist_ok=True)
    TrainingManager(dataframe=training_data, headers = headers, outputDir=currentOutputDir, parameters=parameters, device='cpu')

  print('passed')

def test_train_segmentation_rad_3d():
  print('Starting 3D Rad segmentation tests')
  application_data = '3d_rad_segmentation'
  parameters = parseConfig(inputDir + '/' + application_data + '/sample_training.yaml', version_check = False)
  training_data, headers = parseTrainingCSV(inputDir + '/train_' + application_data + '.csv')
  for model in all_models_segmentation:
    parameters['model']['architecture'] = model 
    currentOutputDir = os.path.join(outputDir, application_data + '_' + model)
    Path(currentOutputDir).mkdir(parents=True, exist_ok=True)
    TrainingManager(dataframe=training_data, headers = headers, outputDir=currentOutputDir, parameters=parameters, device='cpu')

  print('passed')

# def test_regression_rad_2d():
#   application_data = '2d_rad_segmentation'
#   parameters = parseConfig(inputDir + '/' + application_data + '/sample_training.yaml')
#   # training_data, headers = parseTrainingCSV(inputDir + '/train_' + application_data + '.csv')
#   for model in all_models_regression:
#     parameters['model']['architecture'] = model 
#     # currentOutputDir = os.path.join(outputDir, application_data + '_' + model)
#     # TrainingManager(dataframe=training_data, headers = headers, outputDir=currentOutputDir, parameters=parameters, device='cpu')

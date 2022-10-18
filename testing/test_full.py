from pathlib import Path
import requests, zipfile, io, os, csv, random, copy, shutil, sys, yaml, torch, pytest
import SimpleITK as sitk
import numpy as np
import pandas as pd

from pydicom.data import get_testdata_file
import cv2

from GANDLF.data.ImagesFromDataFrame import ImagesFromDataFrame
from GANDLF.utils import *
from GANDLF.data.preprocessing import global_preprocessing_dict
from GANDLF.data.augmentation import global_augs_dict
from GANDLF.parseConfig import parseConfig
from GANDLF.training_manager import TrainingManager
from GANDLF.inference_manager import InferenceManager
from GANDLF.cli import main_run, preprocess_and_save, patch_extraction
from GANDLF.schedulers import global_schedulers_dict
from GANDLF.optimizers import global_optimizer_dict
from GANDLF.models import global_models_dict
from GANDLF.data.post_process import (
    torch_morphological,
    fill_holes,
    get_mapped_label,
    cca,
)
from GANDLF.anonymize import run_anonymizer

device = "cpu"
## global defines
# pre-defined segmentation model types for testing
all_models_segmentation = [
    "lightunet",
    "lightunet_multilayer",
    "unet",
    "unet_multilayer",
    "deep_resunet",
    "fcn",
    "uinc",
    "msdnet",
    "imagenet_unet",
]
# pre-defined regression/classification model types for testing
all_models_regression = [
    "densenet121",
    "vgg16",
    "resnet18",
    "resnet50",
    "efficientnetb0",
    "imagenet_unet",
]
# pre-defined regression/classification model types for testing
all_models_classification = [
    "imagenet_vgg11",
    "imagenet_vgg11_bn",
    "imagenet_vgg13",
    "imagenet_vgg13_bn",
    "imagenet_vgg16",
    "imagenet_vgg16_bn",
    "imagenet_vgg19",
    "imagenet_vgg19_bn",
    "resnet18",
]

all_clip_modes = ["norm", "value", "agc"]
all_norm_type = ["batch", "instance"]

all_model_type = ["torch", "openvino"]

patch_size = {"2D": [128, 128, 1], "3D": [32, 32, 32]}

testingDir = Path(__file__).parent.absolute().__str__()
baseConfigDir = os.path.join(testingDir, os.pardir, "samples")
inputDir = os.path.join(testingDir, "data")
outputDir = os.path.join(testingDir, "data_output")
Path(outputDir).mkdir(parents=True, exist_ok=True)


"""
steps to follow to write tests:
[x] download sample data
[x] construct the training csv
[x] for each dir (application type) and sub-dir (image dimension), run training for a single epoch on cpu
  [x] separate tests for 2D and 3D segmentation
  [x] read default parameters from yaml config
  [x] for each type, iterate through all available segmentation model archs
  [x] call training manager with default parameters + current segmentation model arch
[ ] for each dir (application type) and sub-dir (image dimension), run inference for a single trained model per testing/validation split for a single subject on cpu
"""


def test_generic_download_data():
    print("00: Downloading the sample data")
    urlToDownload = (
        "https://upenn.box.com/shared/static/y8162xkq1zz5555ye3pwadry2m2e39bs.zip"
    )

    files_check = [
        os.path.join(inputDir, "2d_histo_segmentation", "1", "image.tiff"),
        os.path.join(inputDir, "2d_rad_segmentation", "001", "image.png"),
        os.path.join(inputDir, "3d_rad_segmentation", "001", "image.nii.gz"),
    ]
    # check for missing subjects so that we do not download data again
    for file in files_check:
        if not os.path.isfile(file):
            print("Downloading and extracting sample data")
            r = requests.get(urlToDownload)
            z = zipfile.ZipFile(io.BytesIO(r.content))
            z.extractall(testingDir)
            break

    print("passed")


def test_generic_constructTrainingCSV():
    print("01: Constructing training CSVs")
    # inputDir = os.path.normpath('./testing/data')
    # delete previous csv files
    files = os.listdir(inputDir)
    for item in files:
        if item.endswith(".csv"):
            os.remove(os.path.join(inputDir, item))

    for application_data in os.listdir(inputDir):
        currentApplicationDir = os.path.join(inputDir, application_data)

        if "2d_rad_segmentation" in application_data:
            channelsID = "image.png"
            labelID = "mask.png"
        elif "3d_rad_segmentation" in application_data:
            channelsID = "image"
            labelID = "mask"
        elif "2d_histo_segmentation" in application_data:
            channelsID = "image"
            labelID = "mask"
        # else:
        #     continue
        writeTrainingCSV(
            currentApplicationDir,
            channelsID,
            labelID,
            inputDir + "/train_" + application_data + ".csv",
        )

        # write regression and classification files
        application_data_regression = application_data.replace(
            "segmentation", "regression"
        )
        application_data_classification = application_data.replace(
            "segmentation", "classification"
        )
        with open(
            inputDir + "/train_" + application_data + ".csv", "r"
        ) as read_f, open(
            inputDir + "/train_" + application_data_regression + ".csv", "w", newline=""
        ) as write_reg, open(
            inputDir + "/train_" + application_data_classification + ".csv",
            "w",
            newline="",
        ) as write_class:
            csv_reader = csv.reader(read_f)
            csv_writer_1 = csv.writer(write_reg)
            csv_writer_2 = csv.writer(write_class)
            i = 0
            for row in csv_reader:
                if i == 0:
                    row.append("ValueToPredict")
                    csv_writer_2.writerow(row)
                    # row.append('ValueToPredict_2')
                    csv_writer_1.writerow(row)
                else:
                    row_regression = copy.deepcopy(row)
                    row_classification = copy.deepcopy(row)
                    row_regression.append(str(random.uniform(0, 1)))
                    # row_regression.append(str(random.uniform(0, 1)))
                    row_classification.append(str(random.randint(0, 2)))
                    csv_writer_1.writerow(row_regression)
                    csv_writer_2.writerow(row_classification)
                i += 1


# these are helper functions to be used in other tests
def sanitize_outputDir():
    print("02_1: Sanitizing outputDir")
    if os.path.isdir(outputDir):
        shutil.rmtree(outputDir)  # overwrite previous results
    Path(outputDir).mkdir(parents=True, exist_ok=True)


def get_temp_config_path():
    print("02_2: Creating path for temporary config file")
    temp_config_path = os.path.join(outputDir, "config_temp.yaml")
    # if found in previous run, discard.
    if os.path.exists(temp_config_path):
        os.remove(temp_config_path)
    return temp_config_path


# these are helper functions to be used in other tests


def test_train_segmentation_rad_2d(device):
    print("03: Starting 2D Rad segmentation tests")
    # read and parse csv
    parameters = parseConfig(
        testingDir + "/config_segmentation.yaml", version_check_flag=False
    )
    training_data, parameters["headers"] = parseTrainingCSV(
        inputDir + "/train_2d_rad_segmentation.csv"
    )
    parameters["modality"] = "rad"
    parameters["patch_size"] = patch_size["2D"]
    parameters["model"]["dimension"] = 2
    parameters["model"]["class_list"] = [0, 255]
    parameters["model"]["amp"] = True
    parameters["model"]["num_channels"] = 3
    parameters["model"]["onnx_export"] = False
    parameters["model"]["print_summary"] = False
    parameters["data_preprocessing"]["resize_image"] = [224, 224]
    parameters = populate_header_in_parameters(parameters, parameters["headers"])
    # read and initialize parameters for specific data dimension
    for model in all_models_segmentation:
        if model == "imagenet_unet":
            # imagenet_unet encoder needs to be toned down for small patch size
            parameters["model"]["encoder_depth"] = 3
            parameters["model"]["decoder_channels"] = (64, 32, 16)
            parameters["model"]["final_layer"] = random.choice(
                ["sigmoid", "softmax", "logsoftmax", "tanh", "identity"]
            )
            parameters["model"]["converter_type"] = random.choice(
                ["acs", "soft", "conv3d"]
            )
        parameters["model"]["architecture"] = model
        parameters["nested_training"]["testing"] = -5
        parameters["nested_training"]["validation"] = -5
        sanitize_outputDir()
        TrainingManager(
            dataframe=training_data,
            outputDir=outputDir,
            parameters=parameters,
            device=device,
            resume=False,
            reset=True,
        )

    print("passed")


def test_train_segmentation_sdnet_rad_2d(device):
    print("04: Starting 2D Rad segmentation tests")
    # read and parse csv
    parameters = parseConfig(
        testingDir + "/config_segmentation.yaml", version_check_flag=False
    )
    training_data, parameters["headers"] = parseTrainingCSV(
        inputDir + "/train_2d_rad_segmentation.csv"
    )
    # patch_size is custom for sdnet
    parameters["patch_size"] = [224, 224, 1]
    parameters["batch_size"] = 2
    parameters["model"]["dimension"] = 2
    parameters["model"]["class_list"] = [0, 255]
    parameters["model"]["num_channels"] = 1
    parameters["model"]["architecture"] = "sdnet"
    parameters["model"]["onnx_export"] = False
    parameters["model"]["print_summary"] = False
    parameters = populate_header_in_parameters(parameters, parameters["headers"])
    sanitize_outputDir()
    TrainingManager(
        dataframe=training_data,
        outputDir=outputDir,
        parameters=parameters,
        device=device,
        resume=False,
        reset=True,
    )

    print("passed")


def test_train_segmentation_rad_3d(device):
    print("05: Starting 3D Rad segmentation tests")
    # read and parse csv
    # read and initialize parameters for specific data dimension
    parameters = parseConfig(
        testingDir + "/config_segmentation.yaml", version_check_flag=False
    )
    training_data, parameters["headers"] = parseTrainingCSV(
        inputDir + "/train_3d_rad_segmentation.csv"
    )
    parameters["modality"] = "rad"
    parameters["patch_size"] = patch_size["3D"]
    parameters["model"]["dimension"] = 3
    parameters["model"]["class_list"] = [0, 1]
    parameters["model"]["final_layer"] = "softmax"
    parameters["model"]["amp"] = True
    parameters["in_memory"] = True
    parameters["model"]["num_channels"] = len(parameters["headers"]["channelHeaders"])
    parameters["model"]["onnx_export"] = False
    parameters["model"]["print_summary"] = False
    parameters = populate_header_in_parameters(parameters, parameters["headers"])
    # loop through selected models and train for single epoch
    for model in all_models_segmentation:
        if model == "imagenet_unet":
            # imagenet_unet encoder needs to be toned down for small patch size
            parameters["model"]["encoder_depth"] = 3
            parameters["model"]["decoder_channels"] = (64, 32, 16)
            parameters["model"]["final_layer"] = random.choice(
                ["sigmoid", "softmax", "logsoftmax", "tanh", "identity"]
            )
            parameters["model"]["converter_type"] = random.choice(
                ["acs", "soft", "conv3d"]
            )
        parameters["model"]["architecture"] = model
        parameters["nested_training"]["testing"] = -5
        parameters["nested_training"]["validation"] = -5
        sanitize_outputDir()
        TrainingManager(
            dataframe=training_data,
            outputDir=outputDir,
            parameters=parameters,
            device=device,
            resume=False,
            reset=True,
        )

    print("passed")


def test_train_regression_rad_2d(device):
    print("06: Starting 2D Rad regression tests")
    # read and initialize parameters for specific data dimension
    parameters = parseConfig(
        testingDir + "/config_regression.yaml", version_check_flag=False
    )
    parameters["modality"] = "rad"
    parameters["patch_size"] = patch_size["2D"]
    parameters["model"]["dimension"] = 2
    parameters["model"]["amp"] = False
    # read and parse csv
    training_data, parameters["headers"] = parseTrainingCSV(
        inputDir + "/train_2d_rad_regression.csv"
    )
    parameters["model"]["num_channels"] = 3
    parameters["model"]["class_list"] = parameters["headers"]["predictionHeaders"]
    parameters["scaling_factor"] = 1
    parameters["model"]["onnx_export"] = False
    parameters["model"]["print_summary"] = False
    parameters = populate_header_in_parameters(parameters, parameters["headers"])
    # loop through selected models and train for single epoch
    for model in all_models_regression:
        parameters["model"]["architecture"] = model
        parameters["nested_training"]["testing"] = -5
        parameters["nested_training"]["validation"] = -5
        sanitize_outputDir()
        TrainingManager(
            dataframe=training_data,
            outputDir=outputDir,
            parameters=parameters,
            device=device,
            resume=False,
            reset=True,
        )

    print("passed")


def test_train_regression_rad_2d_imagenet(device):
    print("07: Starting 2D Rad regression tests for imagenet models")
    # read and initialize parameters for specific data dimension
    print("Starting 2D Rad regression tests for imagenet models")
    parameters = parseConfig(
        testingDir + "/config_regression.yaml", version_check_flag=False
    )
    parameters["patch_size"] = patch_size["2D"]
    parameters["model"]["dimension"] = 2
    parameters["model"]["amp"] = False
    parameters["model"]["print_summary"] = False
    # read and parse csv
    training_data, parameters["headers"] = parseTrainingCSV(
        inputDir + "/train_2d_rad_regression.csv"
    )
    parameters["model"]["num_channels"] = 3
    parameters["model"]["class_list"] = parameters["headers"]["predictionHeaders"]
    parameters["scaling_factor"] = 1
    parameters = populate_header_in_parameters(parameters, parameters["headers"])
    # loop through selected models and train for single epoch
    for model in all_models_classification:
        parameters["model"]["architecture"] = model
        parameters["nested_training"]["testing"] = 1
        parameters["nested_training"]["validation"] = -5
        sanitize_outputDir()
        TrainingManager(
            dataframe=training_data,
            outputDir=outputDir,
            parameters=parameters,
            device=device,
            resume=False,
            reset=True,
        )

    print("passed")


def test_train_regression_brainage_rad_2d(device):
    print("08: Starting brain age tests")
    # read and initialize parameters for specific data dimension
    parameters = parseConfig(
        testingDir + "/config_regression.yaml", version_check_flag=False
    )
    parameters["modality"] = "rad"
    parameters["patch_size"] = patch_size["2D"]
    parameters["model"]["dimension"] = 2
    parameters["model"]["amp"] = False
    # read and parse csv
    training_data, parameters["headers"] = parseTrainingCSV(
        inputDir + "/train_2d_rad_regression.csv"
    )
    parameters["model"]["num_channels"] = 3
    parameters["model"]["class_list"] = parameters["headers"]["predictionHeaders"]
    parameters["scaling_factor"] = 1
    parameters["model"]["architecture"] = "brain_age"
    parameters["model"]["onnx_export"] = False
    parameters["model"]["print_summary"] = False
    parameters = populate_header_in_parameters(parameters, parameters["headers"])
    sanitize_outputDir()
    TrainingManager(
        dataframe=training_data,
        outputDir=outputDir,
        parameters=parameters,
        device=device,
        resume=False,
        reset=True,
    )

    print("passed")


def test_train_regression_rad_3d(device):
    print("09: Starting 3D Rad regression tests")
    # read and initialize parameters for specific data dimension
    parameters = parseConfig(
        testingDir + "/config_regression.yaml", version_check_flag=False
    )
    parameters["modality"] = "rad"
    parameters["patch_size"] = patch_size["3D"]
    parameters["model"]["dimension"] = 3
    # read and parse csv
    training_data, parameters["headers"] = parseTrainingCSV(
        inputDir + "/train_3d_rad_regression.csv"
    )
    parameters["model"]["num_channels"] = len(parameters["headers"]["channelHeaders"])
    parameters["model"]["class_list"] = parameters["headers"]["predictionHeaders"]
    parameters["model"]["onnx_export"] = False
    parameters["model"]["print_summary"] = False
    parameters = populate_header_in_parameters(parameters, parameters["headers"])
    # loop through selected models and train for single epoch
    for model in all_models_regression:
        if "efficientnet" in model:
            parameters["patch_size"] = [16, 16, 16]
        else:
            parameters["patch_size"] = patch_size["3D"]

        if model == "imagenet_unet":
            parameters["model"]["depth"] = 2
            parameters["model"]["decoder_channels"] = [32, 16]
            parameters["model"]["encoder_weights"] = "None"
            parameters["model"]["converter_type"] = random.choice(
                ["acs", "soft", "conv3d"]
            )
        parameters["model"]["architecture"] = model
        parameters["nested_training"]["testing"] = -5
        parameters["nested_training"]["validation"] = -5
        sanitize_outputDir()
        TrainingManager(
            dataframe=training_data,
            outputDir=outputDir,
            parameters=parameters,
            device=device,
            resume=False,
            reset=True,
        )

    print("passed")


def test_train_classification_rad_2d(device):
    print("10: Starting 2D Rad classification tests")
    # read and initialize parameters for specific data dimension
    parameters = parseConfig(
        testingDir + "/config_classification.yaml", version_check_flag=False
    )
    parameters["modality"] = "rad"
    parameters["track_memory_usage"] = True
    parameters["patch_size"] = patch_size["2D"]
    parameters["model"]["dimension"] = 2
    # read and parse csv
    training_data, parameters["headers"] = parseTrainingCSV(
        inputDir + "/train_2d_rad_classification.csv"
    )
    parameters["model"]["num_channels"] = 3
    parameters["model"]["onnx_export"] = False
    parameters["model"]["print_summary"] = False
    parameters = populate_header_in_parameters(parameters, parameters["headers"])
    # loop through selected models and train for single epoch
    for model in all_models_regression:
        if model == "imagenet_unet":
            parameters["model"]["depth"] = 2
            parameters["model"]["decoder_channels"] = [32, 16]
            parameters["model"]["encoder_weights"] = "None"
            parameters["model"]["converter_type"] = random.choice(
                ["acs", "soft", "conv3d"]
            )
        parameters["model"]["architecture"] = model
        parameters["nested_training"]["testing"] = -5
        parameters["nested_training"]["validation"] = -5
        sanitize_outputDir()
        TrainingManager(
            dataframe=training_data,
            outputDir=outputDir,
            parameters=parameters,
            device=device,
            resume=False,
            reset=True,
        )

    # ensure sigmoid and softmax activations are tested for imagenet models
    for activation_type in ["sigmoid", "softmax"]:
        parameters["model"]["architecture"] = "imagenet_vgg11"
        parameters["model"]["final_layer"] = activation_type
        parameters["nested_training"]["testing"] = -5
        parameters["nested_training"]["validation"] = -5
        sanitize_outputDir()
        TrainingManager(
            dataframe=training_data,
            outputDir=outputDir,
            parameters=parameters,
            device=device,
            resume=False,
            reset=True,
        )

    print("passed")


def test_train_classification_rad_3d(device):
    print("11: Starting 3D Rad classification tests")
    # read and initialize parameters for specific data dimension
    parameters = parseConfig(
        testingDir + "/config_classification.yaml", version_check_flag=False
    )
    parameters["modality"] = "rad"
    parameters["patch_size"] = patch_size["3D"]
    parameters["model"]["dimension"] = 3
    # read and parse csv
    training_data, parameters["headers"] = parseTrainingCSV(
        inputDir + "/train_3d_rad_classification.csv"
    )
    parameters["model"]["num_channels"] = len(parameters["headers"]["channelHeaders"])
    parameters = populate_header_in_parameters(parameters, parameters["headers"])
    parameters["model"]["onnx_export"] = False
    parameters["model"]["print_summary"] = False
    # loop through selected models and train for single epoch
    for model in all_models_regression:
        if "efficientnet" in model:
            parameters["patch_size"] = [16, 16, 16]
        else:
            parameters["patch_size"] = patch_size["3D"]
        if model == "imagenet_unet":
            parameters["model"]["encoder_name"] = "efficientnet-b0"
            parameters["model"]["depth"] = 1
            parameters["model"]["decoder_channels"] = [64]
            parameters["model"]["final_layer"] = random.choice(
                ["sigmoid", "softmax", "logsoftmax", "tanh", "identity"]
            )
            parameters["model"]["converter_type"] = random.choice(
                ["acs", "soft", "conv3d"]
            )
        parameters["model"]["architecture"] = model
        parameters["nested_training"]["testing"] = -5
        parameters["nested_training"]["validation"] = -5
        sanitize_outputDir()
        TrainingManager(
            dataframe=training_data,
            outputDir=outputDir,
            parameters=parameters,
            device=device,
            resume=False,
            reset=True,
        )

    print("passed")


def test_train_resume_inference_classification_rad_3d(device):
    print("12: Starting 3D Rad classification tests for resume and reset")
    # read and initialize parameters for specific data dimension
    parameters = parseConfig(
        testingDir + "/config_classification.yaml", version_check_flag=False
    )
    parameters["modality"] = "rad"
    parameters["patch_size"] = patch_size["3D"]
    parameters["model"]["dimension"] = 3
    # read and parse csv
    training_data, parameters["headers"] = parseTrainingCSV(
        inputDir + "/train_3d_rad_classification.csv"
    )
    parameters["model"]["num_channels"] = len(parameters["headers"]["channelHeaders"])
    parameters = populate_header_in_parameters(parameters, parameters["headers"])
    # loop through selected models and train for single epoch
    model = all_models_regression[0]
    parameters["model"]["architecture"] = model
    parameters["model"]["onnx_export"] = False
    parameters["model"]["print_summary"] = False
    sanitize_outputDir()
    TrainingManager(
        dataframe=training_data,
        outputDir=outputDir,
        parameters=parameters,
        device=device,
        resume=False,
        reset=True,
    )

    ## testing resume with parameter updates
    parameters["num_epochs"] = 2
    parameters["nested_training"]["testing"] = -5
    parameters["nested_training"]["validation"] = -5
    parameters["model"]["save_at_every_epoch"] = True
    TrainingManager(
        dataframe=training_data,
        outputDir=outputDir,
        parameters=parameters,
        device=device,
        resume=True,
        reset=False,
    )

    ## testing resume without parameter updates
    parameters["num_epochs"] = 1
    parameters["nested_training"]["testing"] = -5
    parameters["nested_training"]["validation"] = -5
    TrainingManager(
        dataframe=training_data,
        outputDir=outputDir,
        parameters=parameters,
        device=device,
        resume=False,
        reset=False,
    )

    parameters["output_dir"] = outputDir  # this is in inference mode
    InferenceManager(
        dataframe=training_data,
        outputDir=outputDir,
        parameters=parameters,
        device=device,
    )

    print("passed")


def test_train_inference_optimize_classification_rad_3d(device):
    print("13: Starting 3D Rad segmentation tests for optimization")
    # read and initialize parameters for specific data dimension
    parameters = parseConfig(
        testingDir + "/config_classification.yaml", version_check_flag=False
    )
    parameters["modality"] = "rad"
    parameters["patch_size"] = patch_size["3D"]
    parameters["model"]["dimension"] = 3
    # read and parse csv
    training_data, parameters["headers"] = parseTrainingCSV(
        inputDir + "/train_3d_rad_classification.csv"
    )
    parameters["model"]["num_channels"] = len(parameters["headers"]["channelHeaders"])
    parameters = populate_header_in_parameters(parameters, parameters["headers"])
    parameters["model"]["architecture"] = all_models_regression[0]
    parameters["model"]["onnx_export"] = True
    parameters["model"]["print_summary"] = False
    sanitize_outputDir()
    TrainingManager(
        dataframe=training_data,
        outputDir=outputDir,
        parameters=parameters,
        device=device,
        resume=False,
        reset=True,
    )

    ## testing inference
    for model_type in all_model_type:
        parameters["model"]["type"] = model_type
        parameters["output_dir"] = outputDir  # this is in inference mode
        InferenceManager(
            dataframe=training_data,
            outputDir=outputDir,
            parameters=parameters,
            device=device,
        )

    print("passed")


def test_train_inference_optimize_segmentation_rad_2d(device):
    print("14: Starting 2D Rad segmentation tests for optimization")
    # read and parse csv
    parameters = parseConfig(
        testingDir + "/config_segmentation.yaml", version_check_flag=False
    )
    training_data, parameters["headers"] = parseTrainingCSV(
        inputDir + "/train_2d_rad_segmentation.csv"
    )
    parameters["patch_size"] = patch_size["2D"]
    parameters["modality"] = "rad"
    parameters["model"]["dimension"] = 2
    parameters["model"]["class_list"] = [0, 255]
    parameters["model"]["amp"] = True
    parameters["save_output"] = True
    parameters["model"]["num_channels"] = 3
    parameters["metrics"] = ["dice"]
    parameters["model"]["architecture"] = "resunet"
    parameters["model"]["onnx_export"] = True
    parameters["model"]["print_summary"] = False
    parameters = populate_header_in_parameters(parameters, parameters["headers"])
    sanitize_outputDir()
    TrainingManager(
        dataframe=training_data,
        outputDir=outputDir,
        parameters=parameters,
        device=device,
        resume=False,
        reset=True,
    )

    ## testing inference
    for model_type in all_model_type:
        parameters["model"]["type"] = model_type
        parameters["output_dir"] = outputDir  # this is in inference mode
        InferenceManager(
            dataframe=training_data,
            outputDir=outputDir,
            parameters=parameters,
            device=device,
        )

    print("passed")


def test_train_inference_classification_with_logits_single_fold_rad_3d(device):
    print("15: Starting 3D Rad classification tests for single fold logits inference")
    # read and initialize parameters for specific data dimension
    parameters = parseConfig(
        testingDir + "/config_classification.yaml", version_check_flag=False
    )
    parameters["modality"] = "rad"
    parameters["patch_size"] = patch_size["3D"]
    parameters["model"]["dimension"] = 3
    parameters["model"]["final_layer"] = "logits"

    # read and parse csv
    training_data, parameters["headers"] = parseTrainingCSV(
        inputDir + "/train_3d_rad_classification.csv"
    )
    parameters["model"]["num_channels"] = len(parameters["headers"]["channelHeaders"])
    parameters = populate_header_in_parameters(parameters, parameters["headers"])
    # loop through selected models and train for single epoch
    model = all_models_regression[0]
    parameters["model"]["architecture"] = model
    parameters["model"]["onnx_export"] = False
    parameters["model"]["print_summary"] = False
    sanitize_outputDir()
    TrainingManager(
        dataframe=training_data,
        outputDir=outputDir,
        parameters=parameters,
        device=device,
        resume=False,
        reset=True,
    )
    ## this is to test if inference can run without having ground truth column
    training_data.drop("ValueToPredict", axis=1, inplace=True)
    training_data.drop("Label", axis=1, inplace=True)
    temp_infer_csv = os.path.join(outputDir, "temp_infer_csv.csv")
    training_data.to_csv(temp_infer_csv, index=False)
    # read and parse csv
    parameters = parseConfig(
        testingDir + "/config_classification.yaml", version_check_flag=False
    )
    training_data, parameters["headers"] = parseTrainingCSV(temp_infer_csv)
    parameters["output_dir"] = outputDir  # this is in inference mode
    parameters["output_dir"] = outputDir  # this is in inference mode
    parameters["modality"] = "rad"
    parameters["patch_size"] = patch_size["3D"]
    parameters["model"]["dimension"] = 3
    parameters["model"]["final_layer"] = "logits"
    parameters["model"]["num_channels"] = len(parameters["headers"]["channelHeaders"])
    parameters = populate_header_in_parameters(parameters, parameters["headers"])
    # loop through selected models and train for single epoch
    model = all_models_regression[0]
    parameters["model"]["architecture"] = model
    parameters["model"]["onnx_export"] = False
    InferenceManager(
        dataframe=training_data,
        outputDir=outputDir,
        parameters=parameters,
        device=device,
    )

    print("passed")


def test_train_inference_classification_with_logits_multiple_folds_rad_3d(device):
    print("16: Starting 3D Rad classification tests for multi-fold logits inference")
    # read and initialize parameters for specific data dimension
    parameters = parseConfig(
        testingDir + "/config_classification.yaml", version_check_flag=False
    )
    parameters["modality"] = "rad"
    parameters["patch_size"] = patch_size["3D"]
    parameters["model"]["dimension"] = 3
    parameters["model"]["final_layer"] = "logits"
    # necessary for n-fold cross-validation inference
    parameters["nested_training"]["validation"] = 2
    parameters["model"]["onnx_export"] = False
    parameters["model"]["print_summary"] = False
    # read and parse csv
    training_data, parameters["headers"] = parseTrainingCSV(
        inputDir + "/train_3d_rad_classification.csv"
    )
    parameters["model"]["num_channels"] = len(parameters["headers"]["channelHeaders"])
    parameters = populate_header_in_parameters(parameters, parameters["headers"])
    # loop through selected models and train for single epoch
    model = all_models_regression[0]
    parameters["model"]["architecture"] = model
    sanitize_outputDir()
    TrainingManager(
        dataframe=training_data,
        outputDir=outputDir,
        parameters=parameters,
        device=device,
        resume=False,
        reset=True,
    )
    parameters["output_dir"] = outputDir  # this is in inference mode
    InferenceManager(
        dataframe=training_data,
        outputDir=outputDir,
        parameters=parameters,
        device=device,
    )

    print("passed")


def test_train_scheduler_classification_rad_2d(device):
    print("17: Starting 2D Rad segmentation tests for scheduler")
    # read and initialize parameters for specific data dimension
    # loop through selected models and train for single epoch
    for scheduler in global_schedulers_dict:
        parameters = parseConfig(
            testingDir + "/config_classification.yaml", version_check_flag=False
        )
        parameters["modality"] = "rad"
        parameters["patch_size"] = patch_size["2D"]
        parameters["model"]["dimension"] = 2
        # read and parse csv
        training_data, parameters["headers"] = parseTrainingCSV(
            inputDir + "/train_2d_rad_classification.csv"
        )
        parameters["model"]["num_channels"] = 3
        parameters["model"]["architecture"] = "densenet121"
        parameters["model"]["norm_type"] = "instance"
        parameters = populate_header_in_parameters(parameters, parameters["headers"])
        parameters["model"]["onnx_export"] = False
        parameters["model"]["print_summary"] = False
        parameters["scheduler"] = {}
        parameters["scheduler"]["type"] = scheduler
        parameters["nested_training"]["testing"] = -5
        parameters["nested_training"]["validation"] = -5
        sanitize_outputDir()
        ## ensure parameters are parsed every single time
        file_config_temp = get_temp_config_path()

        with open(file_config_temp, "w") as file:
            yaml.dump(parameters, file)

        parameters = parseConfig(file_config_temp, version_check_flag=False)
        TrainingManager(
            dataframe=training_data,
            outputDir=outputDir,
            parameters=parameters,
            device=device,
            resume=False,
            reset=True,
        )

    print("passed")


def test_train_optimizer_classification_rad_2d(device):
    print("18: Starting 2D Rad classification tests for optimizer")
    # read and initialize parameters for specific data dimension
    parameters = parseConfig(
        testingDir + "/config_classification.yaml", version_check_flag=False
    )
    parameters["modality"] = "rad"
    parameters["patch_size"] = patch_size["2D"]
    parameters["model"]["dimension"] = 2
    # read and parse csv
    training_data, parameters["headers"] = parseTrainingCSV(
        inputDir + "/train_2d_rad_classification.csv"
    )
    parameters["model"]["num_channels"] = 3
    parameters["model"]["architecture"] = "densenet121"
    parameters["model"]["norm_type"] = "none"
    parameters["model"]["onnx_export"] = False
    parameters["model"]["print_summary"] = False
    parameters = populate_header_in_parameters(parameters, parameters["headers"])
    # loop through selected models and train for single epoch
    for optimizer in global_optimizer_dict:
        parameters["optimizer"] = {}
        parameters["optimizer"]["type"] = optimizer
        parameters["nested_training"]["testing"] = -5
        parameters["nested_training"]["validation"] = -5
        if os.path.exists(outputDir):
            shutil.rmtree(outputDir)  # overwrite previous results
        Path(outputDir).mkdir(parents=True, exist_ok=True)
        TrainingManager(
            dataframe=training_data,
            outputDir=outputDir,
            parameters=parameters,
            device=device,
            resume=False,
            reset=True,
        )

    print("passed")


def test_clip_train_classification_rad_3d(device):
    print("19: Starting 3D Rad classification tests for clipping")
    # read and initialize parameters for specific data dimension
    parameters = parseConfig(
        testingDir + "/config_classification.yaml", version_check_flag=False
    )
    parameters["modality"] = "rad"
    parameters["patch_size"] = patch_size["3D"]
    parameters["model"]["dimension"] = 3
    # read and parse csv
    training_data, parameters["headers"] = parseTrainingCSV(
        inputDir + "/train_3d_rad_classification.csv"
    )
    parameters["model"]["num_channels"] = len(parameters["headers"]["channelHeaders"])
    parameters["model"]["architecture"] = "vgg16"
    parameters["model"]["norm_type"] = "None"
    parameters["model"]["onnx_export"] = False
    parameters["model"]["print_summary"] = False
    parameters = populate_header_in_parameters(parameters, parameters["headers"])
    # loop through selected models and train for single epoch
    for clip_mode in all_clip_modes:
        parameters["clip_mode"] = clip_mode
        parameters["nested_training"]["testing"] = -5
        parameters["nested_training"]["validation"] = -5
        sanitize_outputDir()
        TrainingManager(
            dataframe=training_data,
            outputDir=outputDir,
            parameters=parameters,
            device=device,
            resume=False,
            reset=True,
        )
    print("passed")


def test_train_normtype_segmentation_rad_3d(device):
    print("20: Starting 3D Rad segmentation tests for normtype")
    # read and initialize parameters for specific data dimension
    # read and parse csv
    # read and initialize parameters for specific data dimension
    parameters = parseConfig(
        testingDir + "/config_segmentation.yaml", version_check_flag=False
    )
    training_data, parameters["headers"] = parseTrainingCSV(
        inputDir + "/train_3d_rad_segmentation.csv"
    )
    parameters["patch_size"] = patch_size["3D"]
    parameters["model"]["dimension"] = 3
    parameters["model"]["class_list"] = [0, 1]
    parameters["model"]["amp"] = True
    parameters["save_output"] = True
    parameters["data_postprocessing"] = {"fill_holes"}
    parameters["in_memory"] = True
    parameters["model"]["onnx_export"] = False
    parameters["model"]["print_summary"] = False
    parameters["model"]["num_channels"] = len(parameters["headers"]["channelHeaders"])
    parameters = populate_header_in_parameters(parameters, parameters["headers"])
    # loop through selected models and train for single epoch
    for norm in ["batch", "instance"]:
        for model in ["resunet", "unet", "fcn", "unetr"]:
            parameters["model"]["architecture"] = model
            parameters["model"]["norm_type"] = norm
            parameters["nested_training"]["testing"] = -5
            parameters["nested_training"]["validation"] = -5
            if os.path.isdir(outputDir):
                shutil.rmtree(outputDir)  # overwrite previous results
            Path(outputDir).mkdir(parents=True, exist_ok=True)
            TrainingManager(
                dataframe=training_data,
                outputDir=outputDir,
                parameters=parameters,
                device=device,
                resume=False,
                reset=True,
            )

        print("passed")


def test_train_metrics_segmentation_rad_2d(device):
    print("21: Starting 2D Rad segmentation tests for metrics")
    # read and parse csv
    parameters = parseConfig(
        testingDir + "/config_segmentation.yaml", version_check_flag=False
    )
    parameters["modality"] = "rad"
    parameters["patch_size"] = patch_size["2D"]
    parameters["model"]["dimension"] = 2
    parameters["model"]["class_list"] = [0, 255]
    parameters["data_postprocessing"] = {"mapping": {0: 0, 255: 1}}
    parameters["model"]["amp"] = True
    parameters["save_output"] = True
    parameters["model"]["num_channels"] = 3
    parameters["metrics"] = ["dice", "hausdorff", "hausdorff95"]
    parameters["model"]["architecture"] = "resunet"
    parameters["model"]["onnx_export"] = False
    parameters["model"]["print_summary"] = False
    file_config_temp = get_temp_config_path()

    with open(file_config_temp, "w") as file:
        yaml.dump(parameters, file)

    parameters = parseConfig(file_config_temp, version_check_flag=False)
    training_data, parameters["headers"] = parseTrainingCSV(
        inputDir + "/train_2d_rad_segmentation.csv"
    )
    parameters = populate_header_in_parameters(parameters, parameters["headers"])
    sanitize_outputDir()
    TrainingManager(
        dataframe=training_data,
        outputDir=outputDir,
        parameters=parameters,
        device=device,
        resume=False,
        reset=True,
    )

    print("passed")


def test_train_metrics_regression_rad_2d(device):
    print("22: Starting 2D Rad regression tests for metrics")
    # read and parse csv
    parameters = parseConfig(
        testingDir + "/config_regression.yaml", version_check_flag=False
    )
    training_data, parameters["headers"] = parseTrainingCSV(
        inputDir + "/train_2d_rad_regression.csv"
    )
    parameters["modality"] = "rad"
    parameters["patch_size"] = patch_size["2D"]
    parameters["model"]["dimension"] = 2
    parameters["model"]["class_list"] = [0, 255]
    parameters["model"]["norm_type"] = "instance"
    parameters["model"]["amp"] = False
    parameters["model"]["num_channels"] = 3
    parameters["model"]["architecture"] = "vgg11"
    parameters["model"]["onnx_export"] = False
    parameters["model"]["print_summary"] = True
    parameters = populate_header_in_parameters(parameters, parameters["headers"])
    sanitize_outputDir()
    TrainingManager(
        dataframe=training_data,
        outputDir=outputDir,
        parameters=parameters,
        device=device,
        resume=False,
        reset=True,
    )

    print("passed")


def test_train_losses_segmentation_rad_2d(device):
    print("23: Starting 2D Rad segmentation tests for losses")
    # read and parse csv
    parameters = parseConfig(
        testingDir + "/config_segmentation.yaml", version_check_flag=False
    )
    training_data, parameters["headers"] = parseTrainingCSV(
        inputDir + "/train_2d_rad_segmentation.csv"
    )
    parameters["modality"] = "rad"
    parameters["patch_size"] = patch_size["2D"]
    parameters["model"]["dimension"] = 2
    parameters["model"]["class_list"] = [0, 255]
    # disabling amp because some losses do not support Half, yet
    parameters["model"]["amp"] = False
    parameters["model"]["num_channels"] = 3
    parameters["model"]["architecture"] = "resunet"
    parameters["metrics"] = ["dice"]
    parameters["model"]["onnx_export"] = False
    parameters["model"]["print_summary"] = False
    parameters = populate_header_in_parameters(parameters, parameters["headers"])
    # loop through selected models and train for single epoch
    for loss_type in ["dc", "dc_log", "dcce", "dcce_logits", "tversky"]:
        parameters["loss_function"] = loss_type
        parameters["nested_training"]["testing"] = -5
        parameters["nested_training"]["validation"] = -5
        sanitize_outputDir()
        TrainingManager(
            dataframe=training_data,
            outputDir=outputDir,
            parameters=parameters,
            device=device,
            resume=False,
            reset=True,
        )

    print("passed")


def test_generic_config_read():
    print("24: Starting testing reading configuration")
    file_config_temp = get_temp_config_path()
    parameters = parseConfig(
        os.path.join(baseConfigDir, "config_all_options.yaml"),
        version_check_flag=False,
    )
    parameters["data_preprocessing"]["resize_image"] = [128, 128]

    with open(file_config_temp, "w") as file:
        yaml.dump(parameters, file)

    # read and parse csv
    parameters = parseConfig(file_config_temp, version_check_flag=True)

    training_data, parameters["headers"] = parseTrainingCSV(
        inputDir + "/train_2d_rad_segmentation.csv"
    )
    assert parameters is not None, "parameters is None"
    data_loader = ImagesFromDataFrame(training_data, parameters, True, "unit_test")
    assert data_loader is not None, "data_loader is None"

    os.remove(file_config_temp)

    # ensure resize_image is triggered
    parameters["data_preprocessing"].pop("resample")
    parameters["data_preprocessing"].pop("resample_min")
    parameters["data_preprocessing"]["resize_image"] = [128, 128]
    parameters["model"]["print_summary"] = False

    with open(file_config_temp, "w") as file:
        yaml.dump(parameters, file)

    parameters = parseConfig(file_config_temp, version_check_flag=True)

    training_data, parameters["headers"] = parseTrainingCSV(
        inputDir + "/train_2d_rad_segmentation.csv"
    )
    assert parameters is not None, "parameters is None"
    data_loader = ImagesFromDataFrame(training_data, parameters, True, "unit_test")
    assert data_loader is not None, "data_loader is None"

    os.remove(file_config_temp)

    # ensure resize_patch is triggered
    parameters["data_preprocessing"].pop("resize_image")
    parameters["data_preprocessing"]["resize_patch"] = [64, 64]

    with open(file_config_temp, "w") as file:
        yaml.dump(parameters, file)

    parameters = parseConfig(file_config_temp, version_check_flag=True)

    training_data, parameters["headers"] = parseTrainingCSV(
        inputDir + "/train_2d_rad_segmentation.csv"
    )
    assert parameters is not None, "parameters is None"
    data_loader = ImagesFromDataFrame(training_data, parameters, True, "unit_test")
    assert data_loader is not None, "data_loader is None"

    os.remove(file_config_temp)

    # ensure resize_image is triggered
    parameters["data_preprocessing"].pop("resize_patch")
    parameters["data_preprocessing"]["resize"] = [64, 64]

    with open(file_config_temp, "w") as file:
        yaml.dump(parameters, file)

    parameters = parseConfig(file_config_temp, version_check_flag=True)

    training_data, parameters["headers"] = parseTrainingCSV(
        inputDir + "/train_2d_rad_segmentation.csv"
    )
    assert parameters is not None, "parameters is None"
    data_loader = ImagesFromDataFrame(training_data, parameters, True, "unit_test")
    assert data_loader is not None, "data_loader is None"

    os.remove(file_config_temp)

    print("passed")


def test_generic_cli_function_preprocess():
    print("25: Starting testing cli function preprocess")
    file_config = os.path.join(testingDir, "config_segmentation.yaml")
    sanitize_outputDir()
    file_config_temp = get_temp_config_path()
    file_data = os.path.join(inputDir, "train_2d_rad_segmentation.csv")

    parameters = parseConfig(file_config)
    parameters["modality"] = "rad"
    parameters["patch_size"] = patch_size["2D"]
    parameters["model"]["dimension"] = 2
    parameters["model"]["class_list"] = "[0, 255||125]"
    # disabling amp because some losses do not support Half, yet
    parameters["model"]["amp"] = False
    parameters["model"]["print_summary"] = False
    parameters["model"]["num_channels"] = 3
    parameters["model"]["architecture"] = "unet"
    parameters["metrics"] = ["dice"]
    parameters["patch_sampler"] = "label"
    parameters["weighted_loss"] = True
    parameters["save_output"] = True
    parameters["data_preprocessing"]["to_canonical"] = None
    parameters["data_preprocessing"]["rgba_to_rgb"] = None

    # store this separately for preprocess testing
    with open(file_config_temp, "w") as outfile:
        yaml.dump(parameters, outfile, default_flow_style=False)

    preprocess_and_save(file_data, file_config_temp, outputDir)
    training_data, parameters["headers"] = parseTrainingCSV(
        outputDir + "/data_processed.csv"
    )

    # check that the length of training data is what we expect
    assert len(training_data) == 10, "Number of rows in dataframe is not 10"

    print("passed")


def test_generic_cli_function_mainrun(device):
    print("26: Starting testing cli function main_run")
    parameters = parseConfig(
        testingDir + "/config_segmentation.yaml", version_check_flag=False
    )
    file_config_temp = get_temp_config_path()

    parameters["modality"] = "rad"
    parameters["patch_size"] = patch_size["2D"]
    parameters["num_epochs"] = 1
    parameters["nested_training"]["testing"] = 1
    parameters["model"]["dimension"] = 2
    parameters["model"]["class_list"] = [0, 255]
    parameters["model"]["amp"] = True
    parameters["model"]["print_summary"] = False
    parameters["model"]["num_channels"] = 3
    parameters["metrics"] = [
        "dice",
    ]
    parameters["model"]["architecture"] = "unet"

    with open(file_config_temp, "w") as file:
        yaml.dump(parameters, file)

    file_data = os.path.join(inputDir, "train_2d_rad_segmentation.csv")

    main_run(
        file_data, file_config_temp, outputDir, True, device, resume=False, reset=True
    )
    sanitize_outputDir()

    with open(file_config_temp, "w") as file:
        yaml.dump(parameters, file)

    # testing train/valid split
    main_run(
        file_data + "," + file_data,
        file_config_temp,
        outputDir,
        True,
        device,
        resume=False,
        reset=True,
    )
    sanitize_outputDir()

    print("passed")


def test_dataloader_construction_train_segmentation_3d(device):
    print("27: Starting 3D Rad segmentation tests")
    # read and parse csv
    # read and initialize parameters for specific data dimension
    parameters = parseConfig(
        testingDir + "/config_segmentation.yaml", version_check_flag=False
    )
    params_all_preprocessing_and_augs = parseConfig(
        os.path.join(baseConfigDir, "config_all_options.yaml")
    )

    # take preprocessing and augmentations from all options
    for key in ["data_preprocessing", "data_augmentation"]:
        parameters[key] = params_all_preprocessing_and_augs[key]

    # customize parameters to maximize test coverage
    parameters["data_preprocessing"].pop("normalize", None)
    parameters["data_preprocessing"]["normalize_nonZero"] = None
    parameters["data_preprocessing"]["default_probability"] = 1
    parameters.pop("nested_training", None)
    parameters["nested_training"] = {}
    parameters["nested_training"]["testing"] = 1
    parameters["nested_training"]["validation"] = -5

    training_data, parameters["headers"] = parseTrainingCSV(
        inputDir + "/train_3d_rad_segmentation.csv"
    )
    parameters["modality"] = "rad"
    parameters["patch_size"] = patch_size["3D"]
    parameters["save_training"] = True
    parameters["save_output"] = True
    parameters["model"]["dimension"] = 3
    parameters["model"]["class_list"] = [0, 1]
    parameters["model"]["amp"] = True
    parameters["model"]["num_channels"] = len(parameters["headers"]["channelHeaders"])
    parameters["model"]["architecture"] = "unet"
    parameters["weighted_loss"] = False
    parameters["model"]["onnx_export"] = False
    parameters["model"]["print_summary"] = False
    parameters["data_postprocessing"]["mapping"] = {0: 0, 1: 1}
    parameters["data_postprocessing"]["fill_holes"] = True
    parameters = populate_header_in_parameters(parameters, parameters["headers"])
    # loop through selected models and train for single epoch
    sanitize_outputDir()
    TrainingManager(
        dataframe=training_data,
        outputDir=outputDir,
        parameters=parameters,
        device=device,
        resume=False,
        reset=True,
    )

    print("passed")


def test_generic_preprocess_functions():
    print("28: Starting testing preprocessing functions")
    # initialize an input which has values between [-1,1]
    # checking tensor with last dimension of size 1
    input_tensor = torch.rand(4, 256, 256, 1)
    input_transformed = global_preprocessing_dict["rgba2rgb"]()(input_tensor)
    assert input_transformed.shape[1] == 3, "Number of channels is not 3"

    input_tensor = torch.rand(3, 256, 256, 1)
    input_transformed = global_preprocessing_dict["rgb2rgba"]()(input_tensor)
    assert input_transformed.shape[1] == 4, "Number of channels is not 4"

    input_tensor = 2 * torch.rand(3, 256, 256, 1) - 1
    input_transformed = global_preprocessing_dict["normalize_div_by_255"](input_tensor)
    input_tensor = 2 * torch.rand(1, 3, 256, 256) - 1
    input_transformed = global_preprocessing_dict["normalize_imagenet"](input_tensor)
    input_transformed = global_preprocessing_dict["normalize_standardize"](input_tensor)
    input_transformed = global_preprocessing_dict["normalize_div_by_255"](input_tensor)
    parameters_dict = {}
    parameters_dict["min"] = 0.25
    parameters_dict["max"] = 0.75
    input_transformed = global_preprocessing_dict["threshold"](parameters_dict)(
        input_tensor
    )
    assert (
        torch.count_nonzero(
            input_transformed[input_transformed < parameters_dict["min"]]
            > parameters_dict["max"]
        )
        == 0
    ), "Input should be thresholded"

    input_transformed = global_preprocessing_dict["clip"](parameters_dict)(input_tensor)
    assert (
        torch.count_nonzero(
            input_transformed[input_transformed < parameters_dict["min"]]
            > parameters_dict["max"]
        )
        == 0
    ), "Input should be clipped"

    non_zero_normalizer = global_preprocessing_dict["normalize_nonZero_masked"]
    input_transformed = non_zero_normalizer(input_tensor)
    non_zero_normalizer = global_preprocessing_dict["normalize_positive"]
    input_transformed = non_zero_normalizer(input_tensor)
    non_zero_normalizer = global_preprocessing_dict["normalize_nonZero"]
    input_transformed = non_zero_normalizer(input_tensor)

    ## stain_normalization checks
    input_tensor = 2 * torch.rand(3, 256, 256, 1) + 10
    training_data, _ = parseTrainingCSV(inputDir + "/train_2d_rad_segmentation.csv")
    parameters_temp = {}
    parameters_temp["data_preprocessing"] = {}
    parameters_temp["data_preprocessing"]["stain_normalizer"] = {
        "target": training_data["Channel_0"][0]
    }
    for extractor in ["ruifrok", "macenko", "vahadane"]:
        parameters_temp["data_preprocessing"]["stain_normalizer"][
            "extractor"
        ] = extractor
        non_zero_normalizer = global_preprocessing_dict["stain_normalizer"](
            parameters_temp["data_preprocessing"]["stain_normalizer"]
        )
        input_transformed = non_zero_normalizer(input_tensor)

    ## histogram matching tests
    # histogram equalization
    input_tensor = torch.rand(1, 64, 64, 64)
    parameters_temp = {}
    parameters_temp["data_preprocessing"] = {}
    parameters_temp["data_preprocessing"]["histogram_matching"] = {}
    non_zero_normalizer = global_preprocessing_dict["histogram_matching"](
        parameters_temp["data_preprocessing"]["histogram_matching"]
    )
    input_transformed = non_zero_normalizer(input_tensor)
    # adaptive histogram equalization
    parameters_temp = {}
    parameters_temp["data_preprocessing"] = {}
    parameters_temp["data_preprocessing"]["histogram_matching"] = {"target": "adaptive"}
    non_zero_normalizer = global_preprocessing_dict["histogram_matching"](
        parameters_temp["data_preprocessing"]["histogram_matching"]
    )
    input_transformed = non_zero_normalizer(input_tensor)
    # histogram matching
    training_data, _ = parseTrainingCSV(inputDir + "/train_3d_rad_segmentation.csv")
    parameters_temp = {}
    parameters_temp["data_preprocessing"] = {}
    parameters_temp["data_preprocessing"]["histogram_matching"] = {
        "target": training_data["Channel_0"][0]
    }
    non_zero_normalizer = global_preprocessing_dict["histogram_matching"](
        parameters_temp["data_preprocessing"]["histogram_matching"]
    )
    input_transformed = non_zero_normalizer(input_tensor)

    # fill holes
    input_tensor = torch.rand(1, 256, 256, 256) > 0.5
    input_transformed = fill_holes(input_tensor)

    ## CCA tests
    # 3d
    input_tensor = torch.rand(1, 256, 256, 256) > 0.5
    input_transformed = cca(input_tensor)
    # 2d
    input_tensor = torch.rand(1, 256, 256) > 0.5
    input_transformed = cca(input_tensor)
    # 2d rgb
    input_tensor = torch.rand(1, 3, 256, 256) > 0.5
    input_transformed = cca(input_tensor)

    input_tensor = torch.rand(1, 256, 256, 256)
    cropper = global_preprocessing_dict["crop_external_zero_planes"](
        patch_size=[128, 128, 128]
    )
    input_transformed = cropper(input_tensor)

    cropper = global_preprocessing_dict["crop"]([64, 64, 64])
    input_transformed = cropper(input_tensor)
    assert input_transformed.shape == (1, 128, 128, 128), "Cropping should work"

    cropper = global_preprocessing_dict["centercrop"]([128, 128, 128])
    input_transformed = cropper(input_tensor)
    assert input_transformed.shape == (1, 128, 128, 128), "Center-crop should work"

    # test pure morphological operations
    input_tensor_3d = torch.rand(1, 1, 256, 256, 256)
    input_tensor_2d = torch.rand(1, 3, 256, 256)
    for mode in ["dilation", "erosion", "opening", "closing"]:
        input_transformed_3d = torch_morphological(input_tensor_3d, mode=mode)
        assert len(input_transformed_3d.shape) == 5, "Output should be 5D"
        input_transformed_2d = torch_morphological(input_tensor_2d, mode=mode)
        assert len(input_transformed_2d.shape) == 4, "Output should be 4D"

    # test for failure
    with pytest.raises(Exception) as exc_info:
        input_tensor_4d = torch.rand(1, 1, 32, 32, 32, 32)
        input_transformed_3d = torch_morphological(input_tensor_4d)

    print("Exception raised:", exc_info.value)

    # test obtaining arrays
    input_tensor_3d = torch.rand(256, 256, 256)
    input_array = get_array_from_image_or_tensor(input_tensor_3d)
    assert isinstance(input_array, np.ndarray), "Array should be obtained from tensor"
    input_image = sitk.GetImageFromArray(input_array)
    input_array = get_array_from_image_or_tensor(input_image)
    assert isinstance(input_array, np.ndarray), "Array should be obtained from image"
    input_array = get_array_from_image_or_tensor(input_array)
    assert isinstance(input_array, np.ndarray), "Array should be obtained from array"

    with pytest.raises(Exception) as exc_info:
        input_list = [0, 1]
        input_array = get_array_from_image_or_tensor(input_list)
    exception_raised = exc_info.value
    print("Exception raised: ", exception_raised)

    ## image rescaling test
    input_tensor = torch.randint(0, 256, (1, 64, 64, 64))
    # try out different options
    for params in [
        {},
        None,
        {"in_min_max": [5, 250], "out_min_max": [-1, 2]},
        {"out_min_max": [0, 1], "percentiles": [5, 95]},
    ]:
        rescaler = global_preprocessing_dict["rescale"](params)
        input_transformed = rescaler(input_tensor)
        assert (
            input_transformed.min() >= rescaler.out_min_max[0]
        ), "Rescaling should work for min"
        assert (
            input_transformed.max() <= rescaler.out_min_max[1]
        ), "Rescaling should work for max"

    print("passed")


def test_generic_augmentation_functions():
    print("29: Starting testing augmentation functions")
    params_all_preprocessing_and_augs = parseConfig(
        os.path.join(baseConfigDir, "config_all_options.yaml")
    )

    # this is for rgb augmentation
    input_tensor = torch.rand(3, 128, 128, 1)
    temp = global_augs_dict["colorjitter"](
        params_all_preprocessing_and_augs["data_augmentation"]["colorjitter"]
    )
    output_tensor = None
    output_tensor = temp(input_tensor)
    assert output_tensor != None, "RGB Augmentation should work"

    # ensuring all code paths are covered
    for key in ["brightness", "contrast", "saturation", "hue"]:
        params_all_preprocessing_and_augs["data_augmentation"]["colorjitter"][
            key
        ] = 0.25
    temp = global_augs_dict["colorjitter"](
        params_all_preprocessing_and_augs["data_augmentation"]["colorjitter"]
    )
    output_tensor = None
    output_tensor = temp(input_tensor)
    assert output_tensor != None, "RGB Augmentation should work"

    # this is for all other augmentations
    input_tensor = torch.rand(3, 128, 128, 128)
    for aug in params_all_preprocessing_and_augs["data_augmentation"]:
        aug_lower = aug.lower()
        output_tensor = None
        if aug_lower in global_augs_dict:
            print(aug_lower)
            output_tensor = global_augs_dict[aug](
                params_all_preprocessing_and_augs["data_augmentation"][aug_lower]
            )(input_tensor)
            assert output_tensor != None, "Augmentation should work"

    print("passed")


def test_train_checkpointing_segmentation_rad_2d(device):
    print("30: Starting 2D Rad segmentation tests for metrics")
    # read and parse csv
    parameters = parseConfig(
        testingDir + "/config_segmentation.yaml", version_check_flag=False
    )
    training_data, parameters["headers"] = parseTrainingCSV(
        inputDir + "/train_2d_rad_segmentation.csv"
    )
    parameters["modality"] = "rad"
    parameters["patch_size"] = patch_size["2D"]
    parameters["num_epochs"] = 1
    parameters["nested_training"]["testing"] = 1
    parameters["model"]["dimension"] = 2
    parameters["model"]["class_list"] = [0, 255]
    parameters["model"]["amp"] = True
    parameters["model"]["num_channels"] = 3
    parameters["metrics"] = [
        "dice",
        "dice_per_label",
        "hausdorff",
        "hausdorff95",
        "hd95_per_label",
        "hd100_per_label",
    ]
    parameters["model"]["architecture"] = "unet"
    parameters["model"]["onnx_export"] = False
    parameters["model"]["print_summary"] = False
    parameters = populate_header_in_parameters(parameters, parameters["headers"])
    sanitize_outputDir()
    TrainingManager(
        dataframe=training_data,
        outputDir=outputDir,
        parameters=parameters,
        device=device,
        resume=False,
        reset=True,
    )
    parameters["num_epochs"] = 2
    parameters["nested_training"]["validation"] = -2
    parameters["nested_training"]["testing"] = 1
    TrainingManager(
        dataframe=training_data,
        outputDir=outputDir,
        parameters=parameters,
        device=device,
        resume=False,
        reset=False,
    )

    print("passed")


def test_generic_model_patch_divisibility():
    print("31: Starting patch divisibility tests")
    parameters = parseConfig(
        testingDir + "/config_segmentation.yaml", version_check_flag=False
    )
    _, parameters["headers"] = parseTrainingCSV(
        inputDir + "/train_2d_rad_segmentation.csv"
    )
    parameters["model"]["architecture"] = "unet"
    parameters["patch_size"] = [127, 127, 1]
    parameters["num_epochs"] = 1
    parameters["nested_training"]["testing"] = 1
    parameters["model"]["dimension"] = 2
    parameters["model"]["class_list"] = [0, 255]
    parameters["model"]["amp"] = True
    parameters["model"]["print_summary"] = False
    parameters["model"]["num_channels"] = 3
    parameters["metrics"] = ["dice", "hausdorff", "hausdorff95"]
    parameters = populate_header_in_parameters(parameters, parameters["headers"])

    # this assertion should fail
    with pytest.raises(BaseException) as e_info:
        global_models_dict[parameters["model"]["architecture"]](parameters=parameters)

    parameters["model"]["architecture"] = "uinc"
    parameters["model"]["base_filters"] = 11

    # this assertion should fail
    with pytest.raises(BaseException) as e_info:
        global_models_dict[parameters["model"]["architecture"]](parameters=parameters)

    print("passed")


def test_generic_one_hot_logic():
    print("32: Starting one hot logic tests")
    random_array = np.random.randint(5, size=(20, 20, 20))
    img = sitk.GetImageFromArray(random_array)
    img_array = sitk.GetArrayFromImage(img)
    img_tensor = torch.from_numpy(img_array).to(torch.float16)
    img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)

    class_list = [*range(0, np.max(random_array) + 1)]
    img_tensor_oh = one_hot(img_tensor, class_list)
    img_tensor_oh_rev_array = reverse_one_hot(img_tensor_oh[0], class_list)
    comparison = random_array == img_tensor_oh_rev_array
    assert comparison.all(), "Arrays are not equal"

    class_list = ["0", "1||2||3", np.max(random_array)]
    img_tensor_oh = one_hot(img_tensor, class_list)
    img_tensor_oh_rev_array = reverse_one_hot(img_tensor_oh[0], class_list)

    # check for background
    comparison = (random_array == 0) == (img_tensor_oh_rev_array == 0)
    assert comparison.all(), "Arrays at '0' are not equal"

    # check last foreground
    comparison = (random_array == np.max(random_array)) == (
        img_tensor_oh_rev_array == len(class_list) - 1
    )
    assert comparison.all(), "Arrays at final foreground are not equal"

    # check combined foreground
    combined_array = np.logical_or(
        np.logical_or((random_array == 1), (random_array == 2)), (random_array == 3)
    )
    comparison = combined_array == (img_tensor_oh_rev_array == 1)
    assert comparison.all(), "Arrays at the combined foreground are not equal"

    parameters = {"data_postprocessing": {}}
    mapped_output = get_mapped_label(
        torch.from_numpy(img_tensor_oh_rev_array), parameters
    )

    parameters = {}
    mapped_output = get_mapped_label(
        torch.from_numpy(img_tensor_oh_rev_array), parameters
    )

    parameters = {"data_postprocessing": {"mapping": {0: 0, 1: 1, 2: 5}}}
    mapped_output = get_mapped_label(
        torch.from_numpy(img_tensor_oh_rev_array), parameters
    )

    for key, value in parameters["data_postprocessing"]["mapping"].items():
        comparison = (img_tensor_oh_rev_array == key) == (mapped_output == value)
        assert comparison.all(), "Arrays at {}:{} are not equal".format(key, value)

    # check the case where 0 is present as an int in a special case
    class_list = [0, "1||2||3", np.max(random_array)]
    img_tensor_oh = one_hot(img_tensor, class_list)
    img_tensor_oh_rev_array = reverse_one_hot(img_tensor_oh[0], class_list)

    # check for background
    comparison = (random_array == 0) == (img_tensor_oh_rev_array == 0)
    assert comparison.all(), "Arrays at '0' are not equal"

    # check the case where 0 is absent from class_list
    class_list = ["1||2||3", np.max(random_array)]
    img_tensor_oh = one_hot(img_tensor, class_list)
    img_tensor_oh_rev_array = reverse_one_hot(img_tensor_oh[0], class_list)

    # check last foreground
    comparison = (random_array == np.max(random_array)) == (
        img_tensor_oh_rev_array == len(class_list)
    )
    assert comparison.all(), "Arrays at final foreground are not equal"

    # check combined foreground
    combined_array = np.logical_or(
        np.logical_or((random_array == 1), (random_array == 2)), (random_array == 3)
    )
    comparison = combined_array == (img_tensor_oh_rev_array == 1)
    assert comparison.all(), "Arrays at the combined foreground are not equal"

    print("passed")


def test_generic_anonymizer():
    print("33: Starting anomymizer tests")
    input_file = get_testdata_file("MR_small.dcm")

    output_file = os.path.join(outputDir, "MR_small_anonymized.dcm")

    config_file = os.path.join(baseConfigDir, "config_anonymizer.yaml")

    run_anonymizer(input_file, output_file, config_file, "rad")
    assert os.path.exists(output_file), "Anonymized file does not exist"

    # test defaults
    run_anonymizer(input_file, output_file, None, "rad")
    assert os.path.exists(output_file), "Anonymized file does not exist"

    # test nifti conversion
    config_file_for_nifti = os.path.join(outputDir, "config_anonymizer_nifti.yaml")
    with open(config_file, "r") as file_data:
        yaml_data = file_data.read()
    parameters = yaml.safe_load(yaml_data)
    parameters["convert_to_nifti"] = True
    with open(config_file_for_nifti, "w") as file:
        yaml.dump(parameters, file)

    # for nifti conversion, the input needs to be in a dir
    input_folder_for_nifti = os.path.join(outputDir, "nifti_input")
    Path(input_folder_for_nifti).mkdir(parents=True, exist_ok=True)
    shutil.copyfile(input_file, os.path.join(input_folder_for_nifti, "MR_small.dcm"))

    output_file = os.path.join(outputDir, "MR_small.nii.gz")

    run_anonymizer(input_folder_for_nifti, output_file, config_file_for_nifti, "rad")
    assert os.path.exists(output_file), "Anonymized file does not exist"

    if not os.path.exists(output_file):
        raise Exception("Output NIfTI file was not created")

    input_file = os.path.join(inputDir, "2d_histo_segmentation", "1", "image.tiff")
    output_file_histo = os.path.join(outputDir, "histo_anon.tiff")
    # this assertion should fail since histo anonymizer is not implementer
    with pytest.raises(BaseException) as exc_info:
        run_anonymizer(input_folder_for_nifti, output_file_histo, None, "histo")
        assert os.path.exists(output_file_histo), "Anonymized file does not exist"
    print("Exception raised: ", exc_info.value)
    sanitize_outputDir()

    print("passed")


def test_train_inference_segmentation_histology_2d(device):
    print("34: Starting histology train/inference segmentation tests")
    # overwrite previous results
    sanitize_outputDir()
    output_dir_patches = os.path.join(outputDir, "histo_patches")
    if os.path.isdir(output_dir_patches):
        shutil.rmtree(output_dir_patches)
    Path(output_dir_patches).mkdir(parents=True, exist_ok=True)
    output_dir_patches_output = os.path.join(output_dir_patches, "histo_patches_output")
    Path(output_dir_patches_output).mkdir(parents=True, exist_ok=True)
    file_config_temp = get_temp_config_path()

    parameters_patch = {}
    # extracting minimal number of patches to ensure that the test does not take too long
    parameters_patch["num_patches"] = 3
    # define patches to be extracted in terms of microns
    parameters_patch["patch_size"] = ["1000m", "1000m"]

    with open(file_config_temp, "w") as file:
        yaml.dump(parameters_patch, file)

    patch_extraction(
        inputDir + "/train_2d_histo_segmentation.csv",
        output_dir_patches_output,
        file_config_temp,
    )

    file_for_Training = os.path.join(output_dir_patches_output, "opm_train.csv")
    # read and parse csv
    parameters = parseConfig(
        testingDir + "/config_segmentation.yaml", version_check_flag=False
    )
    training_data, parameters["headers"] = parseTrainingCSV(file_for_Training)
    parameters["patch_size"] = patch_size["2D"]
    parameters["modality"] = "histo"
    parameters["model"]["dimension"] = 2
    parameters["model"]["class_list"] = [0, 255]
    parameters["model"]["amp"] = True
    parameters["model"]["num_channels"] = 3
    parameters = populate_header_in_parameters(parameters, parameters["headers"])
    parameters["model"]["architecture"] = "resunet"
    parameters["nested_training"]["testing"] = 1
    parameters["nested_training"]["validation"] = -2
    parameters["metrics"] = ["dice"]
    parameters["model"]["onnx_export"] = True
    parameters["model"]["print_summary"] = True
    parameters["data_preprocessing"]["resize_image"] = [128, 128]
    modelDir = os.path.join(outputDir, "modelDir")
    Path(modelDir).mkdir(parents=True, exist_ok=True)
    TrainingManager(
        dataframe=training_data,
        outputDir=modelDir,
        parameters=parameters,
        device=device,
        resume=False,
        reset=True,
    )
    inference_data, parameters["headers"] = parseTrainingCSV(
        inputDir + "/train_2d_histo_segmentation.csv", train=False
    )
    inference_data.drop(index=inference_data.index[-1], axis=0, inplace=True)
    InferenceManager(
        dataframe=inference_data,
        outputDir=modelDir,
        parameters=parameters,
        device=device,
    )

    print("passed")


def test_train_inference_classification_histology_large_2d(device):
    print(
        "35: Starting histology train/inference classification tests for large images to check exception handling"
    )
    # overwrite previous results
    sanitize_outputDir()
    output_dir_patches = os.path.join(outputDir, "histo_patches")
    if os.path.isdir(output_dir_patches):
        shutil.rmtree(output_dir_patches)
    Path(output_dir_patches).mkdir(parents=True, exist_ok=True)
    output_dir_patches_output = os.path.join(output_dir_patches, "histo_patches_output")
    Path(output_dir_patches_output).mkdir(parents=True, exist_ok=True)
    file_config_temp = get_temp_config_path()

    for sub in ["1", "2"]:
        file_to_check = os.path.join(
            inputDir, "2d_histo_segmentation", sub, "image_resize.tiff"
        )
        if os.path.exists(file_to_check):
            os.remove(file_to_check)

    parameters_patch = {}
    # extracting minimal number of patches to ensure that the test does not take too long
    parameters_patch["num_patches"] = 3
    parameters_patch["patch_size"] = [128, 128]

    with open(file_config_temp, "w") as file:
        yaml.dump(parameters_patch, file)

    # resize the image
    input_df = pd.read_csv(inputDir + "/train_2d_histo_classification.csv")
    files_to_delete = []
    for _, row in input_df.iterrows():
        scaling_factor = 10
        new_filename = row["Channel_0"].replace(".tiff", "_resize.tiff")
        try:
            img = cv2.imread(row["Channel_0"])
            dims = img.shape
            img_resize = cv2.resize(
                img, (dims[1] * scaling_factor, dims[0] * scaling_factor)
            )
            cv2.imwrite(new_filename, img_resize)
        except:
            # this is only used in CI
            try:
                os.system(
                    "vips resize "
                    + row["Channel_0"]
                    + " "
                    + new_filename
                    + " "
                    + str(scaling_factor)
                )
            except:
                print("Resize could not be done")
                break
        row["Channel_0"] = new_filename
        files_to_delete.append(new_filename)

    resized_inference_data_list = os.path.join(
        inputDir, "train_2d_histo_classification_resize.csv"
    )
    # drop last subject
    input_df.drop(index=input_df.index[-1], axis=0, inplace=True)
    input_df.to_csv(resized_inference_data_list, index=False)
    files_to_delete.append(resized_inference_data_list)

    patch_extraction(
        inputDir + "/train_2d_histo_classification.csv",
        output_dir_patches_output,
        file_config_temp,
    )

    file_for_Training = os.path.join(output_dir_patches_output, "opm_train.csv")
    temp_df = pd.read_csv(file_for_Training)
    temp_df.drop("Label", axis=1, inplace=True)
    temp_df["valuetopredict"] = np.random.randint(2, size=len(temp_df))
    temp_df.to_csv(file_for_Training, index=False)
    # read and parse csv
    parameters = parseConfig(
        testingDir + "/config_classification.yaml", version_check_flag=False
    )
    parameters["modality"] = "histo"
    parameters["patch_size"] = parameters_patch["patch_size"][0]
    file_config_temp = get_temp_config_path()
    with open(file_config_temp, "w") as file:
        yaml.dump(parameters, file)
    parameters = parseConfig(file_config_temp, version_check_flag=False)
    parameters["model"]["dimension"] = 2
    # read and parse csv
    training_data, parameters["headers"] = parseTrainingCSV(file_for_Training)
    parameters["model"]["num_channels"] = 3
    parameters["model"]["architecture"] = "densenet121"
    parameters["model"]["norm_type"] = "none"
    parameters["data_preprocessing"]["rgba2rgb"] = ""
    parameters = populate_header_in_parameters(parameters, parameters["headers"])
    parameters["nested_training"]["testing"] = 1
    parameters["nested_training"]["validation"] = -2
    parameters["model"]["print_summary"] = False
    modelDir = os.path.join(outputDir, "modelDir")
    if os.path.isdir(modelDir):
        shutil.rmtree(modelDir)
    Path(modelDir).mkdir(parents=True, exist_ok=True)
    TrainingManager(
        dataframe=training_data,
        outputDir=modelDir,
        parameters=parameters,
        device=device,
        resume=False,
        reset=True,
    )
    parameters["output_dir"] = modelDir  # this is in inference mode
    parameters["data_preprocessing"]["resize_patch"] = parameters_patch["patch_size"]
    parameters["patch_size"] = [
        parameters_patch["patch_size"][0] * 10,
        parameters_patch["patch_size"][1] * 10,
    ]
    parameters["nested_training"]["validation"] = 1
    inference_data, parameters["headers"] = parseTrainingCSV(
        resized_inference_data_list, train=False
    )
    for model_type in all_model_type:
        parameters["model"]["type"] = model_type
        InferenceManager(
            dataframe=inference_data,
            outputDir=modelDir,
            parameters=parameters,
            device=device,
        )
        # if 'predictions.csv' are not found, give error
        output_subject_dir = os.path.join(modelDir, str(input_df["SubjectID"][0]))
        assert (
            os.path.exists(os.path.join(output_subject_dir, "predictions.csv")) is True
        )
        # ensure previous results are removed
        shutil.rmtree(output_subject_dir)

    for file in files_to_delete:
        os.remove(file)

    print("passed")


def test_train_inference_classification_histology_2d(device):
    print("36: Starting histology train/inference classification tests")
    # overwrite previous results
    sanitize_outputDir()
    output_dir_patches = os.path.join(outputDir, "histo_patches")
    if os.path.isdir(output_dir_patches):
        shutil.rmtree(output_dir_patches)
    Path(output_dir_patches).mkdir(parents=True, exist_ok=True)
    output_dir_patches_output = os.path.join(output_dir_patches, "histo_patches_output")
    Path(output_dir_patches_output).mkdir(parents=True, exist_ok=True)
    file_config_temp = get_temp_config_path()

    parameters_patch = {}
    # extracting minimal number of patches to ensure that the test does not take too long
    parameters_patch["num_patches"] = 3
    parameters_patch["patch_size"] = [128, 128]

    with open(file_config_temp, "w") as file:
        yaml.dump(parameters_patch, file)

    patch_extraction(
        inputDir + "/train_2d_histo_classification.csv",
        output_dir_patches_output,
        file_config_temp,
    )

    file_for_Training = os.path.join(output_dir_patches_output, "opm_train.csv")
    temp_df = pd.read_csv(file_for_Training)
    temp_df.drop("Label", axis=1, inplace=True)
    temp_df["valuetopredict"] = np.random.randint(2, size=6)
    temp_df.to_csv(file_for_Training, index=False)
    # read and parse csv
    parameters = parseConfig(
        testingDir + "/config_classification.yaml", version_check_flag=False
    )
    parameters["modality"] = "histo"
    parameters["patch_size"] = 128
    file_config_temp = get_temp_config_path()
    with open(file_config_temp, "w") as file:
        yaml.dump(parameters, file)
    parameters = parseConfig(file_config_temp, version_check_flag=False)
    parameters["model"]["dimension"] = 2
    # read and parse csv
    training_data, parameters["headers"] = parseTrainingCSV(file_for_Training)
    parameters["model"]["num_channels"] = 3
    parameters["model"]["architecture"] = "densenet121"
    parameters["model"]["norm_type"] = "none"
    parameters["data_preprocessing"]["rgba2rgb"] = ""
    parameters = populate_header_in_parameters(parameters, parameters["headers"])
    parameters["nested_training"]["testing"] = 1
    parameters["nested_training"]["validation"] = -2
    parameters["model"]["print_summary"] = False
    modelDir = os.path.join(outputDir, "modelDir")
    if os.path.isdir(modelDir):
        shutil.rmtree(modelDir)
    Path(modelDir).mkdir(parents=True, exist_ok=True)
    TrainingManager(
        dataframe=training_data,
        outputDir=modelDir,
        parameters=parameters,
        device=device,
        resume=False,
        reset=True,
    )
    parameters["output_dir"] = modelDir  # this is in inference mode
    inference_data, parameters["headers"] = parseTrainingCSV(
        inputDir + "/train_2d_histo_classification.csv", train=False
    )
    for model_type in all_model_type:
        parameters["nested_training"]["testing"] = 1
        parameters["nested_training"]["validation"] = -2
        parameters["output_dir"] = modelDir  # this is in inference mode
        inference_data, parameters["headers"] = parseTrainingCSV(
            inputDir + "/train_2d_histo_segmentation.csv", train=False
        )
        parameters["model"]["type"] = model_type
        InferenceManager(
            dataframe=inference_data,
            outputDir=modelDir,
            parameters=parameters,
            device=device,
        )

    print("passed")


def test_train_segmentation_unet_layerchange_rad_2d(device):
    # test case to up code coverage --> test decreasing allowed layers for unet
    print("37: Starting 2D Rad segmentation tests for normtype")
    # read and parse csv
    # read and initialize parameters for specific data dimension
    parameters = parseConfig(
        testingDir + "/config_segmentation.yaml", version_check_flag=False
    )
    training_data, parameters["headers"] = parseTrainingCSV(
        inputDir + "/train_2d_rad_segmentation.csv"
    )
    for model in ["unet_multilayer", "lightunet_multilayer", "unetr"]:
        parameters["model"]["architecture"] = model
        parameters["patch_size"] = [4, 4, 1]
        parameters["model"]["dimension"] = 2

        # this assertion should fail
        with pytest.raises(BaseException) as e_info:
            global_models_dict[parameters["model"]["architecture"]](
                parameters=parameters
            )

        parameters["patch_size"] = patch_size["2D"]
        parameters["model"]["depth"] = 7
        parameters["model"]["class_list"] = [0, 255]
        parameters["model"]["amp"] = True
        parameters["model"]["print_summary"] = False
        parameters["model"]["num_channels"] = 3
        parameters = populate_header_in_parameters(parameters, parameters["headers"])
        # loop through selected models and train for single epoch
        parameters["model"]["norm_type"] = "batch"
        parameters["nested_training"]["testing"] = -5
        parameters["nested_training"]["validation"] = -5
        if os.path.isdir(outputDir):
            shutil.rmtree(outputDir)  # overwrite previous results
        sanitize_outputDir()
        TrainingManager(
            dataframe=training_data,
            outputDir=outputDir,
            parameters=parameters,
            device=device,
            resume=False,
            reset=True,
        )

    print("passed")


def test_train_segmentation_unetr_rad_3d(device):
    print("38: Testing UNETR for 3D segmentation")
    parameters = parseConfig(
        testingDir + "/config_segmentation.yaml", version_check_flag=False
    )
    training_data, parameters["headers"] = parseTrainingCSV(
        inputDir + "/train_3d_rad_segmentation.csv"
    )
    parameters["model"]["architecture"] = "unetr"
    parameters["patch_size"] = [4, 4, 4]
    parameters["model"]["dimension"] = 3
    parameters["model"]["depth"] = 2
    parameters["model"]["print_summary"] = False

    # this assertion should fail
    with pytest.raises(BaseException) as e_info:
        global_models_dict[parameters["model"]["architecture"]](parameters=parameters)

    parameters["model"]["dimension"] = 3
    parameters["patch_size"] = [32, 32, 32]

    with pytest.raises(BaseException) as e_info:
        parameters["model"]["inner_patch_size"] = 19
        global_models_dict[parameters["model"]["architecture"]](parameters=parameters)

    with pytest.raises(BaseException) as e_info:
        parameters["model"]["inner_patch_size"] = 64
        global_models_dict[parameters["model"]["architecture"]](parameters=parameters)

    for patch in [16, 8]:
        parameters["model"]["inner_patch_size"] = patch
        parameters["model"]["class_list"] = [0, 255]
        parameters["model"]["amp"] = True
        parameters["model"]["num_channels"] = len(
            parameters["headers"]["channelHeaders"]
        )
        parameters = populate_header_in_parameters(parameters, parameters["headers"])
        # loop through selected models and train for single epoch
        parameters["model"]["norm_type"] = "batch"
        parameters["nested_training"]["testing"] = -5
        parameters["nested_training"]["validation"] = -5
        if os.path.isdir(outputDir):
            shutil.rmtree(outputDir)  # overwrite previous results
        sanitize_outputDir()
        TrainingManager(
            dataframe=training_data,
            outputDir=outputDir,
            parameters=parameters,
            device=device,
            resume=False,
            reset=True,
        )

    print("passed")


def test_train_segmentation_unetr_rad_2d(device):
    print("39: Testing UNETR for 2D segmentation")
    parameters = parseConfig(
        testingDir + "/config_segmentation.yaml", version_check_flag=False
    )
    training_data, parameters["headers"] = parseTrainingCSV(
        inputDir + "/train_2d_rad_segmentation.csv"
    )
    parameters["model"]["architecture"] = "unetr"
    parameters["patch_size"] = [128, 128, 1]
    parameters["model"]["dimension"] = 2

    for patch in [16, 8]:
        parameters["model"]["inner_patch_size"] = patch
        parameters["model"]["class_list"] = [0, 255]
        parameters["model"]["amp"] = True
        parameters["model"]["print_summary"] = False
        parameters["model"]["num_channels"] = 3
        parameters = populate_header_in_parameters(parameters, parameters["headers"])
        # loop through selected models and train for single epoch
        parameters["model"]["norm_type"] = "batch"
        parameters["nested_training"]["testing"] = -5
        parameters["nested_training"]["validation"] = -5
        if os.path.isdir(outputDir):
            shutil.rmtree(outputDir)  # overwrite previous results
        sanitize_outputDir()
        TrainingManager(
            dataframe=training_data,
            outputDir=outputDir,
            parameters=parameters,
            device=device,
            resume=False,
            reset=True,
        )

    print("passed")


def test_train_segmentation_transunet_rad_2d(device):
    print("40: Testing TransUNet for 2D segmentation")
    parameters = parseConfig(
        testingDir + "/config_segmentation.yaml", version_check_flag=False
    )
    training_data, parameters["headers"] = parseTrainingCSV(
        inputDir + "/train_2d_rad_segmentation.csv"
    )
    parameters["model"]["architecture"] = "transunet"
    parameters["patch_size"] = [128, 128, 1]
    parameters["model"]["dimension"] = 2
    parameters["model"]["print_summary"] = False

    with pytest.raises(BaseException) as e_info:
        parameters["model"]["num_heads"] = 6
        parameters["model"]["embed_dim"] = 64
        global_models_dict[parameters["model"]["architecture"]](parameters=parameters)

    with pytest.raises(BaseException) as e_info:
        parameters["model"]["num_heads"] = 3
        parameters["model"]["embed_dim"] = 50
        global_models_dict[parameters["model"]["architecture"]](parameters=parameters)

    parameters["model"]["embed_dim"] = 64
    parameters["model"]["depth"] = 2
    parameters["model"]["class_list"] = [0, 255]
    parameters["model"]["num_heads"] = 8
    parameters["model"]["amp"] = True
    parameters["model"]["num_channels"] = 3
    parameters = populate_header_in_parameters(parameters, parameters["headers"])
    # loop through selected models and train for single epoch
    parameters["model"]["norm_type"] = "batch"
    parameters["nested_training"]["testing"] = -5
    parameters["nested_training"]["validation"] = -5
    if os.path.isdir(outputDir):
        shutil.rmtree(outputDir)  # overwrite previous results
    sanitize_outputDir()
    TrainingManager(
        dataframe=training_data,
        outputDir=outputDir,
        parameters=parameters,
        device=device,
        resume=False,
        reset=True,
    )

    print("passed")


def test_train_segmentation_transunet_rad_3d(device):
    print("41: Testing TransUNet for 3D segmentation")
    parameters = parseConfig(
        testingDir + "/config_segmentation.yaml", version_check_flag=False
    )
    training_data, parameters["headers"] = parseTrainingCSV(
        inputDir + "/train_3d_rad_segmentation.csv"
    )
    parameters["model"]["architecture"] = "transunet"
    parameters["patch_size"] = [4, 4, 4]
    parameters["model"]["dimension"] = 3
    parameters["model"]["print_summary"] = False

    # this assertion should fail
    with pytest.raises(BaseException) as e_info:
        global_models_dict[parameters["model"]["architecture"]](parameters=parameters)

    parameters["model"]["dimension"] = 3
    parameters["patch_size"] = [32, 32, 32]

    with pytest.raises(BaseException) as e_info:
        parameters["model"]["depth"] = 1
        global_models_dict[parameters["model"]["architecture"]](parameters=parameters)

    with pytest.raises(BaseException) as e_info:
        parameters["model"]["num_heads"] = 6
        parameters["model"]["embed_dim"] = 64
        global_models_dict[parameters["model"]["architecture"]](parameters=parameters)

    with pytest.raises(BaseException) as e_info:
        parameters["model"]["num_heads"] = 3
        parameters["model"]["embed_dim"] = 50
        global_models_dict[parameters["model"]["architecture"]](parameters=parameters)

    parameters["model"]["num_heads"] = 8
    parameters["model"]["embed_dim"] = 64
    parameters["model"]["depth"] = 2
    parameters["model"]["class_list"] = [0, 255]
    parameters["model"]["amp"] = True
    parameters["model"]["num_channels"] = len(parameters["headers"]["channelHeaders"])
    parameters = populate_header_in_parameters(parameters, parameters["headers"])
    # loop through selected models and train for single epoch
    parameters["model"]["norm_type"] = "batch"
    parameters["nested_training"]["testing"] = -5
    parameters["nested_training"]["validation"] = -5
    if os.path.isdir(outputDir):
        shutil.rmtree(outputDir)  # overwrite previous results
    sanitize_outputDir()
    TrainingManager(
        dataframe=training_data,
        outputDir=outputDir,
        parameters=parameters,
        device=device,
        resume=False,
        reset=True,
    )

    print("passed")


def test_train_gradient_clipping_classification_rad_2d(device):
    print("42: Testing gradient clipping")
    # read and initialize parameters for specific data dimension
    parameters = parseConfig(
        testingDir + "/config_classification.yaml", version_check_flag=False
    )
    parameters["modality"] = "rad"
    parameters["track_memory_usage"] = True
    parameters["patch_size"] = patch_size["2D"]
    parameters["model"]["dimension"] = 2
    # read and parse csv
    training_data, parameters["headers"] = parseTrainingCSV(
        inputDir + "/train_2d_rad_classification.csv"
    )
    parameters["model"]["num_channels"] = 3
    parameters["model"]["onnx_export"] = False
    parameters["model"]["print_summary"] = False
    parameters = populate_header_in_parameters(parameters, parameters["headers"])
    # ensure gradient clipping is getting tested
    for clip_mode in ["norm", "value", "agc"]:
        parameters["model"]["architecture"] = "imagenet_vgg11"
        parameters["model"]["final_layer"] = "softmax"
        parameters["nested_training"]["testing"] = -5
        parameters["nested_training"]["validation"] = -5
        parameters["clip_mode"] = clip_mode
        sanitize_outputDir()
        TrainingManager(
            dataframe=training_data,
            outputDir=outputDir,
            parameters=parameters,
            device=device,
            resume=False,
            reset=True,
        )
    print("passed")


def test_train_segmentation_unet_conversion_rad_3d(device):
    print("43: Starting 3D Rad segmentation tests for unet with ACS conversion")
    # read and parse csv
    # read and initialize parameters for specific data dimension
    parameters = parseConfig(
        testingDir + "/config_segmentation.yaml", version_check_flag=False
    )
    training_data, parameters["headers"] = parseTrainingCSV(
        inputDir + "/train_3d_rad_segmentation.csv"
    )
    parameters["modality"] = "rad"
    parameters["patch_size"] = patch_size["3D"]
    parameters["model"]["dimension"] = 3
    parameters["model"]["class_list"] = [0, 1]
    parameters["model"]["final_layer"] = "softmax"
    parameters["model"]["amp"] = True
    parameters["in_memory"] = True
    parameters["verbose"] = False
    parameters["model"]["num_channels"] = len(parameters["headers"]["channelHeaders"])
    parameters["model"]["onnx_export"] = False
    parameters["model"]["print_summary"] = False
    parameters = populate_header_in_parameters(parameters, parameters["headers"])
    # loop through selected models and train for single epoch
    for model in ["unet", "unet_multilayer", "lightunet_multilayer"]:
        parameters["model"]["converter_type"] = random.choice(["acs", "soft", "conv3d"])
        parameters["model"]["architecture"] = model
        parameters["nested_training"]["testing"] = -5
        parameters["nested_training"]["validation"] = -5
        sanitize_outputDir()
        TrainingManager(
            dataframe=training_data,
            outputDir=outputDir,
            parameters=parameters,
            device=device,
            resume=False,
            reset=True,
        )

    print("passed")

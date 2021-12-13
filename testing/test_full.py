from pathlib import Path
import requests, zipfile, io, os, csv, random, copy, shutil, sys, yaml, torch, pytest
import SimpleITK as sitk
import numpy as np

from GANDLF.data.ImagesFromDataFrame import ImagesFromDataFrame
from GANDLF.utils import *
from GANDLF.data.preprocessing import global_preprocessing_dict
from GANDLF.data.augmentation import global_augs_dict
from GANDLF.parseConfig import parseConfig
from GANDLF.training_manager import TrainingManager
from GANDLF.inference_manager import InferenceManager
from GANDLF.cli.main_run import main_run
from GANDLF.cli.preprocess_and_save import preprocess_and_save
from GANDLF.schedulers import global_schedulers_dict
from GANDLF.optimizers import global_optimizer_dict
from GANDLF.models import global_models_dict

device = "cpu"
## global defines
# pre-defined segmentation model types for testing
all_models_segmentation = [
    "lightunet",
    "unet",
    "deep_resunet",
    "fcn",
    "uinc",
    "msdnet",
]
# pre-defined regression/classification model types for testing
all_models_regression = ["densenet121", "vgg16"]
all_clip_modes = ["norm", "value", "agc"]
all_norm_type = ["batch", "instance"]

patch_size = {"2D": [128, 128, 1], "3D": [32, 32, 32]}

baseConfigDir = os.path.abspath(os.path.normpath("./samples"))
testingDir = os.path.abspath(os.path.normpath("./testing"))
inputDir = os.path.abspath(os.path.normpath("./testing/data"))
outputDir = os.path.abspath(os.path.normpath("./testing/data_output"))
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


def test_download_data():
    """
    This function downloads the sample data, which is the first step towards getting everything ready
    """
    urlToDownload = "https://github.com/CBICA/GaNDLF/raw/master/testing/data.zip"
    # do not download data again
    if not Path(
        os.getcwd() + "/testing/data/test/3d_rad_segmentation/001/image.nii.gz"
    ).exists():
        print("Downloading and extracting sample data")
        r = requests.get(urlToDownload)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall("./testing")


def test_constructTrainingCSV():
    """
    This function constructs training csv
    """
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


def test_train_segmentation_rad_2d(device):
    print("Starting 2D Rad segmentation tests")
    # read and parse csv
    parameters = parseConfig(
        testingDir + "/config_segmentation.yaml", version_check_flag=False
    )
    training_data, parameters["headers"] = parseTrainingCSV(
        inputDir + "/train_2d_rad_segmentation.csv"
    )
    parameters = populate_header_in_parameters(parameters, parameters["headers"])
    parameters["patch_size"] = patch_size["2D"]
    parameters["model"]["dimension"] = 2
    parameters["model"]["class_list"] = [0, 255]
    parameters["model"]["amp"] = True
    parameters["model"]["num_channels"] = 3
    # read and initialize parameters for specific data dimension
    for model in all_models_segmentation:
        parameters["model"]["architecture"] = model
        parameters["nested_training"]["testing"] = -5
        parameters["nested_training"]["validation"] = -5
        shutil.rmtree(outputDir)  # overwrite previous results
        Path(outputDir).mkdir(parents=True, exist_ok=True)
        TrainingManager(
            dataframe=training_data,
            outputDir=outputDir,
            parameters=parameters,
            device=device,
            reset_prev=True,
        )

    print("passed")


def test_train_segmentation_sdnet_rad_2d(device):
    print("Starting 2D Rad segmentation tests")
    # read and parse csv
    parameters = parseConfig(
        testingDir + "/config_segmentation.yaml", version_check_flag=False
    )
    training_data, parameters["headers"] = parseTrainingCSV(
        inputDir + "/train_2d_rad_segmentation.csv"
    )
    parameters = populate_header_in_parameters(parameters, parameters["headers"])
    # patch_size is custom for sdnet
    parameters["patch_size"] = [224, 224, 1]
    parameters["batch_size"] = 2
    parameters["model"]["dimension"] = 2
    parameters["model"]["class_list"] = [0, 255]
    parameters["model"]["num_channels"] = 1
    parameters["model"]["architecture"] = "sdnet"
    shutil.rmtree(outputDir)  # overwrite previous results
    Path(outputDir).mkdir(parents=True, exist_ok=True)
    TrainingManager(
        dataframe=training_data,
        outputDir=outputDir,
        parameters=parameters,
        device=device,
        reset_prev=True,
    )

    print("passed")


def test_train_segmentation_rad_3d(device):
    print("Starting 3D Rad segmentation tests")
    # read and parse csv
    # read and initialize parameters for specific data dimension
    parameters = parseConfig(
        testingDir + "/config_segmentation.yaml", version_check_flag=False
    )
    training_data, parameters["headers"] = parseTrainingCSV(
        inputDir + "/train_3d_rad_segmentation.csv"
    )
    parameters = populate_header_in_parameters(parameters, parameters["headers"])
    parameters["patch_size"] = patch_size["3D"]
    parameters["model"]["dimension"] = 3
    parameters["model"]["class_list"] = [0, 1]
    parameters["model"]["amp"] = True
    parameters["in_memory"] = True
    parameters["model"]["num_channels"] = len(parameters["headers"]["channelHeaders"])
    # loop through selected models and train for single epoch
    for model in all_models_segmentation:
        parameters["model"]["architecture"] = model
        parameters["nested_training"]["testing"] = -5
        parameters["nested_training"]["validation"] = -5
        shutil.rmtree(outputDir)  # overwrite previous results
        Path(outputDir).mkdir(parents=True, exist_ok=True)
        TrainingManager(
            dataframe=training_data,
            outputDir=outputDir,
            parameters=parameters,
            device=device,
            reset_prev=True,
        )

    print("passed")


def test_train_regression_rad_2d(device):
    # read and initialize parameters for specific data dimension
    parameters = parseConfig(
        testingDir + "/config_regression.yaml", version_check_flag=False
    )
    parameters["patch_size"] = patch_size["2D"]
    parameters["model"]["dimension"] = 2
    parameters["model"]["amp"] = False
    # read and parse csv
    training_data, parameters["headers"] = parseTrainingCSV(
        inputDir + "/train_2d_rad_regression.csv"
    )
    parameters = populate_header_in_parameters(parameters, parameters["headers"])
    parameters["model"]["num_channels"] = 3
    parameters["model"]["class_list"] = parameters["headers"]["predictionHeaders"]
    parameters["scaling_factor"] = 1
    # loop through selected models and train for single epoch
    for model in all_models_regression:
        parameters["model"]["architecture"] = model
        parameters["nested_training"]["testing"] = -5
        parameters["nested_training"]["validation"] = -5
        shutil.rmtree(outputDir)  # overwrite previous results
        Path(outputDir).mkdir(parents=True, exist_ok=True)
        TrainingManager(
            dataframe=training_data,
            outputDir=outputDir,
            parameters=parameters,
            device=device,
            reset_prev=True,
        )

    print("passed")


def test_train_brainage_rad_2d(device):
    # read and initialize parameters for specific data dimension
    parameters = parseConfig(
        testingDir + "/config_regression.yaml", version_check_flag=False
    )
    parameters["patch_size"] = patch_size["2D"]
    parameters["model"]["dimension"] = 2
    parameters["model"]["amp"] = False
    # read and parse csv
    training_data, parameters["headers"] = parseTrainingCSV(
        inputDir + "/train_2d_rad_regression.csv"
    )
    parameters = populate_header_in_parameters(parameters, parameters["headers"])
    parameters["model"]["num_channels"] = 3
    parameters["model"]["class_list"] = parameters["headers"]["predictionHeaders"]
    parameters["scaling_factor"] = 1
    parameters["model"]["architecture"] = "brain_age"
    shutil.rmtree(outputDir)  # overwrite previous results
    Path(outputDir).mkdir(parents=True, exist_ok=True)
    TrainingManager(
        dataframe=training_data,
        outputDir=outputDir,
        parameters=parameters,
        device=device,
        reset_prev=True,
    )

    print("passed")


def test_train_regression_rad_3d(device):
    # read and initialize parameters for specific data dimension
    parameters = parseConfig(
        testingDir + "/config_regression.yaml", version_check_flag=False
    )
    parameters["patch_size"] = patch_size["3D"]
    parameters["model"]["dimension"] = 3
    parameters["model"]["amp"] = True
    # read and parse csv
    training_data, parameters["headers"] = parseTrainingCSV(
        inputDir + "/train_3d_rad_regression.csv"
    )
    parameters = populate_header_in_parameters(parameters, parameters["headers"])
    parameters["model"]["num_channels"] = len(parameters["headers"]["channelHeaders"])
    parameters["model"]["class_list"] = parameters["headers"]["predictionHeaders"]
    # loop through selected models and train for single epoch
    for model in all_models_regression:
        parameters["model"]["architecture"] = model
        parameters["nested_training"]["testing"] = -5
        parameters["nested_training"]["validation"] = -5
        shutil.rmtree(outputDir)  # overwrite previous results
        Path(outputDir).mkdir(parents=True, exist_ok=True)
        TrainingManager(
            dataframe=training_data,
            outputDir=outputDir,
            parameters=parameters,
            device=device,
            reset_prev=True,
        )

    print("passed")


def test_train_classification_rad_2d(device):
    # read and initialize parameters for specific data dimension
    parameters = parseConfig(
        testingDir + "/config_classification.yaml", version_check_flag=False
    )
    parameters["modality"] = "rad"
    parameters["track_memory_usage"] = True
    parameters["patch_size"] = patch_size["2D"]
    parameters["model"]["dimension"] = 2
    parameters["model"]["amp"] = True
    # read and parse csv
    training_data, parameters["headers"] = parseTrainingCSV(
        inputDir + "/train_2d_rad_classification.csv"
    )
    parameters = populate_header_in_parameters(parameters, parameters["headers"])
    parameters["model"]["num_channels"] = 3
    # loop through selected models and train for single epoch
    for model in all_models_regression:
        parameters["model"]["architecture"] = model
        parameters["nested_training"]["testing"] = -5
        parameters["nested_training"]["validation"] = -5
        shutil.rmtree(outputDir)  # overwrite previous results
        Path(outputDir).mkdir(parents=True, exist_ok=True)
        TrainingManager(
            dataframe=training_data,
            outputDir=outputDir,
            parameters=parameters,
            device=device,
            reset_prev=True,
        )

    print("passed")


def test_train_classification_rad_3d(device):
    # read and initialize parameters for specific data dimension
    parameters = parseConfig(
        testingDir + "/config_classification.yaml", version_check_flag=False
    )
    parameters["modality"] = "rad"
    parameters["patch_size"] = patch_size["3D"]
    parameters["model"]["dimension"] = 3
    parameters["model"]["amp"] = True
    # read and parse csv
    training_data, parameters["headers"] = parseTrainingCSV(
        inputDir + "/train_3d_rad_classification.csv"
    )
    parameters = populate_header_in_parameters(parameters, parameters["headers"])
    parameters["model"]["num_channels"] = len(parameters["headers"]["channelHeaders"])
    # loop through selected models and train for single epoch
    for model in all_models_regression:
        parameters["model"]["architecture"] = model
        parameters["nested_training"]["testing"] = -5
        parameters["nested_training"]["validation"] = -5
        shutil.rmtree(outputDir)  # overwrite previous results
        Path(outputDir).mkdir(parents=True, exist_ok=True)
        TrainingManager(
            dataframe=training_data,
            outputDir=outputDir,
            parameters=parameters,
            device=device,
            reset_prev=True,
        )

    print("passed")


def test_inference_classification_rad_3d(device):
    # read and initialize parameters for specific data dimension
    parameters = parseConfig(
        testingDir + "/config_classification.yaml", version_check_flag=False
    )
    parameters["modality"] = "rad"
    parameters["patch_size"] = patch_size["3D"]
    parameters["model"]["dimension"] = 3
    parameters["model"]["amp"] = True
    # read and parse csv
    training_data, parameters["headers"] = parseTrainingCSV(
        inputDir + "/train_3d_rad_classification.csv"
    )
    parameters = populate_header_in_parameters(parameters, parameters["headers"])
    parameters["model"]["num_channels"] = len(parameters["headers"]["channelHeaders"])
    # loop through selected models and train for single epoch
    model = all_models_regression[0]
    parameters["model"]["architecture"] = model
    Path(outputDir).mkdir(parents=True, exist_ok=True)
    TrainingManager(
        dataframe=training_data,
        outputDir=outputDir,
        parameters=parameters,
        device=device,
        reset_prev=True,
    )
    parameters["output_dir"] = outputDir  # this is in inference mode
    InferenceManager(
        dataframe=training_data,
        outputDir=outputDir,
        parameters=parameters,
        device=device,
    )

    print("passed")


def test_inference_classification_with_logits_single_fold_rad_3d(device):
    # read and initialize parameters for specific data dimension
    parameters = parseConfig(
        testingDir + "/config_classification.yaml", version_check_flag=False
    )
    parameters["modality"] = "rad"
    parameters["patch_size"] = patch_size["3D"]
    parameters["model"]["dimension"] = 3
    parameters["model"]["amp"] = True
    parameters["model"]["final_layer"] = "logits"

    # read and parse csv
    training_data, parameters["headers"] = parseTrainingCSV(
        inputDir + "/train_3d_rad_classification.csv"
    )
    parameters = populate_header_in_parameters(parameters, parameters["headers"])
    parameters["model"]["num_channels"] = len(parameters["headers"]["channelHeaders"])
    # loop through selected models and train for single epoch
    model = all_models_regression[0]
    parameters["model"]["architecture"] = model
    Path(outputDir).mkdir(parents=True, exist_ok=True)
    TrainingManager(
        dataframe=training_data,
        outputDir=outputDir,
        parameters=parameters,
        device=device,
        reset_prev=True,
    )
    parameters["output_dir"] = outputDir  # this is in inference mode
    InferenceManager(
        dataframe=training_data,
        outputDir=outputDir,
        parameters=parameters,
        device=device,
    )

    print("passed")


def test_inference_classification_with_logits_multiple_folds_rad_3d(device):
    # read and initialize parameters for specific data dimension
    parameters = parseConfig(
        testingDir + "/config_classification.yaml", version_check_flag=False
    )
    parameters["modality"] = "rad"
    parameters["patch_size"] = patch_size["3D"]
    parameters["model"]["dimension"] = 3
    parameters["model"]["amp"] = True
    parameters["model"]["final_layer"] = "logits"
    # necessary for n-fold cross-validation inference
    parameters["nested_training"]["validation"] = 2

    # read and parse csv
    training_data, parameters["headers"] = parseTrainingCSV(
        inputDir + "/train_3d_rad_classification.csv"
    )
    parameters = populate_header_in_parameters(parameters, parameters["headers"])
    parameters["model"]["num_channels"] = len(parameters["headers"]["channelHeaders"])
    # loop through selected models and train for single epoch
    model = all_models_regression[0]
    parameters["model"]["architecture"] = model
    Path(outputDir).mkdir(parents=True, exist_ok=True)
    TrainingManager(
        dataframe=training_data,
        outputDir=outputDir,
        parameters=parameters,
        device=device,
        reset_prev=True,
    )
    parameters["output_dir"] = outputDir  # this is in inference mode
    InferenceManager(
        dataframe=training_data,
        outputDir=outputDir,
        parameters=parameters,
        device=device,
    )

    print("passed")


def test_scheduler_classification_rad_2d(device):
    # read and initialize parameters for specific data dimension
    parameters = parseConfig(
        testingDir + "/config_classification.yaml", version_check_flag=False
    )
    parameters["modality"] = "rad"
    parameters["patch_size"] = patch_size["2D"]
    parameters["model"]["dimension"] = 2
    parameters["model"]["amp"] = True
    # read and parse csv
    training_data, parameters["headers"] = parseTrainingCSV(
        inputDir + "/train_2d_rad_classification.csv"
    )
    parameters = populate_header_in_parameters(parameters, parameters["headers"])
    parameters["model"]["num_channels"] = 3
    parameters["model"]["architecture"] = "densenet121"
    # loop through selected models and train for single epoch
    for scheduler in global_schedulers_dict:
        parameters["scheduler"] = {}
        parameters["scheduler"]["type"] = scheduler
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
            reset_prev=True,
        )

    shutil.rmtree(outputDir)
    print("passed")


def test_optimizer_classification_rad_2d(device):
    # read and initialize parameters for specific data dimension
    parameters = parseConfig(
        testingDir + "/config_classification.yaml", version_check_flag=False
    )
    parameters["modality"] = "rad"
    parameters["patch_size"] = patch_size["2D"]
    parameters["model"]["dimension"] = 2
    parameters["model"]["amp"] = True
    # read and parse csv
    training_data, parameters["headers"] = parseTrainingCSV(
        inputDir + "/train_2d_rad_classification.csv"
    )
    parameters = populate_header_in_parameters(parameters, parameters["headers"])
    parameters["model"]["num_channels"] = 3
    parameters["model"]["architecture"] = "densenet121"
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
            reset_prev=True,
        )

    shutil.rmtree(outputDir)
    print("passed")


def test_clip_train_classification_rad_3d(device):
    # read and initialize parameters for specific data dimension
    parameters = parseConfig(
        testingDir + "/config_classification.yaml", version_check_flag=False
    )
    parameters["modality"] = "rad"
    parameters["patch_size"] = patch_size["3D"]
    parameters["model"]["dimension"] = 3
    parameters["model"]["amp"] = True
    # read and parse csv
    training_data, parameters["headers"] = parseTrainingCSV(
        inputDir + "/train_3d_rad_classification.csv"
    )
    parameters = populate_header_in_parameters(parameters, parameters["headers"])
    parameters["model"]["num_channels"] = len(parameters["headers"]["channelHeaders"])
    parameters["model"]["architecture"] = "vgg16"
    # loop through selected models and train for single epoch
    for clip_mode in all_clip_modes:
        parameters["clip_mode"] = clip_mode
        parameters["nested_training"]["testing"] = -5
        parameters["nested_training"]["validation"] = -5
        # shutil.rmtree(outputDir)  # overwrite previous results
        Path(outputDir).mkdir(parents=True, exist_ok=True)
        TrainingManager(
            dataframe=training_data,
            outputDir=outputDir,
            parameters=parameters,
            device=device,
            reset_prev=True,
        )
    shutil.rmtree(outputDir)  # overwrite previous results
    print("passed")


def test_normtype_train_segmentation_rad_3d(device):
    # read and initialize parameters for specific data dimension
    print("Starting 3D Rad segmentation tests for normtype")
    # read and parse csv
    # read and initialize parameters for specific data dimension
    parameters = parseConfig(
        testingDir + "/config_segmentation.yaml", version_check_flag=False
    )
    training_data, parameters["headers"] = parseTrainingCSV(
        inputDir + "/train_3d_rad_segmentation.csv"
    )
    parameters = populate_header_in_parameters(parameters, parameters["headers"])
    parameters["patch_size"] = patch_size["3D"]
    parameters["model"]["dimension"] = 3
    parameters["model"]["class_list"] = [0, 1]
    parameters["model"]["amp"] = True
    parameters["in_memory"] = True
    parameters["model"]["num_channels"] = len(parameters["headers"]["channelHeaders"])
    # loop through selected models and train for single epoch
    for norm in ["batch", "instance"]:
        for model in ["resunet", "unet", "fcn"]:
            parameters["model"]["architecture"] = model
            parameters["model"]["norm_type"] = norm
            parameters["nested_training"]["testing"] = -5
            parameters["nested_training"]["validation"] = -5
            Path(outputDir).mkdir(parents=True, exist_ok=True)
            TrainingManager(
                dataframe=training_data,
                outputDir=outputDir,
                parameters=parameters,
                device=device,
                reset_prev=True,
            )
            shutil.rmtree(outputDir)  # overwrite previous results
        print("passed")


def test_metrics_segmentation_rad_2d(device):
    print("Starting 2D Rad segmentation tests for metrics")
    # read and parse csv
    parameters = parseConfig(
        testingDir + "/config_segmentation.yaml", version_check_flag=False
    )
    training_data, parameters["headers"] = parseTrainingCSV(
        inputDir + "/train_2d_rad_segmentation.csv"
    )
    parameters = populate_header_in_parameters(parameters, parameters["headers"])
    parameters["patch_size"] = patch_size["2D"]
    parameters["model"]["dimension"] = 2
    parameters["model"]["class_list"] = [0, 255]
    parameters["model"]["amp"] = True
    parameters["model"]["num_channels"] = 3
    parameters["metrics"] = ["dice", "hausdorff", "hausdorff95"]
    parameters["model"]["architecture"] = "resunet"
    Path(outputDir).mkdir(parents=True, exist_ok=True)
    TrainingManager(
        dataframe=training_data,
        outputDir=outputDir,
        parameters=parameters,
        device=device,
        reset_prev=True,
    )
    shutil.rmtree(outputDir)  # overwrite previous results

    print("passed")


def test_metrics_regression_rad_2d(device):
    print("Starting 2D Rad regression tests for metrics")
    # read and parse csv
    parameters = parseConfig(
        testingDir + "/config_regression.yaml", version_check_flag=False
    )
    training_data, parameters["headers"] = parseTrainingCSV(
        inputDir + "/train_2d_rad_regression.csv"
    )
    parameters = populate_header_in_parameters(parameters, parameters["headers"])
    parameters["patch_size"] = patch_size["2D"]
    parameters["model"]["dimension"] = 2
    parameters["model"]["class_list"] = [0, 255]
    parameters["model"]["norm_type"] = "instance"
    parameters["model"]["amp"] = False
    parameters["model"]["num_channels"] = 3
    parameters["metrics"] = {}
    parameters["metrics"]["mse"] = {}
    parameters["metrics"]["accuracy"] = {}
    parameters["metrics"]["accuracy"]["threshold"] = 0.5
    parameters["model"]["architecture"] = "vgg11"
    Path(outputDir).mkdir(parents=True, exist_ok=True)
    TrainingManager(
        dataframe=training_data,
        outputDir=outputDir,
        parameters=parameters,
        device=device,
        reset_prev=True,
    )
    shutil.rmtree(outputDir)  # overwrite previous results

    print("passed")


def test_losses_segmentation_rad_2d(device):
    print("Starting 2D Rad segmentation tests for losses")
    # read and parse csv
    parameters = parseConfig(
        testingDir + "/config_segmentation.yaml", version_check_flag=False
    )
    training_data, parameters["headers"] = parseTrainingCSV(
        inputDir + "/train_2d_rad_segmentation.csv"
    )
    parameters = populate_header_in_parameters(parameters, parameters["headers"])
    parameters["patch_size"] = patch_size["2D"]
    parameters["model"]["dimension"] = 2
    parameters["model"]["class_list"] = [0, 255]
    # disabling amp because some losses do not support Half, yet
    parameters["model"]["amp"] = False
    parameters["model"]["num_channels"] = 3
    parameters["model"]["architecture"] = "resunet"
    parameters["metrics"] = ["dice"]
    # loop through selected models and train for single epoch
    for loss_type in ["dc", "dc_log", "dcce", "dcce_logits", "tversky"]:
        parameters["loss_function"] = loss_type
        parameters["nested_training"]["testing"] = -5
        parameters["nested_training"]["validation"] = -5
        Path(outputDir).mkdir(parents=True, exist_ok=True)
        TrainingManager(
            dataframe=training_data,
            outputDir=outputDir,
            parameters=parameters,
            device=device,
            reset_prev=True,
        )
        shutil.rmtree(outputDir)  # overwrite previous results
    print("passed")


def test_config_read():
    print("Starting testing reading configuration")
    # read and parse csv
    file_config_temp = os.path.join(testingDir, "config_segmentation_temp.yaml")
    # if found in previous run, discard.
    if os.path.exists(file_config_temp):
        os.remove(file_config_temp)

    parameters = parseConfig(
        os.path.abspath(baseConfigDir + "/config_all_options.yaml"),
        version_check_flag=False,
    )
    parameters["data_preprocessing"]["resize"] = [128, 128]

    with open(file_config_temp, "w") as file:
        yaml.dump(parameters, file)

    parameters = parseConfig(file_config_temp, version_check=True)

    training_data, parameters["headers"] = parseTrainingCSV(
        inputDir + "/train_2d_rad_segmentation.csv"
    )
    if not parameters:
        sys.exit(1)
    data_loader = ImagesFromDataFrame(training_data, parameters, True)
    if not data_loader:
        sys.exit(1)
    print("passed")


def test_cli_function_preprocess():
    print("Starting testing cli function preprocess")
    file_config = os.path.join(testingDir, "config_segmentation.yaml")
    file_config_temp = os.path.join(testingDir, "config_segmentation_temp.yaml")
    # if found in previous run, discard.
    if os.path.exists(file_config_temp):
        os.remove(file_config_temp)
        parameter_pickle_file = os.path.join(outputDir, "parameters.pkl")
        if os.path.exists(parameter_pickle_file):
            os.remove(parameter_pickle_file)
    file_data = os.path.join(inputDir, "train_2d_rad_segmentation.csv")

    parameters = parseConfig(file_config)
    parameters["patch_size"] = patch_size["2D"]
    parameters["model"]["dimension"] = 2
    parameters["model"]["class_list"] = "[0, 255||125]"
    # disabling amp because some losses do not support Half, yet
    parameters["model"]["amp"] = False
    parameters["model"]["num_channels"] = 3
    parameters["model"]["architecture"] = "unet"
    parameters["metrics"] = ["dice"]
    parameters["patch_sampler"] = "label"
    parameters["weighted_loss"] = True
    parameters["save_output"] = True
    parameters["data_preprocessing"]["to_canonical"] = None

    # store this separately for preprocess testing
    with open(file_config_temp, "w") as outfile:
        yaml.dump(parameters, outfile, default_flow_style=False)

    preprocess_and_save(file_data, file_config_temp, outputDir)
    training_data, parameters["headers"] = parseTrainingCSV(
        outputDir + "/data_processed.csv"
    )

    # check that the length of training data is what we expect
    assert len(training_data) == 10, "Number of rows in dataframe is not 10"

    shutil.rmtree(outputDir)  # overwrite previous results
    print("passed")


def test_cli_function_mainrun(device):
    print("Starting testing cli function main_run")
    file_config_temp = os.path.join(testingDir, "config_segmentation_temp.yaml")
    # if preprocess wasn't run, this file should not be present
    if not os.path.exists(file_config_temp):
        file_config_temp = os.path.join(testingDir, "config_segmentation.yaml")

    file_data = os.path.join(inputDir, "train_2d_rad_segmentation.csv")

    main_run(file_data, file_config_temp, outputDir, True, device, True)
    shutil.rmtree(outputDir)  # overwrite previous results
    print("passed")


def test_dataloader_construction_train_segmentation_3d(device):
    print("Starting 3D Rad segmentation tests")
    # read and parse csv
    # read and initialize parameters for specific data dimension
    parameters = parseConfig(
        testingDir + "/config_segmentation.yaml", version_check_flag=False
    )
    params_all_preprocessing_and_augs = parseConfig(
        testingDir + "/../samples/config_all_options.yaml"
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
    parameters = populate_header_in_parameters(parameters, parameters["headers"])
    parameters["patch_size"] = patch_size["3D"]
    parameters["model"]["dimension"] = 3
    parameters["model"]["class_list"] = [0, 1]
    parameters["model"]["amp"] = True
    parameters["model"]["num_channels"] = len(parameters["headers"]["channelHeaders"])
    parameters["model"]["architecture"] = "unet"
    parameters["weighted_loss"] = False
    # loop through selected models and train for single epoch
    Path(outputDir).mkdir(parents=True, exist_ok=True)
    TrainingManager(
        dataframe=training_data,
        outputDir=outputDir,
        parameters=parameters,
        device=device,
        reset_prev=True,
    )
    shutil.rmtree(outputDir)  # overwrite previous results
    print("passed")


def test_preprocess_functions():
    print("Starting testing preprocessing functions")
    # initialize an input which has values between [-1,1]
    # checking tensor with last dimension of size 1
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

    input_image = sitk.GetImageFromArray(input_tensor[0].numpy())
    img_resized = resize_image(
        input_image,
        [128, 128, 3],
    )
    temp_array = sitk.GetArrayFromImage(img_resized)
    assert temp_array.shape == (3, 128, 128), "Resampling should work"

    input_tensor = torch.rand(1, 256, 256, 256)
    cropper = global_preprocessing_dict["crop_external_zero_planes"](
        patch_size=[128, 128, 128]
    )
    input_transformed = cropper(input_tensor)

    cropper = global_preprocessing_dict["crop"]([64, 64, 64])
    input_transformed = cropper(input_tensor)
    assert input_transformed.shape == (1, 128, 128, 128), "Resampling should work"

    cropper = global_preprocessing_dict["centercrop"]([128, 128, 128])
    input_transformed = cropper(input_tensor)
    assert input_transformed.shape == (1, 128, 128, 128), "Resampling should work"

    print("passed")


def test_augmentation_functions():
    print("Starting testing augmentation functions")
    params_all_preprocessing_and_augs = parseConfig(
        testingDir + "/../samples/config_all_options.yaml"
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


def test_checkpointing_segmentation_rad_2d(device):
    print("Starting 2D Rad segmentation tests for metrics")
    # read and parse csv
    parameters = parseConfig(
        testingDir + "/config_segmentation.yaml", version_check_flag=False
    )
    training_data, parameters["headers"] = parseTrainingCSV(
        inputDir + "/train_2d_rad_segmentation.csv"
    )
    parameters = populate_header_in_parameters(parameters, parameters["headers"])
    parameters["patch_size"] = patch_size["2D"]
    parameters["num_epochs"] = 1
    parameters["nested_training"]["testing"] = 1
    parameters["model"]["dimension"] = 2
    parameters["model"]["class_list"] = [0, 255]
    parameters["model"]["amp"] = True
    parameters["model"]["num_channels"] = 3
    parameters["metrics"] = ["dice", "hausdorff", "hausdorff95"]
    parameters["model"]["architecture"] = "unet"
    Path(outputDir).mkdir(parents=True, exist_ok=True)
    TrainingManager(
        dataframe=training_data,
        outputDir=outputDir,
        parameters=parameters,
        device=device,
        reset_prev=True,
    )
    parameters["num_epochs"] = 2
    parameters["nested_training"]["validation"] = -2
    parameters["nested_training"]["testing"] = 1
    TrainingManager(
        dataframe=training_data,
        outputDir=outputDir,
        parameters=parameters,
        device=device,
        reset_prev=False,
    )
    shutil.rmtree(outputDir)  # overwrite previous results

    print("passed")


def test_model_patch_divisibility():

    parameters = parseConfig(
        testingDir + "/config_segmentation.yaml", version_check_flag=False
    )
    training_data, parameters["headers"] = parseTrainingCSV(
        inputDir + "/train_2d_rad_segmentation.csv"
    )
    parameters["model"]["architecture"] = "unet"
    parameters["patch_size"] = [127, 127, 1]
    parameters = populate_header_in_parameters(parameters, parameters["headers"])
    parameters["num_epochs"] = 1
    parameters["nested_training"]["testing"] = 1
    parameters["model"]["dimension"] = 2
    parameters["model"]["class_list"] = [0, 255]
    parameters["model"]["amp"] = True
    parameters["model"]["num_channels"] = 3
    parameters["metrics"] = ["dice", "hausdorff", "hausdorff95"]

    # this assertion should fail
    with pytest.raises(Exception) as e_info:
        global_models_dict[parameters["model"]["architecture"]](parameters=parameters)

    parameters["model"]["architecture"] = "uinc"
    parameters["model"]["base_filters"] = 11

    # this assertion should fail
    with pytest.raises(Exception) as e_info:
        global_models_dict[parameters["model"]["architecture"]](parameters=parameters)

    print("passed")


def test_one_hot_logic():

    random_array = np.random.randint(5, size=(20, 20, 20))
    img = sitk.GetImageFromArray(random_array)
    img_array = sitk.GetArrayFromImage(img)
    img_tensor = torch.from_numpy(img_array).to(torch.float16)
    img_tensor = img_tensor.unsqueeze(0)
    img_tensor = img_tensor.unsqueeze(0)

    class_list = [*range(0, np.max(random_array) + 1)]
    img_tensor_oh = one_hot(img_tensor, class_list)
    img_tensor_oh_rev_array = reverse_one_hot(img_tensor_oh[0], class_list)
    comparison = random_array == img_tensor_oh_rev_array
    assert comparison.all(), "Arrays are not equal"

    class_list = [0, "1||2||3", 4]
    img_tensor_oh = one_hot(img_tensor, class_list)
    img_tensor_oh_rev_array = reverse_one_hot(img_tensor_oh[0], class_list)
    comparison = (random_array == 0) == (img_tensor_oh_rev_array == 0)
    assert comparison.all(), "Arrays at '0' are not equal"

    random_array_sp = (random_array == 1) + (random_array == 2) + (random_array == 3)
    img_tensor_oh_rev_array_sp = img_tensor_oh_rev_array == 1
    img_tensor_oh_rev_array_sp[random_array == 4] = False
    comparison = random_array_sp == img_tensor_oh_rev_array_sp
    assert comparison.all(), "Special arrays are not equal"

    # check for '4'
    img_tensor_oh_rev_array_sp = img_tensor_oh_rev_array == 1
    for i in [1, 2, 3]:
        img_tensor_oh_rev_array_sp[random_array == i] = False
    comparison = (random_array == 4) == img_tensor_oh_rev_array_sp
    assert comparison.all(), "Arrays at '4' are not equal"
    print("passed")

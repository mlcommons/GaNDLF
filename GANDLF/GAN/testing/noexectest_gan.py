from pathlib import Path
import gdown, zipfile, os, csv, random, copy, shutil, yaml, torch, pytest

from GANDLF.cli import (
    patch_extraction,
)
from GANDLF.GAN.parseConfigGAN import parseConfigGAN
from GANDLF.GAN.training_manager_gan import TrainingManagerGAN
from GANDLF.GAN.inference_manager_gan import InferenceManagerGAN
from GANDLF.utils import (
    populate_header_in_parameters,
    parseTrainingCSV,
    writeTrainingCSV,
)

device = "cpu"

all_models_generation = [
    "dcgan",
]

all_clip_modes = ["norm", "value", "agc"]
all_norm_types = ["batch", "instance"]

all_model_type = ["torch", "openvino"]

patch_size = {"2D": [128, 128, 1], "3D": [48, 48, 48]}

testingDir = Path(__file__).parent.absolute().__str__()
baseConfigDir = os.path.join(testingDir, os.pardir, "samples")
inputDir = os.path.join(testingDir, "data")
outputDir = os.path.join(testingDir, "data_output")
Path(outputDir).mkdir(parents=True, exist_ok=True)
gandlfRootDir = Path(__file__).parent.parent.absolute().__str__()


def test_generic_download_data():
    print("00: Downloading the sample data")
    urlToDownload = (
        "https://drive.google.com/uc?id=1c4Yrv-jnK6Tk7Ne1HmMTChv-4nYk43NT"
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
            output = os.path.join(testingDir, "gandlf_unit_test_data.tgz")
            gdown.download(urlToDownload, output, quiet=False, verify=True)
            with zipfile.ZipFile(output, "r") as zip_ref:
                zip_ref.extractall(testingDir)
            os.remove(output)
            break
    sanitize_outputDir()

    print("passed")


def test_generic_constructTrainingCSV():
    print("01: Constructing training CSVs")
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
        elif "2d_rad_generation" in application_data:
            channelsID = "png"
            labelID = None
        elif "3d_rad_generation" in application_data:
            channelsID = "nii.gz"
            labelID = None
        elif "2d_histo_generation" in application_data:
            channelsID = "image"
            labelID = "mask"
        elif "dataset_1000" in application_data:
            channelsID = "jpg"
            labelID = None
        # else:
        #     continue
        outputFile = inputDir + "/train_" + application_data + ".csv"
        outputFile_rel = (
            inputDir + "/train_" + application_data + "_relative.csv"
        )
        # Test with various combinations of relative/absolute paths
        # Absolute input/output

        writeTrainingCSV(
            currentApplicationDir,
            channelsID,
            labelID,
            outputFile_rel,
            relativizePathsToOutput=True,
        )
        writeTrainingCSV(
            currentApplicationDir,
            channelsID,
            labelID,
            outputFile,
            relativizePathsToOutput=False,
        )
        if "generation" or "1000" in application_data:
            continue
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
            inputDir + "/train_" + application_data_regression + ".csv",
            "w",
            newline="",
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


# # these are helper functions to be used in other tests
def sanitize_outputDir():
    print("02_1: Sanitizing outputDir")
    if os.path.isdir(outputDir):
        shutil.rmtree(outputDir)  # overwrite previous results
    Path(outputDir).mkdir(parents=True, exist_ok=True)


def write_temp_config_path(parameters_to_write):
    print("02_2: Creating path for temporary config file")
    temp_config_path = os.path.join(outputDir, "config_temp.yaml")
    # if found in previous run, discard.
    if os.path.exists(temp_config_path):
        os.remove(temp_config_path)
    if parameters_to_write is not None:
        with open(temp_config_path, "w") as file:
            yaml.dump(parameters_to_write, file)
    return temp_config_path


# # these are helper functions to be used in other tests


def test_train_generation_rad_2d(device):
    print("03: Starting 2D Rad segmentation tests")
    # read and parse csv
    parameters = parseConfigGAN(
        testingDir + "/config_generation.yaml", version_check_flag=False
    )
    training_data, parameters["headers"] = parseTrainingCSV(
        inputDir + "/train_2d_rad_generation.csv"
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
    parameters = populate_header_in_parameters(
        parameters, parameters["headers"]
    )
    # read and initialize parameters for specific data dimension
    for model in all_models_generation:
        parameters["model"]["architecture"] = model
        parameters["nested_training"]["testing"] = -5
        parameters["nested_training"]["validation"] = -5
        sanitize_outputDir()
        TrainingManagerGAN(
            dataframe=training_data,
            outputDir=outputDir,
            parameters=parameters,
            device=device,
            resume=False,
            reset=True,
        )

    sanitize_outputDir()

    print("passed")


def test_train_generation_rad_3d(device):
    print("05: Starting 3D Rad segmentation tests")
    # read and parse csv
    # read and initialize parameters for specific data dimension
    parameters = parseConfigGAN(
        testingDir + "/config_generation.yaml", version_check_flag=False
    )
    training_data, parameters["headers"] = parseTrainingCSV(
        inputDir + "/train_3d_rad_generation.csv"
    )
    parameters["modality"] = "rad"
    parameters["patch_size"] = patch_size["3D"]
    parameters["model"]["dimension"] = 3
    parameters["model"]["class_list"] = [0, 1]
    parameters["model"]["final_layer"] = "softmax"
    parameters["inference_config"]["save_format"] = "nii.gz"
    parameters["model"]["amp"] = True
    parameters["in_memory"] = True
    if "fid" in parameters["metrics"]:
        parameters["metrics"].pop("fid")
        # fid is not supported for 3D data
    parameters["model"]["num_channels"] = len(
        parameters["headers"]["channelHeaders"]
    )
    parameters["save_output"] = True
    parameters["model"]["onnx_export"] = False
    parameters["model"]["print_summary"] = False
    parameters = populate_header_in_parameters(
        parameters, parameters["headers"]
    )
    # loop through selected models and train for single epoch
    for model in all_models_generation:
        parameters["model"]["architecture"] = model
        parameters["nested_training"]["testing"] = -5
        parameters["nested_training"]["validation"] = -5
        sanitize_outputDir()
        TrainingManagerGAN(
            dataframe=training_data,
            outputDir=outputDir,
            parameters=parameters,
            device=device,
            resume=False,
            reset=True,
        )

    sanitize_outputDir()

    print("passed")


def test_train_inference_optimize_generation_rad_2d(device):
    print("14: Starting 2D Rad segmentation tests for optimization")
    # read and parse csv
    parameters = parseConfigGAN(
        testingDir + "/config_generation.yaml", version_check_flag=False
    )
    training_data, parameters["headers"] = parseTrainingCSV(
        inputDir + "/train_2d_rad_generation.csv"
    )
    parameters["patch_size"] = patch_size["2D"]
    parameters["modality"] = "rad"
    parameters["model"]["dimension"] = 2
    parameters["model"]["class_list"] = [0, 255]
    parameters["model"]["amp"] = True
    parameters["save_output"] = True
    parameters["model"]["num_channels"] = 3
    parameters["metrics"] = ["ssim", "lpips"]
    parameters["model"]["architecture"] = "dcgan"
    parameters["model"]["onnx_export"] = True
    parameters["model"]["print_summary"] = True
    parameters = populate_header_in_parameters(
        parameters, parameters["headers"]
    )
    sanitize_outputDir()
    TrainingManagerGAN(
        dataframe=training_data,
        outputDir=outputDir,
        parameters=parameters,
        device=device,
        resume=False,
        reset=True,
    )

    ## testing inference
    # for model_type in all_model_type:
    parameters["model"]["type"] = "torch"
    parameters["output_dir"] = outputDir  # this is in inference mode
    InferenceManagerGAN(
        dataframe=training_data,
        modelDir=outputDir,
        parameters=parameters,
        device=device,
    )

    sanitize_outputDir()

    print("passed")


def test_train_inference_generation_histology_2d(device):
    print("34: Starting histology train/inference segmentation tests")
    # overwrite previous results
    sanitize_outputDir()
    output_dir_patches = os.path.join(outputDir, "histo_patches")
    if os.path.isdir(output_dir_patches):
        shutil.rmtree(output_dir_patches)
    Path(output_dir_patches).mkdir(parents=True, exist_ok=True)
    output_dir_patches_output = os.path.join(
        output_dir_patches, "histo_patches_output"
    )
    Path(output_dir_patches_output).mkdir(parents=True, exist_ok=True)

    parameters_patch = {}
    # extracting minimal number of patches to ensure that the test does not take too long
    parameters_patch["num_patches"] = 10
    parameters_patch["read_type"] = "sequential"
    # define patches to be extracted in terms of microns
    parameters_patch["patch_size"] = ["1000m", "1000m"]

    file_config_temp = write_temp_config_path(parameters_patch)

    patch_extraction(
        inputDir + "/train_2d_histo_generation.csv",
        output_dir_patches_output,
        file_config_temp,
    )

    file_for_Training = os.path.join(
        output_dir_patches_output, "opm_train.csv"
    )
    # read and parse csv
    parameters = parseConfigGAN(
        testingDir + "/config_generation.yaml", version_check_flag=False
    )

    training_data, parameters["headers"] = parseTrainingCSV(file_for_Training)
    parameters["patch_size"] = patch_size["2D"]
    parameters["modality"] = "histo"
    parameters["model"]["dimension"] = 2
    parameters["model"]["class_list"] = [0, 255]
    parameters["model"]["amp"] = True
    parameters["model"]["num_channels"] = 3
    parameters = populate_header_in_parameters(
        parameters, parameters["headers"]
    )
    parameters["model"]["architecture"] = "dcgan"
    parameters["nested_training"]["testing"] = 1
    parameters["nested_training"]["validation"] = -2
    parameters["model"]["onnx_export"] = True  # not supported currently
    parameters["model"]["print_summary"] = True
    parameters["data_preprocessing"]["resize_image"] = [128, 128]
    modelDir = os.path.join(outputDir, "modelDir")
    Path(modelDir).mkdir(parents=True, exist_ok=True)
    TrainingManagerGAN(
        dataframe=training_data,
        outputDir=modelDir,
        parameters=parameters,
        device=device,
        resume=False,
        reset=True,
    )
    inference_data, parameters["headers"] = parseTrainingCSV(
        inputDir + "/train_2d_histo_generation.csv", train=False
    )
    inference_data.drop(index=inference_data.index[-1], axis=0, inplace=True)
    InferenceManagerGAN(
        dataframe=inference_data,
        modelDir=modelDir,
        parameters=parameters,
        device=device,
    )

    sanitize_outputDir()

    print("passed")

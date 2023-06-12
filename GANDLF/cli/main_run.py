import os, pickle
from pathlib import Path

from GANDLF.training_manager import TrainingManager, TrainingManager_split
from GANDLF.inference_manager import InferenceManager
from GANDLF.parseConfig import parseConfig
from GANDLF.utils import (
    populate_header_in_parameters,
    parseTrainingCSV,
    parseTestingCSV,
    set_determinism,
)


def main_run(
    data_csv, config_file, model_dir, train_mode, device, resume, reset, output_dir=None
):
    """
    Main function that runs the training and inference.

    Args:
        data_csv (str): The CSV file of the training data.
        config_file (str): The YAML file of the training configuration.
        model_dir (str): The model directory; for training, model is written out here, and for inference, trained model is expected here.
        train_mode (bool): Whether to train or infer.
        device (str): The device type.
        resume (bool): Whether the previous run will be resumed or not.
        reset (bool): Whether the previous run will be reset or not.
        output_dir (str): The output directory for the inference session.

    Returns:
        None
    """
    file_data_full = data_csv
    model_parameters = config_file
    device = device
    parameters = parseConfig(model_parameters)
    parameters["device_id"] = -1

    if train_mode:
        if resume:
            print(
                "Trying to resume training without changing any parameters from previous run.",
                flush=True,
            )
        parameters["output_dir"] = model_dir
        Path(parameters["output_dir"]).mkdir(parents=True, exist_ok=True)

    # if the output directory is not specified, then use the model directory even for the testing data
    # default behavior
    parameters["output_dir"] = parameters.get("output_dir", output_dir)
    if parameters["output_dir"] is None:
        parameters["output_dir"] = model_dir
    Path(parameters["output_dir"]).mkdir(parents=True, exist_ok=True)

    if "-1" in device:
        device = "cpu"

    # parse training CSV
    if "," in file_data_full:
        # training and validation pre-split
        data_full = None
        all_csvs = file_data_full.split(",")
        data_train, headers_train = parseTrainingCSV(all_csvs[0], train=train_mode)
        data_validation, headers_validation = parseTrainingCSV(
            all_csvs[1], train=train_mode
        )
        assert (
            headers_train == headers_validation
        ), "The training and validation CSVs do not have the same header information."

        # testing data is present
        data_testing = None
        headers_testing = headers_train
        if len(all_csvs) == 3:
            data_testing, headers_testing = parseTrainingCSV(
                all_csvs[2], train=train_mode
            )
        assert (
            headers_train == headers_testing
        ), "The training and testing CSVs do not have the same header information."

        parameters = populate_header_in_parameters(parameters, headers_train)
        # if we are here, it is assumed that the user wants to do training
        if train_mode:
            TrainingManager_split(
                dataframe_train=data_train,
                dataframe_validation=data_validation,
                dataframe_testing=data_testing,
                outputDir=parameters["output_dir"],
                parameters=parameters,
                device=device,
                resume=resume,
                reset=reset,
            )
    else:
        data_full, headers = parseTrainingCSV(file_data_full, train=train_mode)
        parameters = populate_header_in_parameters(parameters, headers)
        if train_mode:
            TrainingManager(
                dataframe=data_full,
                outputDir=parameters["output_dir"],
                parameters=parameters,
                device=device,
                resume=resume,
                reset=reset,
            )
        else:
            _, data_full, headers = parseTestingCSV(
                file_data_full, parameters["output_dir"]
            )
            InferenceManager(
                dataframe=data_full,
                modelDir=model_dir,
                outputDir=output_dir,
                parameters=parameters,
                device=device,
            )

import os, pickle
from pathlib import Path

from GANDLF.training_manager import TrainingManager, TrainingManager_split
from GANDLF.inference_manager import InferenceManager
from GANDLF.parseConfig import parseConfig
from GANDLF.utils import populate_header_in_parameters, parseTrainingCSV


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
    # in case the data being passed is already processed, check if the previous parameters exists,
    # and if it does, compare the two and if they are the same, ensure no preprocess is done.
    model_parameters_prev = os.path.join(os.path.dirname(model_dir), "parameters.pkl")
    if train_mode:
        if not (reset) or not (resume):
            print(
                "Trying to resume training without changing any parameters from previous run.",
                flush=True,
            )
            if os.path.exists(model_parameters_prev):
                parameters_prev = pickle.load(open(model_parameters_prev, "rb"))
                assert (
                    parameters == parameters_prev
                ), "The parameters are not the same as the ones stored in the previous run, please re-check."
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
            InferenceManager(
                dataframe=data_full,
                modelDir=model_dir,
                outputDir=output_dir,
                parameters=parameters,
                device=device,
            )

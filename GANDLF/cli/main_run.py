import os, pickle, sys
from pathlib import Path

from GANDLF.training_manager import TrainingManager, TrainingManager_split
from GANDLF.inference_manager import InferenceManager
from GANDLF.parseConfig import parseConfig
from GANDLF.utils import populate_header_in_parameters, parseTrainingCSV


def main_run(data_csv, config_file, output_dir, train_mode, device, resume, reset):
    """
    Main function that runs the training and inference.

    Args:
        data_csv (str): The CSV file of the training data.
        config_file (str): The YAML file of the training configuration.
        output_dir (str): The output directory.
        train_mode (bool): Whether to train or infer.
        device (str): The device type.
        resume (bool): Whether the previous run will be resumed or not.
        reset (bool): Whether the previous run will be reset or not.

    Raises:
        ValueError: Parameter check from previous run.
    """
    file_data_full = data_csv
    model_parameters = config_file
    device = device
    parameters = parseConfig(model_parameters)
    # in case the data being passed is already processed, check if the previous parameters exists,
    # and if it does, compare the two and if they are the same, ensure no preprocess is done.
    model_parameters_prev = os.path.join(os.path.dirname(output_dir), "parameters.pkl")
    if train_mode:
        if not (reset) or not (resume):
            print(
                "Trying to resume training without changing any parameters from previous run.",
                flush=True,
            )
            if os.path.exists(model_parameters_prev):
                parameters_prev = pickle.load(open(model_parameters_prev, "rb"))
                if parameters != parameters_prev:
                    raise ValueError(
                        "The parameters are not the same as the ones stored in the previous run, please re-check."
                    )

    parameters["output_dir"] = output_dir

    if "-1" in device:
        device = "cpu"

    if train_mode:  # train mode
        Path(parameters["output_dir"]).mkdir(parents=True, exist_ok=True)

    # parse training CSV
    if "," in file_data_full:
        # training and validation pre-split
        data_full = None
        both_csvs = file_data_full.split(",")
        data_train, headers_train = parseTrainingCSV(both_csvs[0], train=train_mode)
        data_validation, headers_validation = parseTrainingCSV(
            both_csvs[1], train=train_mode
        )

        if headers_train != headers_validation:
            sys.exit(
                "The training and validation CSVs do not have the same header information."
            )

        parameters = populate_header_in_parameters(parameters, headers_train)
        # if we are here, it is assumed that the user wants to do training
        if train_mode:
            TrainingManager_split(
                dataframe_train=data_train,
                dataframe_validation=data_validation,
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
                outputDir=parameters["output_dir"],
                parameters=parameters,
                device=device,
            )

import pandas as pd
import os, pickle, shutil
from pathlib import Path

from GANDLF.compute import training_loop
from GANDLF.utils import get_dataframe, split_data


def TrainingManager(
    dataframe: pd.DataFrame,
    outputDir: str,
    parameters: dict,
    device: str,
    resume: bool,
    reset: bool,
) -> None:
    """
    This is the training manager that ties all the training functionality together

    Args:
        dataframe (pandas.DataFrame): The full data from CSV.
        outputDir (str): The main output directory.
        parameters (dict): The parameters dictionary.
        device (str): The device to perform computations on.
        resume (bool): Whether the previous run will be resumed or not.
        reset (bool): Whether the previous run will be reset or not.
    """
    if reset:
        shutil.rmtree(outputDir)
        Path(outputDir).mkdir(parents=True, exist_ok=True)

    # save the current model configuration as a sanity check
    currentModelConfigPickle = os.path.join(outputDir, "parameters.pkl")
    if (not os.path.exists(currentModelConfigPickle)) or reset or resume:
        with open(currentModelConfigPickle, "wb") as handle:
            pickle.dump(parameters, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        if os.path.exists(currentModelConfigPickle):
            print(
                "Using previously saved parameter file",
                currentModelConfigPickle,
                flush=True,
            )
            parameters = pickle.load(open(currentModelConfigPickle, "rb"))

    dataframe_split = split_data(dataframe, parameters)

    last_indeces, _, _, _ = dataframe_split[-1]

    # check the last indeces to see if single fold training is requested
    singleFoldTesting = True if last_indeces[0] == 0 else False
    singleFoldValidation = True if last_indeces[1] == 0 else False

    for (
        testing_and_valid_indeces,
        trainingData,
        validationData,
        testingData,
    ) in dataframe_split:
        # the output of the current fold is only needed if multi-fold training is happening
        currentTestingOutputFolder = outputDir
        if not singleFoldTesting:
            currentTestingOutputFolder = os.path.join(
                outputDir, "testing_" + str(testing_and_valid_indeces[0])
            )
            Path(currentTestingOutputFolder).mkdir(parents=True, exist_ok=True)

        currentValidationOutputFolder = currentTestingOutputFolder
        if not singleFoldValidation:
            currentValidationOutputFolder = os.path.join(
                currentTestingOutputFolder, str(testing_and_valid_indeces[1])
            )
            Path(currentValidationOutputFolder).mkdir(parents=True, exist_ok=True)

        # initialize the dataframes and save them to disk
        data_dict = {
            "training": trainingData,
            "validation": validationData,
            "testing": testingData,
        }
        data_dict_files = {}
        for data_type, data in data_dict.items():
            data_dict_files[data_type] = None
            if data is not None:
                currentDataPickle = os.path.join(
                    currentValidationOutputFolder, "data_" + data_type + ".pkl"
                )
                data_dict_files[data_type] = currentDataPickle
                if (not os.path.exists(currentDataPickle)) or reset or resume:
                    data.to_pickle(currentDataPickle)
                    data.to_csv(currentDataPickle.replace(".pkl", ".csv"), index=False)
                else:
                    # read the data from the pickle if present
                    data_dict[data_type] = get_dataframe(currentDataPickle)

        # parallel_compute_command is an empty string, thus no parallel computing requested
        if not parameters["parallel_compute_command"]:
            training_loop(
                training_data=data_dict["training"],
                validation_data=data_dict["validation"],
                output_dir=currentValidationOutputFolder,
                device=device,
                params=parameters,
                testing_data=data_dict["testing"],
            )

        else:
            # call hpc command here
            parallel_compute_command_actual = parameters[
                "parallel_compute_command"
            ].replace("${outputDir}", currentValidationOutputFolder)

            assert (
                "python" in parallel_compute_command_actual
            ), "The 'parallel_compute_command_actual' needs to have the python from the virtual environment, which is usually '${GANDLF_dir}/venv/bin/python'"

            command = (
                parallel_compute_command_actual
                + " -m GANDLF.training_loop -train_loader_pickle "
                + data_dict_files["training"]
                + " -val_loader_pickle "
                + data_dict_files["validation"]
                + " -parameter_pickle "
                + currentModelConfigPickle
                + " -device "
                + str(device)
                + " -outputDir "
                + currentValidationOutputFolder
                + " -testing_loader_pickle "
                + data_dict_files["testing"]
            )

            print("Running command: ", command, flush=True)
            os.system(command, flush=True)


def TrainingManager_split(
    dataframe_train: pd.DataFrame,
    dataframe_validation: pd.DataFrame,
    dataframe_testing: pd.DataFrame,
    outputDir: str,
    parameters: dict,
    device: str,
    resume: bool,
    reset: bool,
):
    """
    This is the training manager that ties all the training functionality together

    Args:
        dataframe_train (pd.DataFrame): The training data from CSV.
        dataframe_validation (pd.DataFrame): The validation data from CSV.
        dataframe_testing (pd.DataFrame): The testing data from CSV.
        outputDir (str): The main output directory.
        parameters (dict): The parameters dictionary.
        device (str): The device to perform computations on.
        resume (bool): Whether the previous run will be resumed or not.
        reset (bool): Whether the previous run will be reset or not.
    """
    currentModelConfigPickle = os.path.join(outputDir, "parameters.pkl")
    if (not os.path.exists(currentModelConfigPickle)) or reset or resume:
        with open(currentModelConfigPickle, "wb") as handle:
            pickle.dump(parameters, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        if os.path.exists(currentModelConfigPickle):
            print(
                "Using previously saved parameter file",
                currentModelConfigPickle,
                flush=True,
            )
            parameters = pickle.load(open(currentModelConfigPickle, "rb"))

    training_loop(
        training_data=dataframe_train,
        validation_data=dataframe_validation,
        output_dir=outputDir,
        device=device,
        params=parameters,
        testing_data=dataframe_testing,
    )

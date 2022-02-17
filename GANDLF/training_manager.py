import pandas as pd
import os, sys, pickle, subprocess, shutil
from sklearn.model_selection import KFold
from pathlib import Path

# from GANDLF.data.ImagesFromDataFrame import ImagesFromDataFrame
from GANDLF.compute import training_loop


def TrainingManager(dataframe, outputDir, parameters, device, reset_prev):
    """
    This is the training manager that ties all the training functionality together

    dataframe = full data from CSV
    outputDir = the main output directory
    parameters = parameters read from YAML
    device = self-explanatory
    reset_prev = whether the previous run in the same output directory is used or not
    """
    if reset_prev:
        shutil.rmtree(outputDir)
        Path(outputDir).mkdir(parents=True, exist_ok=True)

    # save the current model configuration as a sanity check
    currentModelConfigPickle = os.path.join(outputDir, "parameters.pkl")
    if (not os.path.exists(currentModelConfigPickle)) or reset_prev:
        with open(currentModelConfigPickle, "wb") as handle:
            pickle.dump(parameters, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        print(
            "Using previously saved parameter file",
            currentModelConfigPickle,
            flush=True,
        )
        parameters = pickle.load(open(currentModelConfigPickle, "rb"))

    # initialize model type for processing: if not defined, default to Torch
    if not ("type" in parameters["model"]):
        parameters["model"]["type"] = "Torch"

    # check for single fold training
    singleFoldValidation = False
    singleFoldTesting = False
    noTestingData = False
    # if the user wants a single fold training
    if parameters["nested_training"]["testing"] < 0:
        parameters["nested_training"]["testing"] = abs(
            parameters["nested_training"]["testing"]
        )
        singleFoldTesting = True

    # if the user wants a single fold training
    if parameters["nested_training"]["validation"] < 0:
        parameters["nested_training"]["validation"] = abs(
            parameters["nested_training"]["validation"]
        )
        singleFoldValidation = True

    # this is the condition where testing data is not to be kept
    if parameters["nested_training"]["testing"] == 1:
        noTestingData = True
        singleFoldTesting = True
        # put 2 just so that the first for-loop does not fail
        parameters["nested_training"]["testing"] = 2

    # initialize the kfold structures
    kf_testing = KFold(n_splits=parameters["nested_training"]["testing"])
    kf_validation = KFold(n_splits=parameters["nested_training"]["validation"])

    currentTestingFold = 0

    # split across subjects
    subjectIDs_full = (
        dataframe[dataframe.columns[parameters["headers"]["subjectIDHeader"]]]
        .unique()
        .tolist()
    )

    # get the indeces for kfold splitting
    trainingData_full = dataframe

    # start the kFold train for testing
    for trainAndVal_index, testing_index in kf_testing.split(subjectIDs_full):

        # ensure the validation fold is initialized per-testing split
        currentValidationFold = 0

        trainingAndValidationData = pd.DataFrame()  # initialize the variable
        testingData = pd.DataFrame()  # initialize the variable
        # get the current training and testing data
        if noTestingData:
            # don't consider the split indeces for this case
            trainingAndValidationData = trainingData_full
            testingData = None
        else:
            # loop over all trainAndVal_index and construct new dataframe
            for subject_idx in trainAndVal_index:
                trainingAndValidationData = trainingAndValidationData.append(
                    trainingData_full[
                        trainingData_full[
                            trainingData_full.columns[
                                parameters["headers"]["subjectIDHeader"]
                            ]
                        ]
                        == subjectIDs_full[subject_idx]
                    ]
                )

            # loop over all testing_index and construct new dataframe
            for subject_idx in testing_index:
                testingData = testingData.append(
                    trainingData_full[
                        trainingData_full[
                            trainingData_full.columns[
                                parameters["headers"]["subjectIDHeader"]
                            ]
                        ]
                        == subjectIDs_full[subject_idx]
                    ]
                )

        # the output of the current fold is only needed if multi-fold training is happening
        if singleFoldTesting:
            currentOutputFolder = outputDir
        else:
            currentOutputFolder = os.path.join(
                outputDir, "testing_" + str(currentTestingFold)
            )
            Path(currentOutputFolder).mkdir(parents=True, exist_ok=True)

        # save the current training+validation and testing datasets
        if noTestingData:
            print(
                "WARNING: Testing data is empty, which will result in scientifically incorrect results; use at your own risk."
            )
            current_training_subject_indeces_full = subjectIDs_full
            currentTestingDataPickle = "None"
        else:
            currentTrainingAndValidationDataPickle = os.path.join(
                currentOutputFolder, "data_trainAndVal.pkl"
            )
            currentTestingDataPickle = os.path.join(
                currentOutputFolder, "data_testing.pkl"
            )

            if (not os.path.exists(currentTestingDataPickle)) or reset_prev:
                testingData.to_pickle(currentTestingDataPickle)
            if (
                not os.path.exists(currentTrainingAndValidationDataPickle)
            ) or reset_prev:
                trainingAndValidationData.to_pickle(
                    currentTrainingAndValidationDataPickle
                )

            current_training_subject_indeces_full = (
                trainingAndValidationData[
                    trainingAndValidationData.columns[
                        parameters["headers"]["subjectIDHeader"]
                    ]
                ]
                .unique()
                .tolist()
            )

        # start the kFold train for validation
        for train_index, val_index in kf_validation.split(
            current_training_subject_indeces_full
        ):

            # the output of the current fold is only needed if multi-fold training is happening
            if singleFoldValidation:
                currentValOutputFolder = currentOutputFolder
            else:
                currentValOutputFolder = os.path.join(
                    currentOutputFolder, str(currentValidationFold)
                )
                Path(currentValOutputFolder).mkdir(parents=True, exist_ok=True)

            trainingData = pd.DataFrame()  # initialize the variable
            validationData = pd.DataFrame()  # initialize the variable

            # loop over all train_index and construct new dataframe
            for subject_idx in train_index:
                trainingData = trainingData.append(
                    trainingData_full[
                        trainingData_full[
                            trainingData_full.columns[
                                parameters["headers"]["subjectIDHeader"]
                            ]
                        ]
                        == subjectIDs_full[subject_idx]
                    ]
                )

            # loop over all val_index and construct new dataframe
            for subject_idx in val_index:
                validationData = validationData.append(
                    trainingData_full[
                        trainingData_full[
                            trainingData_full.columns[
                                parameters["headers"]["subjectIDHeader"]
                            ]
                        ]
                        == subjectIDs_full[subject_idx]
                    ]
                )

            # # write parameters to pickle - this should not change for the different folds, so keeping is independent
            ## pickle/unpickle data
            # pickle the data
            currentTrainingDataPickle = os.path.join(
                currentValOutputFolder, "data_training.pkl"
            )
            currentValidationDataPickle = os.path.join(
                currentValOutputFolder, "data_validation.pkl"
            )
            if (not os.path.exists(currentTrainingDataPickle)) or reset_prev:
                trainingData.to_pickle(currentTrainingDataPickle)
            else:
                trainingData = pd.read_pickle(currentTrainingDataPickle)
            if (not os.path.exists(currentValidationDataPickle)) or reset_prev:
                validationData.to_pickle(currentValidationDataPickle)
            else:
                validationData = pd.read_pickle(currentValidationDataPickle)

            # parallel_compute_command is an empty string, thus no parallel computing requested
            if (not parameters["parallel_compute_command"]) or (singleFoldValidation):
                training_loop(
                    training_data=trainingData,
                    validation_data=validationData,
                    output_dir=currentValOutputFolder,
                    device=device,
                    params=parameters,
                    testing_data=testingData,
                )

            else:
                # call qsub here
                parallel_compute_command_actual = parameters[
                    "parallel_compute_command"
                ].replace("${outputDir}", currentValOutputFolder)

                if not ("python" in parallel_compute_command_actual):
                    sys.exit(
                        "The 'parallel_compute_command_actual' needs to have the python from the virtual environment, which is usually '${GANDLF_dir}/venv/bin/python'"
                    )

                command = (
                    parallel_compute_command_actual
                    + " -m GANDLF.training_loop -train_loader_pickle "
                    + currentTrainingDataPickle
                    + " -val_loader_pickle "
                    + currentValidationDataPickle
                    + " -parameter_pickle "
                    + currentModelConfigPickle
                    + " -device "
                    + str(device)
                    + " -outputDir "
                    + currentValOutputFolder
                    + " -testing_loader_pickle "
                    + currentTestingDataPickle
                )

                print(
                    "Submitting job for testing split "
                    + str(currentTestingFold)
                    + " and validation split "
                    + str(currentValidationFold)
                )
                subprocess.Popen(command, shell=True).wait()

            if singleFoldValidation:
                break
            currentValidationFold += 1  # go to next fold

        if singleFoldTesting:
            break
        currentTestingFold += 1  # go to next fold


def TrainingManager_split(
    dataframe_train, dataframe_validation, outputDir, parameters, device, reset_prev
):
    """
    This is the training manager that ties all the training functionality together

    dataframe_train = training data from CSV
    dataframe_validation = validation data from CSV
    outputDir = the main output directory
    parameters = parameters read from YAML
    device = self-explanatory
    reset_prev = whether the previous run in the same output directory is used or not
    """
    currentModelConfigPickle = os.path.join(outputDir, "parameters.pkl")
    if (not os.path.exists(currentModelConfigPickle)) or reset_prev:
        with open(currentModelConfigPickle, "wb") as handle:
            pickle.dump(parameters, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
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
        testing_data=None,
    )

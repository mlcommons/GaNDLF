from typing import List, Tuple
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold

from . import parseTrainingCSV, populate_header_in_parameters


def split_data(
    full_dataset: pd.DataFrame, parameters: dict
) -> List[Tuple[Tuple[int, int], pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """
    Split the data into training, validation, and testing sets.

    Args:
        full_dataset (pd.DataFrame): The full dataset to split.
        parameters (dict): The parameters to use for splitting the data, which should contain the "nested_training" key with relevant information.

    Returns:
        List[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]: A list of tuples, each containing the a tuple of the testing & validation split indeces, and training, validation, and testing sets.
    """
    assert (
        "nested_training" in parameters
    ), "`nested_training` key missing in parameters"
    # populate the headers
    if "headers" not in parameters:
        _, parameters["headers"] = parseTrainingCSV(full_dataset)

    parameters = (
        populate_header_in_parameters(parameters, parameters["headers"])
        if "problem_type" not in parameters
        else parameters
    )

    stratified_splitting = parameters["nested_training"].get("stratified")

    return_data = []

    # check for single fold training
    singleFoldValidation = False
    singleFoldTesting = False
    # if the user wants a single fold training
    testing_folds = parameters["nested_training"]["testing"]
    if testing_folds < 0:
        testing_folds = abs(testing_folds)
        singleFoldTesting = True

    # if the user wants a single fold training
    validation_folds = parameters["nested_training"]["validation"]
    if validation_folds < 0:
        validation_folds = abs(validation_folds)
        singleFoldValidation = True

    # this is the condition where testing data is not to be kept
    noTestingData = False
    if testing_folds == 1:
        noTestingData = True
        singleFoldTesting = True
        # put 2 just so that the first for-loop does not fail
        testing_folds = 2
        print(
            "WARNING: Cross-validation is set to run on a train/validation scheme without testing data. For a more rigorous evaluation and if you wish to tune hyperparameters, make sure to use nested cross-validation."
        )

    # get unique subject IDs
    subjectIDs_full = (
        full_dataset[full_dataset.columns[parameters["headers"]["subjectIDHeader"]]]
        .unique()
        .tolist()
    )

    all_subjects_are_unique = len(subjectIDs_full) == len(full_dataset.index)

    # checks for stratified splitting
    if stratified_splitting:
        # it can only be done for classification problems
        assert (
            parameters["problem_type"] == "classification"
        ), "Stratified splitting is only possible for classification problems."
        # it can only be done when all subjects are unique
        assert (
            all_subjects_are_unique
        ), "Stratified splitting is not possible when duplicate subjects IDs are present in the dataset."

    # get the targets for prediction for classification
    target_testing = False  # initialize this so that the downstream code does not fail - for KFold, this is shuffle
    if parameters["problem_type"] == "classification":
        target_testing = full_dataset.loc[
            :, full_dataset.columns[parameters["headers"]["predictionHeaders"]]
        ]
    target_validation = target_testing

    folding_type = KFold
    if stratified_splitting:
        folding_type = StratifiedKFold

    kf_testing = folding_type(n_splits=testing_folds)
    kf_validation = folding_type(n_splits=validation_folds)

    # start StratifiedKFold splitting
    currentTestingFold = 0
    if stratified_splitting:
        for trainAndVal_index, testing_index in kf_testing.split(
            full_dataset, target_testing
        ):
            # ensure the validation fold is initialized per-testing split
            currentValidationFold = 0

            trainingAndValidationData, testingData = (
                pd.DataFrame(),
                pd.DataFrame(),
            )  # initialize the variables
            # get the current training and testing data
            if noTestingData:
                # don't consider the split indeces for this case
                trainingAndValidationData = full_dataset
                # this should be None to ensure downstream code does not fail
                testingData = None
            else:
                trainingAndValidationData = full_dataset.loc[trainAndVal_index, :]
                trainingAndValidationData.reset_index(drop=True, inplace=True)
                testingData = full_dataset.loc[testing_index, :]
                # update the targets after the split
                target_validation = trainingAndValidationData.loc[
                    :, full_dataset.columns[parameters["headers"]["predictionHeaders"]]
                ]

            for train_index, val_index in kf_validation.split(
                trainingAndValidationData, target_validation
            ):
                # get the current training and validation data
                trainingData = trainingAndValidationData.loc[train_index, :]
                validationData = trainingAndValidationData.loc[val_index, :]
                return_data.append(
                    (
                        (currentTestingFold, currentValidationFold),
                        trainingData,
                        validationData,
                        testingData,
                    )
                )
                currentValidationFold += 1  # increment the validation fold
                if singleFoldValidation:
                    break

            currentTestingFold += 1  # increment the testing fold
            if singleFoldTesting:
                break
    else:
        # start the kFold train for testing
        for trainAndVal_index, testing_index in kf_testing.split(subjectIDs_full):
            # ensure the validation fold is initialized per-testing split
            currentValidationFold = 0

            trainingAndValidationData, testingData = (
                pd.DataFrame(),
                pd.DataFrame(),
            )  # initialize the variables
            # get the current training and testing data
            if noTestingData:
                # don't consider the split indeces for this case
                trainingAndValidationData = full_dataset
                # this should be None to ensure downstream code does not fail
                testingData = None
            else:
                # loop over all trainAndVal_index and construct new dataframe
                for subject_idx in trainAndVal_index:
                    trainingAndValidationData = trainingAndValidationData._append(
                        full_dataset[
                            full_dataset[
                                full_dataset.columns[
                                    parameters["headers"]["subjectIDHeader"]
                                ]
                            ]
                            == subjectIDs_full[subject_idx]
                        ]
                    )

                # loop over all testing_index and construct new dataframe
                for subject_idx in testing_index:
                    testingData = testingData._append(
                        full_dataset[
                            full_dataset[
                                full_dataset.columns[
                                    parameters["headers"]["subjectIDHeader"]
                                ]
                            ]
                            == subjectIDs_full[subject_idx]
                        ]
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
                trainingData = pd.DataFrame()  # initialize the variable
                validationData = pd.DataFrame()  # initialize the variable

                # loop over all train_index and construct new dataframe
                for subject_idx in train_index:
                    trainingData = trainingData._append(
                        full_dataset[
                            full_dataset[
                                full_dataset.columns[
                                    parameters["headers"]["subjectIDHeader"]
                                ]
                            ]
                            == subjectIDs_full[subject_idx]
                        ]
                    )

                # loop over all val_index and construct new dataframe
                for subject_idx in val_index:
                    validationData = validationData._append(
                        full_dataset[
                            full_dataset[
                                full_dataset.columns[
                                    parameters["headers"]["subjectIDHeader"]
                                ]
                            ]
                            == subjectIDs_full[subject_idx]
                        ]
                    )

                return_data.append(
                    (
                        (currentTestingFold, currentValidationFold),
                        trainingData,
                        validationData,
                        testingData,
                    )
                )

                currentValidationFold += 1  # go to next fold
                if singleFoldValidation:
                    break

            currentTestingFold += 1  # go to next fold
            if singleFoldTesting:
                break

    return return_data

import os
import pathlib
import sys
from typing import Optional, Tuple, Union

import pandas as pd

from .handle_collisions import handle_collisions


def writeTrainingCSV(
    inputDir: str,
    channelsID: str,
    labelID: str,
    outputFile: str,
    relativizePathsToOutput: Optional[bool] = False,
) -> None:
    """
    This function writes a CSV file containing the paths to the training data.

    Args:
        inputDir (str): The input directory containing all the training data.
        channelsID (str): The channel IDs.
        labelID (str): The label ID.
        outputFile (str): The output CSV file.
        relativizePathsToOutput (Optional[bool], optional): Whether to relativize the paths to the output file. Defaults to False.
    """
    channelsID_list = channelsID.split(",")  # split into list

    outputToWrite = "SubjectID,"
    outputToWrite += (
        ",".join(["Channel_" + str(i) for i, n in enumerate(channelsID_list)]) + ","
    )
    if labelID is not None:
        outputToWrite += "Label"
    outputToWrite += "\n"

    # iterate over all subject directories
    for dirs in os.listdir(inputDir):
        currentSubjectDir = os.path.join(inputDir, dirs)
        # only consider sub-folders
        if os.path.isdir(currentSubjectDir):
            # get all files in each directory
            filesInDir = os.listdir(currentSubjectDir)
            maskFile = ""
            allImageFiles = ""
            for channel in channelsID_list:
                for i, n in enumerate(filesInDir):
                    currentFile = pathlib.Path(
                        os.path.join(currentSubjectDir, n)
                    ).as_posix()
                    if relativizePathsToOutput:
                        # commonRoot = os.path.commonpath(currentFile, outputFile)
                        currentFile = (
                            pathlib.Path(currentFile)
                            .resolve()
                            .relative_to(pathlib.Path(outputFile).resolve().parent)
                            .as_posix()
                        )
                    if channel in n:
                        allImageFiles += currentFile + ","
                    elif labelID is not None:
                        if labelID in n:
                            maskFile = currentFile
            if allImageFiles:
                outputToWrite += dirs + "," + allImageFiles + maskFile + "\n"

    file = open(outputFile, "w")
    file.write(outputToWrite)
    file.close()


def parseTrainingCSV(
    inputTrainingCSVFile: str, train: Optional[bool] = True
) -> Tuple[pd.DataFrame, dict]:
    """
    This function parses the input training CSV and returns a dictionary of headers and the full (randomized) data frame

    Args:
        inputTrainingCSVFile (str): The input data CSV file which contains all training data.
        train (Optional[bool], optional): Whether to train the model. Defaults to True.

    Returns:
        Tuple[pd.DataFrame, dict]: The full dataset for computation and the dictionary containing all relevant CSV headers.
    """
    ## read training dataset into data frame
    data_full = get_dataframe(inputTrainingCSVFile)
    # shuffle the data - this is a useful level of randomization for the training process
    if train:
        data_full = data_full.sample(frac=1).reset_index(drop=True)

    # find actual header locations for input channel and label
    # the user might put the label first and the channels afterwards
    # or might do it completely randomly
    headers = {}
    headers["channelHeaders"] = []
    headers["predictionHeaders"] = []
    headers["labelHeader"] = None
    headers["subjectIDHeader"] = None

    for col in data_full.columns:
        # add appropriate headers to read here, as needed
        col_lower = col.lower()
        currentHeaderLoc = data_full.columns.get_loc(col)
        if (
            ("channel" in col_lower)
            or ("modality" in col_lower)
            or ("image" in col_lower)
        ):
            headers["channelHeaders"].append(currentHeaderLoc)
        elif "valuetopredict" in col_lower:
            headers["predictionHeaders"].append(currentHeaderLoc)
        elif (
            ("subject" in col_lower) or ("patient" in col_lower) or ("pid" in col_lower)
        ):
            headers["subjectIDHeader"] = currentHeaderLoc
        elif (
            ("label" in col_lower)
            or ("mask" in col_lower)
            or ("segmentation" in col_lower)
            or ("ground_truth" in col_lower)
            or ("groundtruth" in col_lower)
        ):
            if headers["labelHeader"] is None:
                headers["labelHeader"] = currentHeaderLoc
            else:
                print(
                    "WARNING: Multiple label headers found in training CSV, only the first one will be used",
                    file=sys.stderr,
                )
    convert_relative_paths_in_dataframe(data_full, headers, inputTrainingCSVFile)
    return data_full, headers


def parseTestingCSV(
    inputTrainingCSVFile, output_dir
) -> Tuple[bool, pd.DataFrame, dict]:
    """
    This function parses the input testing CSV and returns a dictionary of headers and the full (randomized) data frame

    Args:
        inputTrainingCSVFile (str): The input data CSV file which contains all testing data.
        output_dir (str): The output directory for the updated_test_mapping.csv and the collision.csv.

    Returns:
        Tuple[bool, pd.DataFrame, dict]: A boolean indicating whether any collisions were found, the full dataset for computation, and the dictionary containing all relevant CSV headers.
    """
    data_full, headers = parseTrainingCSV(inputTrainingCSVFile, train=False)

    collision_status, data_full = handle_collisions(data_full, headers, output_dir)

    # If collisions are True, raise a warning that some patients with colliding subject_id were found
    # and a new mapping_csv was created to be used and write the location of the new mapping_csv
    # and the collision_csv to the user
    if collision_status:
        print(
            """WARNING: Some patients with colliding subject_id were found.
            A new mapping_csv was created to be used and a collision_csv was created
            to be used to map the old subject_id to the new subject_id.
            The location of the updated_test_mapping.csv and the collision.csv are: """
            + output_dir,
            file=sys.stderr,
        )

    return collision_status, data_full, headers


def get_dataframe(input_file: Union[str, pd.DataFrame]) -> pd.DataFrame:
    """
    This function parses the input and returns a data frame

    Args:
        input_file (Union[str, pd.DataFrame]): The input data file.

    Returns:
        pandas.DataFrame: The full dataset for computation.
    """
    return_dataframe = None
    if isinstance(input_file, str):
        if input_file.endswith(".pkl"):
            return_dataframe = pd.read_pickle(input_file)
        elif input_file.endswith(".csv"):
            return_dataframe = pd.read_csv(input_file)
    elif isinstance(input_file, pd.DataFrame):
        return_dataframe = input_file

    return return_dataframe


def convert_relative_paths_in_dataframe(
    input_dataframe: pd.DataFrame, headers: dict, path_root: str
) -> pd.DataFrame:
    """
    This function takes a dataframe containing paths and a root path (usually to a data CSV file).
    These paths are combined with that root to create an absolute path.
    This allows data to be found relative to the data.csv even if the working directory is in a different location.

    This should only be used when loading, not when saving a CSV.
    Args:
        input_dataframe (pd.DataFrame): The dataframe to be operated on (this is also modified).
        headers (dict): headers created from parseTrainingCSV (used for identifying fields to interpret as paths)
        path_root (str): A "root path" to which data is to be relatively found. Usually a data CSV.

    Returns:
        pandas.DataFrame: The dataset but with paths relativized.
    """
    if isinstance(path_root, pd.DataFrame):
        # Whenever this happens, we cannot get a csv file location,
        # but at this point the data has already been loaded from a CSV previously.
        return input_dataframe
    for column in input_dataframe.columns:
        loc = input_dataframe.columns.get_loc(column)
        if (loc == headers["labelHeader"]) or (loc in headers["channelHeaders"]):
            # These entries can be considered as paths to files
            for index, entry in enumerate(input_dataframe[column]):
                if isinstance(entry, str):
                    this_path = pathlib.Path(entry)
                    start_path = pathlib.Path(path_root)
                    if start_path.is_file():
                        start_path = start_path.parent
                    if not this_path.is_file():
                        if not this_path.is_absolute():
                            input_dataframe.loc[index, column] = str(
                                start_path.joinpath(this_path)
                            )
    return input_dataframe

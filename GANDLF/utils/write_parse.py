import os, sys, pathlib
import pandas as pd


def writeTrainingCSV(inputDir, channelsID, labelID, outputFile):
    """
    This function writes the CSV file based on the input directory, channelsID + labelsID strings

    Args:
        inputDir (str): The input directory.
        channelsID (str): The channel header(s) identifiers.
        labelID (str): The label header identifier.
        outputFile (str): The output files to write
    """
    channelsID_list = channelsID.split(",")  # split into list

    outputToWrite = "SubjectID,"
    for i, n in enumerate(channelsID_list):
        outputToWrite += "Channel_" + str(i) + ","
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


def parseTrainingCSV(inputTrainingCSVFile, train=True):
    """
    This function parses the input training CSV and returns a dictionary of headers and the full (randomized) data frame

    Args:
        inputTrainingCSVFile (str): The input data CSV file which contains all training data.
        train (bool, optional): Whether performing training. Defaults to True.

    Returns:
        pandas.DataFrame: The full dataset for computation.
        dict: The dictionary containing all relevant CSV headers.
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
            if headers["labelHeader"] == None:
                headers["labelHeader"] = currentHeaderLoc
            else:
                print(
                    "WARNING: Multiple label headers found in training CSV, only the first one will be used",
                    file=sys.stderr,
                )

    return data_full, headers


def get_dataframe(input_file):
    """
    This function parses the input and returns a data frame

    Args:
        input_file (Union[str, pd.DataFrame]): The input data file.

    Returns:
        pandas.DataFrame: The full dataset for computation.
    """
    if isinstance(input_file, str):
        if input_file.endswith(".pkl"):
            return pd.read_pickle(input_file)
        elif input_file.endswith(".csv"):
            return pd.read_csv(input_file)
    elif isinstance(input_file, pd.DataFrame):
        return input_file

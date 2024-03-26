from typing import Union
import os

import pandas as pd
from GANDLF.utils import get_dataframe, split_data


def split_data_and_save_csvs(
    input_data: Union[pd.DataFrame, str], output_dir: str, parameters: dict
) -> None:
    """
    Split the data into training, validation, and testing sets and save them as csvs in the output directory

    Args:
        input_data (Union[pd.Dataframe, str]): The input data to be split and saved.
        output_dir (str): The output directory to save the split data.
        parameters (dict): The parameters dictionary.
    """

    full_data = get_dataframe(input_data)

    dataframe_split = split_data(full_data, parameters)

    for (
        testing_and_valid_indeces,
        trainingData,
        validationData,
        testingData,
    ) in dataframe_split:
        # training and validation dataframes use the same index, since they are based on the validation split
        training_data_path = os.path.join(
            output_dir, f"training_{testing_and_valid_indeces[1]}.csv"
        )
        validation_data_path = os.path.join(
            output_dir, f"validation_{testing_and_valid_indeces[1]}.csv"
        )
        # testing dataframes use the first index
        testing_data_path = os.path.join(
            output_dir, f"testing_{testing_and_valid_indeces[0]}.csv"
        )

        for data, path in zip(
            [trainingData, validationData, testingData],
            [training_data_path, validation_data_path, testing_data_path],
        ):
            # check if the data is not None and the path does not exist
            if not os.path.exists(path):
                if data is not None:
                    data.to_csv(path, index=False)

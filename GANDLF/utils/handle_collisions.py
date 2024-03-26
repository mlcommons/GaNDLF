import os
from typing import Tuple

import pandas as pd


def handle_collisions(
    df: pd.DataFrame, headers: dict, output_path: str
) -> Tuple[bool, pd.DataFrame]:
    """
    This function checks for collisions in the subject IDs and updates the subject IDs in the dataframe to avoid collisions.

    This function takes a dataframe as input and checks if there are any pairs of subject IDs
    that are similar to each other. If it finds any such pairs, it renames the subject IDs by
    adding a suffix of '_v1', '_v2', or '_v3' to differentiate them. The function then creates
    a new dataframe that can be used for inference purposes. Additionally, it writes the original
    dataframe to disk for future reference and creates a 'collision.csv' file to inform the user
    of any subject ID collisions that were detected during the process.

    Args:
        df (pd.DataFrame): The dataframe containing the subject IDs to be checked for collisions.
        headers (dict): The headers dictionary containing the subjectIDHeader key.
        output_path (str): The path to the output directory where the updated dataframe and the collision.csv file will be written.

    Returns:
        Tuple[bool, pd.DataFrame]: A tuple containing a boolean indicating whether any collisions were found, and the updated dataframe.
    """

    # Find the subjectID header
    subject_id_column_name = headers.get("subjectIDHeader", None)
    assert (
        subject_id_column_name is not None
    ), "No subject ID column found in the headers."

    # Create a dictionary to store the count of each subjectid
    subjectid_counts = {}

    # Create a list to store the colliding subjectids
    collisions = []

    # Create a new dataframe to store the updated subjectids
    new_df = df.copy()

    # Loop through each row in the original dataframe
    for i, row in df.iterrows():
        subjectid = row[subject_id_column_name]

        # If the subjectid has not been seen before, add it to the dictionary
        if subjectid not in subjectid_counts:
            subjectid_counts[subjectid] = 0
        else:
            # If the subjectid has been seen before, increment the count and add it to the collisions list
            subjectid_counts[subjectid] += 1
            collisions.append(subjectid)
            # Update the subjectid in the new dataframe
            new_subjectid = f"{subjectid}_v{subjectid_counts[subjectid]}"
            new_df.at[i, subject_id_column_name] = new_subjectid

    # Write the colliding subject IDs to the collision.csv file
    if collisions:
        collision_path = os.path.join(output_path, "collision.csv")
        pd.DataFrame({subject_id_column_name: collisions}).to_csv(
            collision_path, index=False
        )

    # Write the updated dataframe to the updated_test_mapping.csv file
    mapping_path = os.path.join(output_path, "updated_test_mapping.csv")
    df.to_csv(mapping_path, index=False)

    # Return a tuple indicating whether any collisions were found, and the updated dataframe
    return len(collisions) > 0, df

import os

import pandas as pd


def handle_collisions(df, output_path):
    # Create a dictionary to store the count of each subjectid
    subjectid_counts = {}

    # Find the subjectid header
    # Find the subjectID header
    subject_id_column_name = None
    for col in df.columns:
        currentHeaderLoc = df.columns.get_loc(col)
        col_lower = col.lower()
        if (
            ("subject" in col_lower) or ("patient" in col_lower) or ("pid" in col_lower)
        ):
            subject_id_column_name = currentHeaderLoc
        else:
            raise ValueError("SubjectID not found in the header of the csv file")

    # Create the path to the collision.csv file and mapping.csv file
    collision_path = os.path.join(output_path, "collision.csv")
    mapping_path = os.path.join(output_path, "updated_test_mapping.csv")

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

    # Write the colliding subjectids to the collision.csv file
    pd.DataFrame({subject_id_column_name: collisions}).to_csv(collision_path, index=False)

    # Write the updated dataframe to the new_test_mapping.csv file
    new_df.to_csv(mapping_path, index=False)

    collisions_found = len(collisions) > 0

    return collisions_found, new_df


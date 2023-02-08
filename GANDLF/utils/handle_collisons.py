import pandas as pd

def handle_collisions(df, collision_path, mapping_path):
    # Create a dictionary to store the count of each subjectid
    subjectid_counts = {}
    
    # Create a list to store the colliding subjectids
    collisions = []
    
    # Create a new dataframe to store the updated subjectids
    new_df = df.copy()
    
    # Loop through each row in the original dataframe
    for i, row in df.iterrows():
        subjectid = row['subjectid']
        
        # If the subjectid has not been seen before, add it to the dictionary
        if subjectid not in subjectid_counts:
            subjectid_counts[subjectid] = 0
        else:
            # If the subjectid has been seen before, increment the count and add it to the collisions list
            subjectid_counts[subjectid] += 1
            collisions.append(subjectid)
        
        # Update the subjectid in the new dataframe
        new_subjectid = f"{subjectid}_v{subjectid_counts[subjectid]}"
        new_df.at[i, 'subjectid'] = new_subjectid
    
    # Write the colliding subjectids to the collision.csv file
    pd.DataFrame({'subjectid': collisions}).to_csv(collision_path, index=False)
    
    # Write the updated dataframe to the new_test_mapping.csv file
    new_df.to_csv(mapping_path, index=False)

    return new_df
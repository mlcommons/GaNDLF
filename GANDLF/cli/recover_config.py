import yaml
import pickle
import os



def recover_config(modelDir, outputFile):
    if not os.path.exists(modelDir):
        print("The model directory does not appear to exist. Please check parameters.")
        return False
        
    pickle_location = os.path.join(modelDir, "parameters.pkl")
    if (not os.path.exists(pickle_location)):
        print("The model does not appear to have a configuration file. Please check parameters.")
        return False
    else:
        with open(pickle_location, "rb") as handle:
            parameters = pickle.load(handle)
            os.makedirs(os.path.dirname(outputFile), exist_ok=True)
            
            # Remove a few problematic objects from the output
            removable_entries = [
                'output_dir',
                'second_output_dir',
                'training_data',
                'validation_data',
                'testing_data',
                'device',
                'subject_spacing', 
            ]
            for param in parameters:
                print(f"For param: {param} :")
                res = yaml.safe_dump( {parameters[param]} )
            #for entry in removable_entries:
            #    if entry in parameters:
            #        del parameters[entry]
            with open(outputFile, 'w') as f:
                print(parameters)
                f.write(yaml.safe_dump(parameters, default_flow_style=False))
    
    print(f"Config written to {outputFile}.")
    return True
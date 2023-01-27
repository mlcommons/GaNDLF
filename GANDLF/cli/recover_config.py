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
            with open(outputFile, 'w') as f:
                print(parameters)
                f.write(yaml.dump(parameters, default_flow_style=False))
    
    print(f"Config written to {outputFile}.")
    return True
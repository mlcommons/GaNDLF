import os
import pickle
import sys

from GANDLF.parseConfig import parseConfig

def readConfig(torch_model_dir, config_file = None):
    if config_file == None:
        if os.path.exists(os.path.join(torch_model_dir, 'parameters.pkl')):
           with open(os.path.join(torch_model_dir, 'parameters.pkl'), 'rb') as f:
                parameter = pickle.load(f)
                f.close()
        elif any(File.endswith(".yaml") for File in os.listdir(torch_model_dir)):
            for File in os.listdir(torch_model_dir):
                if File.endswith(".yaml"):
                    break
            parameter = parseConfig(os.path.join(torch_model_dir, File))
            if not 'num_classes' in parameter['model'].keys():
                parameter['model']['num_classes'] = len(
                    parameter['model']['class_list'])
            with open(os.path.join(torch_model_dir, 'parameters.pkl'), 'wb') as f:
                pickle.dump(parameter, f)
                f.close()
        else:
            sys.exit("Either a yaml config file or a pkl parameter file needs to be available under PyTorch model directory")
    else:
        if config_file.endswith(".yaml"):
            parameter = parseConfig(config_file)
            if not 'num_classes' in parameter['model'].keys():
                parameter['model']['num_classes'] = len(parameter['model']['class_list'])
        elif config_file.endswith(".pkl"):
            with open(config_file, 'rb') as f:
                parameter = pickle.load(f)
                f.close()
        else:
            sys.exit("Input configuration file should be in .yaml or .pkl format.")
    return(parameter)

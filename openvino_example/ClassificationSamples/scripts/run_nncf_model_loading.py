import os
import argparse

import torch
import nncf  # Important - should be imported directly after torch
from nncf import NNCFConfig
from nncf.torch import create_compressed_model
from nncf.torch.checkpoint_loading import load_state

from gandlf_func.readConfig import readConfig
from GANDLF.models import global_models_dict

parser = argparse.ArgumentParser(
    description='Convert the NNCF PyTorch model to ONNX model.')
parser.add_argument('-i', '--nncf_model',
                    help='The NNCF compressed PyTorch model path.')
parser.add_argument('-o', '--onnx_model',
                    help='The exported ONNX model path.')
parser.add_argument('-c', '--nncf_config',
                    help="The NNCG config file")
parser.add_argument('-p', '--config_file', required=False, 
                    help='Config yaml file or the parameter file')
args = parser.parse_args()


parameter = readConfig(config_file=args.config_file)


print(parameter)

model = global_models_dict[parameter["model"]
                           ["architecture"]](parameters=parameter)

nncf_config = NNCFConfig.from_json(args.nncf_config) 

# load parta
resuming_checkpoint = torch.load(args.nncf_model)
compression_state = resuming_checkpoint['compression_state'] 
compression_ctrl, compressed_model = create_compressed_model(model, nncf_config, compression_state=compression_state)
state_dict = resuming_checkpoint['state_dict'] 

# load model in a preferable way
load_state(compressed_model, state_dict, is_resume=True)     
compressed_model.eval()

if not(os.path.exists(os.path.dirname(args.onnx_model))):
    print("Generate new folder {0}".format(os.path.dirname(args.onnx_model)))
    os.mkdir(os.path.dirname(args.onnx_model))

compression_ctrl.export_model(args.onnx_model)

print("Onnx model is written to {0}.".format(args.onnx_model))


# Usage example: python benchmark_pt_ov.py -m './infer_models' -mn 'resunet' -ptm 3dresunet_pt -ovm 3dresunet_ov -nnm 3dresunet_ov_nncf -p ./3dresunet_exp_nncf/data_dir/parameters.pkl -d ./3dresunet_exp_nncf/tcga-val-data-pre-ma-test.csv -o ./3dresunet_exp_nncf/data_dir -v False

from pathlib import Path

import os
import sys
import time
import torch
print(torch.__version__)

import argparse
import pickle
from openvino.inference_engine import IECore

parser = argparse.ArgumentParser(
    description='Convert the NNCF PyTorch model to ONNX model.')
parser.add_argument('-nfold', '--n_fold',
                    help='The fold to use for evaluation')
parser.add_argument('-mn', '--model_name',
                    help='The model name', default='resunet')
parser.add_argument('-m', '--model_dir',
                    help='The PyTorch or OpenVINO model root directory path.')
parser.add_argument('-ptm', '--pytorch_model',
                    help='The PyTorch model path.')
parser.add_argument('-ovm', '--ov_model',
                    help='The OpenVINO model path.')
parser.add_argument('-nnm', '--nncf_model',
                    help='The NNCF optimized model path.')
parser.add_argument('-d', '--data_csv',
                    help='The path to data csv containing path to images and labels.')
parser.add_argument('-p', '--parameters_file', required=False, 
                    help='Config yaml file or the parameter file')
parser.add_argument('-o', '--output_dir', required=False, 
                    help='Output directory to store segmenation results')
parser.add_argument('-v', '--verbose', required=False, 
                    help='Whether to print verbose results')
args = parser.parse_args()

from tqdm import tqdm
import torchio

from gandlf_func.generate_dataloader_and_parameter import generate_data_loader
import gandlf_func.forward_pass_ov as forward_pass_ov

def load_torch_model(path, model, key = "model_state_dict"):
    main_dict = torch.load(path, map_location=torch.device('cpu'))
    model.load_state_dict(main_dict[key])
    model.eval()
    return model

def load_ov_model(path):
    ie = IECore()

    net = ie.read_network(model=path.with_suffix(".xml"), \
                          weights=path.with_suffix(".bin"))
    input_blob = next(iter(net.input_info))
    out_blob = next(iter(net.outputs))

    exec_net = ie.load_network(network=net, device_name="CPU")
    return exec_net, input_blob, out_blob

with open(args.parameters_file, 'rb') as f:
    parameters = pickle.load(f)

model, val_dataloader, parameters = generate_data_loader(args.data_csv, parameters, args.output_dir, args.verbose)

###Original PyTorch Model
orig_pth = os.path.join(args.model_dir, args.pytorch_model, args.model_name +"_best.pth.tar")
model = load_torch_model(orig_pth, model)
parameters['model']['type'] = "Torch"
st_time = time.time()
epoch_valid_loss, epoch_valid_metric = forward_pass_ov.validate_network_ov(
            model, val_dataloader, scheduler=None, params=parameters, epoch=0, mode="validation")
ed_time = time.time()
print("*****" + "Avg inference Time for Original PyTorch Model is: " + str((ed_time - st_time)/len(val_dataloader)))

### Original OpenVINO FP32 Model 
fp32_ir_path = Path(os.path.join(args.model_dir, args.ov_model + '/FP32'))
exec_net, input_blob, out_blob = load_ov_model(Path(fp32_ir_path / (args.model_name)))
parameters['model']['type'] = "OV"
parameters['model']['IO'] = [input_blob, out_blob]
st_time = time.time()
epoch_valid_loss, epoch_valid_metric = forward_pass_ov.validate_network_ov(
            exec_net, val_dataloader, scheduler=None, params=parameters, epoch=0, mode="validation")
ed_time = time.time()
print("*****" + "Avg inference Time for Original OV FP32 Model is: " + str((ed_time - st_time)/len(val_dataloader)))

###OpenVINO POT INT8 Model 
int8_ir_path = Path(os.path.join(args.model_dir, args.ov_model + '/INT8'))
exec_net, input_blob, out_blob = load_ov_model(Path(int8_ir_path / (args.model_name)))
parameters['model']['type'] = "OV"
parameters['model']['IO'] = [input_blob, out_blob]
st_time = time.time()
epoch_valid_loss, epoch_valid_metric = forward_pass_ov.validate_network_ov(
            exec_net, val_dataloader, scheduler=None, params=parameters, epoch=0, mode="validation")
ed_time = time.time()
print("*****" + "Avg inference Time for POT OV INT8 Model is: " + str((ed_time - st_time)/len(val_dataloader)))

###OpenVINO NNCF INT8 Model 
int8_nncf_path = Path(os.path.join(args.model_dir, args.nncf_model))
exec_net, input_blob, out_blob = load_ov_model(Path(int8_nncf_path / (args.model_name)))
parameters['model']['type'] = "OV"
parameters['model']['IO'] = [input_blob, out_blob]
st_time = time.time()
epoch_valid_loss, epoch_valid_metric = forward_pass_ov.validate_network_ov(
            exec_net, val_dataloader, scheduler=None, params=parameters, epoch=0, mode="validation")
ed_time = time.time()
print("*****" + "Avg inference Time for NNCF OV INT8 Model is: " + str((ed_time - st_time)/len(val_dataloader)))

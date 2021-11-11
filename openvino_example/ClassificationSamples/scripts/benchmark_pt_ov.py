from pathlib import Path

import os
import sys
import time

import torch
print(torch.__version__)

from openvino.inference_engine import IECore

N_FOLD = "0"
ROOT_DIR = "/Share/Junwen/UPenn/DFU_Example_vgg11/ClassificationModel/ClassificationModel/"
TORCH_MODEL_PTH = "models/DenseNet121/densenet121/trained_models/"
OV_MODEL_PTH = "models/DenseNet121/densenet121/"
NNCF_MODEL_PTH = "models/DenseNet121/densenet121/nncf_models/quantization/"
BASE_MODEL_NAME = "densenet121"

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

model, parameters, train_dataloader, val_dataloader, scheduler, optimizer = generate_data_loader(os.path.join(ROOT_DIR, "data"), N_FOLD, True, os.path.join(ROOT_DIR, TORCH_MODEL_PTH, "parameters.pkl"))

###Original PyTorch Model
orig_pth = os.path.join(ROOT_DIR, TORCH_MODEL_PTH,  N_FOLD, BASE_MODEL_NAME+"_best.pth.tar")
model = load_torch_model(orig_pth, model)
parameters['model']['type'] = "Torch"
st_time = time.time()
epoch_valid_loss, epoch_valid_metric = forward_pass_ov.validate_network_ov(
            model, val_dataloader, scheduler, parameters, epoch=0, mode="validation")
ed_time = time.time()
print("*****" + "Inference Time for Original PyTorch Model is: " + str((ed_time - st_time)/len(val_dataloader)))

### Original OpenVINO FP32 Model 
fp32_ir_path = Path(os.path.join(ROOT_DIR, OV_MODEL_PTH, "ov_models",  N_FOLD + '/'))
exec_net, input_blob, out_blob = load_ov_model(Path(fp32_ir_path / (BASE_MODEL_NAME + "_best")))
parameters['model']['type'] = "OV"
parameters['model']['IO'] = [input_blob, out_blob]
st_time = time.time()
epoch_valid_loss, epoch_valid_metric = forward_pass_ov.validate_network_ov(
            exec_net, val_dataloader, scheduler, parameters, epoch=0, mode="validation")
ed_time = time.time()
print("*****" + "Inference Time for Original OV FP32 Model is: " + str((ed_time - st_time)/len(val_dataloader)))

###OpenVINO POT INT8 Model 
int8_ir_path = Path(os.path.join(ROOT_DIR, OV_MODEL_PTH, 'ov_models', N_FOLD, 'INT8/'))
exec_net, input_blob, out_blob = load_ov_model(Path(int8_ir_path / (BASE_MODEL_NAME + "_best")))
parameters['model']['type'] = "OV"
parameters['model']['IO'] = [input_blob, out_blob]
st_time = time.time()
epoch_valid_loss, epoch_valid_metric = forward_pass_ov.validate_network_ov(
            exec_net, val_dataloader, scheduler, parameters, epoch=0, mode="validation")
ed_time = time.time()
print("*****" + "Inference Time for POT OV INT8 Model is: " + str((ed_time - st_time)/len(val_dataloader)))

###OpenVINO NNCF INT8 Model 
int8_onnx_path = Path(os.path.join(ROOT_DIR, NNCF_MODEL_PTH, "onnx", N_FOLD, "/"))
exec_net, input_blob, out_blob = load_ov_model(Path(int8_onnx_path / (BASE_MODEL_NAME + "_nncf_best")))
parameters['model']['type'] = "OV"
parameters['model']['IO'] = [input_blob, out_blob]
st_time = time.time()
epoch_valid_loss, epoch_valid_metric = forward_pass_ov.validate_network_ov(
            exec_net, val_dataloader, scheduler, parameters, epoch=0, mode="validation")
ed_time = time.time()
print("*****" + "Inference Time for NNCF OV INT8 Model is: " + str((ed_time - st_time)/len(val_dataloader)))

'''
pruned_factor = "05"
MODEL_DIR = Path(os.path.join(ROOT_DIR, "NNCF/models/pruning/models/" + N_FOLD + "/" + pruned_factor + "/"))
OUTPUT_DIR = Path(os.path.join(ROOT_DIR, "NNCF/models/pruning/outputs/" + N_FOLD + "/" + pruned_factor + "/"))
###OpenVINO NNCF filter pruned Model 
int8_onnx_path = Path(OUTPUT_DIR / (BASE_MODEL_NAME + "_pruned"))
int8_ir_path = int8_onnx_path.with_suffix(".xml")
exec_net, input_blob, out_blob = load_ov_model(Path(int8_ir_path / (BASE_MODEL_NAME + "_pruned")))
parameters['model']['type'] = "OV"
parameters['model']['IO'] = [input_blob, out_blob]
st_time = time.time()
epoch_valid_loss, epoch_valid_metric = forward_pass_ov.validate_network_ov(
            exec_net, val_dataloader, scheduler, parameters, epoch=0, mode="validation")
ed_time = time.time()
print("*****" + "Inference Time for NNCF OV filter prunning Model is: " + str((ed_time - st_time)/len(val_dataloader)))

# OpenVINO NNCF Pruned Model
pruned_model_path = Path("/Share/Junwen/UPenn/DFU_Example_vgg11/ClassificationModel/ClassificationModel/NNCF/models/pruning/ckpt/pruning_50pcnt_01pcnt")
exec_net, input_blob, out_blob = load_ov_model(pruned_model_path)
parameters['model']['type'] = "OV"
parameters['model']['IO'] = [input_blob, out_blob]
st_time = time.time()
epoch_valid_loss, epoch_valid_metric = forward_pass_ov.validate_network_ov(
            exec_net, val_dataloader, scheduler, parameters, epoch=0, mode="validation")
ed_time = time.time()
print("*****" + "Inference Time for NNCF OV PRUNED Model is: " + str((ed_time - st_time)/len(val_dataloader)))
'''

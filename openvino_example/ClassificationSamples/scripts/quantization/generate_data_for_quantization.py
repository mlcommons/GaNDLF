from pathlib import Path

import os
import sys
import time

import torch
import pickle
import argparse

parser = argparse.ArgumentParser(description='Convert the pretrained PyTorch model to ONNX model.')
parser.add_argument('-n', '--N_FOLD', default="0", help='n-th fold of the model.')
parser.add_argument('-p', '--pytorch_model_dir', default="models/DFU_experiments_vgg11_5fold_without_preprocess/", help='The PyTorch model directory.')
parser.add_argument('-d', '--data_dir', default="data", help='The PyTorch model directory.')
parser.add_argument('-t', '--data_type', default="train", help="train or validation data, options: train/validation")
parser.add_argument('-s', '--sampling_rate', type=float, default=1, help="sampling rate")
args = parser.parse_args()
N_FOLD = args.N_FOLD
PYTORCH_MODEL_DIR = args.pytorch_model_dir
DATA_DIR = args.data_dir
data_type = args.data_type

sys.path.append("../gandlf_func")

from generate_dataloader_and_parameter import generate_data_loader
import forward_pass_ov as forward_pass_ov

train_mode = False
model, parameters, train_dataloader, val_dataloader, scheduler, optimizer = generate_data_loader(DATA_DIR, N_FOLD, train_mode)
BASE_MODEL_NAME = parameters["model"]["architecture"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = Path(os.path.join(DATA_DIR, "patch_data",  N_FOLD, data_type ))

DATA_DIR.mkdir(exist_ok=True)

main_dict = torch.load(os.path.join(PYTORCH_MODEL_DIR, N_FOLD, BASE_MODEL_NAME + "_best.pth.tar"), map_location=device)
model.load_state_dict(main_dict["model_state_dict"])

parameters['model']['type'] = "Torch"
parameters['model']['save_data'] = True
parameters['model']['save_data_subsample'] = 1 - args.sampling_rate
if parameters['model']['save_data_subsample'] == 0:
    parameters['model']['save_data_filename'] = os.path.join(DATA_DIR, "patch_samples.npz")
else:
    parameters['model']['save_data_filename'] = os.path.join(DATA_DIR, "patch_samples_"+str(args.sampling_rate) +".npz")

if data_type == "validation":
    epoch_valid_loss, epoch_valid_metric = forward_pass_ov.validate_network_ov(
            model, val_dataloader, scheduler, parameters, epoch=0, mode="validation")
else:
    epoch_valid_loss, epoch_valid_metric = forward_pass_ov.validate_network_ov(
            model, train_dataloader, scheduler, parameters, epoch=0, mode="validation")

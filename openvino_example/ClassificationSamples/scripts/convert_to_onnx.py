import torch 
print(torch.__version__)
import os
import sys

from GANDLF.models.vgg import vgg11
import pickle

import onnx

import argparse

parser = argparse.ArgumentParser(description='Convert the pretrained PyTorch model to ONNX model.')
parser.add_argument('-i', '--torch_model_dir', help='The pretrained model directory.')
parser.add_argument('-o', '--onnx_model_dir', help='The exported ONNX model directory.')
args = parser.parse_args()

with open(os.path.join(args.torch_model_dir, 'parameters.pkl'), 'rb') as f:
    parameter = pickle.load(f)

device = torch.device("cpu")

model = vgg11(parameter)

for i in range(5):
	checkpoint = torch.load(os.path.join(args.torch_model_dir, str(i) + '/vgg11_best.pth.tar'), map_location=torch.device('cpu'))
	model.load_state_dict(checkpoint['model_state_dict'])
	model.to(device)
	model.eval()

	dummy_input = torch.randn((1,3,parameter['patch_size'][0], parameter['patch_size'][1]))
	torch.onnx.export(model,
                dummy_input.to("cpu"),
                os.path.join(args.onnx_model_dir, str(i) + '/vgg11_best.onnx'),
                opset_version=11,
                export_params=True,
                verbose=True,
                input_names = ['input'],
                output_names = ['output'])


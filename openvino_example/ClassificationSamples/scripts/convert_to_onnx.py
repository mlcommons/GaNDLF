import argparse
import onnx
import pickle
from GANDLF.parseConfig import parseConfig
from GANDLF.models.vgg import vgg11
from GANDLF.models import global_models_dict
import sys
import os
import torch
print(torch.__version__)


parser = argparse.ArgumentParser(
    description='Convert the pretrained PyTorch model to ONNX model.')
parser.add_argument('-i', '--torch_model_dir',
                    help='The pretrained model directory.')
parser.add_argument('-o', '--onnx_model_dir',
                    help='The exported ONNX model directory.')
parser.add_argument('-n', '--model_name', help='Model name')
args = parser.parse_args()

if os.path.exists(os.path.join(args.torch_model_dir, 'parameters.pkl')):
    with open(os.path.join(args.torch_model_dir, 'parameters.pkl'), 'rb') as f:
        parameter = pickle.load(f)
        f.close()
elif any(File.endswith(".yaml") for File in os.listdir(args.torch_model_dir)):
    for File in os.listdir(args.torch_model_dir):
        if File.endswith(".yaml"):
            break
    parameter = parseConfig(os.path.join(args.torch_model_dir, File))
    if not 'num_classes' in parameter['model'].keys():
        parameter['model']['num_classes'] = len(
            parameter['model']['class_list'])
    with open(os.path.join(args.torch_model_dir, 'parameters.pkl'), 'wb') as f:
        pickle.dump(parameter, f)
        f.close()
else:
    print("Either a yaml config file or a pkl parameter file needs to be available under PyTorch model directory")

print(parameter)
device = torch.device("cpu")

model = global_models_dict[parameter["model"]
                           ["architecture"]](parameters=parameter)

for i in range(5):
    checkpoint = torch.load(os.path.join(args.torch_model_dir, str(
        i), args.model_name+'_best.pth.tar'), map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    dummy_input = torch.randn(
        (1, 3, parameter['patch_size'][0], parameter['patch_size'][1]))

    if not(os.path.exists(args.onnx_model_dir)):
        print("Generate new folder {0}".format(args.onnx_model_dir))
        os.mkdir(args.onnx_model_dir)
    
    if not(os.path.exists(os.path.join(args.onnx_model_dir, str(i)))):
        print("Generate new folder {0}".format(os.path.join(args.onnx_model_dir, str(i))))
        os.mkdir(os.path.join(args.onnx_model_dir, str(i)))

    torch.onnx.export(model,
                      dummy_input.to("cpu"),
                      os.path.join(args.onnx_model_dir, str(
                          i), args.model_name+'_best.onnx'),
                      opset_version=11,
                      export_params=True,
                      verbose=True,
                      input_names=['input'],
                      output_names=['output'])

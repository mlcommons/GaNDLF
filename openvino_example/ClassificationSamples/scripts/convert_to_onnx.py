import argparse
from gandlf_func.readConfig import readConfig
from gandlf_func.convert_single_torch_to_onnx import convert_single_torch_to_onnx
from GANDLF.models import global_models_dict
import os

parser = argparse.ArgumentParser(
    description='Convert the pretrained PyTorch model to ONNX model.')
parser.add_argument('-i', '--torch_model_dir',
                    help='The pretrained model directory.')
parser.add_argument('-o', '--onnx_model_dir',
                    help='The exported ONNX model directory.')
parser.add_argument('-n', '--model_name', help='Model name')
parser.add_argument('-p', '--config_file', required=False, help='Config yaml file or the parameter file')
args = parser.parse_args()


if not(args.config_file):
    parameter = readConfig(args.torch_model_dir)
else:
    parameter - readConfig(args.torch_model_dir, args.config_file)

        
print(parameter)

model = global_models_dict[parameter["model"]
                           ["architecture"]](parameters=parameter)

if not(os.path.exists(args.onnx_model_dir)):
        print("Generate new folder {0}".format(args.onnx_model_dir))
        os.mkdir(args.onnx_model_dir)

for i in range(5):

    torch_weights = os.path.join(args.torch_model_dir, str(i), args.model_name+'_best.pth.tar')
    input_size =  (1, 3, parameter['patch_size'][0], parameter['patch_size'][1])
    onnx_model = os.path.join(args.onnx_model_dir, str(i), args.model_name+'_best.onnx')

    convert_single_torch_to_onnx(model, torch_weights, input_size, onnx_model)


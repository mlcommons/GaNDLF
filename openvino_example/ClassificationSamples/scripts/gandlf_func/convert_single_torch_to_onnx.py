import torch
import onnx
import os

print(torch.__version__)

device = torch.device('cpu')

def convert_single_torch_to_onnx(torch_model, torch_weight, input_shape, onnx_model):
    checkpoint = torch.load(torch_weight, map_location=device)
    torch_model.load_state_dict(checkpoint['model_state_dict'])
    torch_model.to(device)
    torch_model.eval()

    dummy_input = torch.randn(input_shape)

    if not(os.path.exists(os.path.dirname(onnx_model))):
        print("Generate new folder {0}".format(os.path.dirname(onnx_model)))
        os.mkdir(os.path.dirname(onnx_model))

    torch.onnx.export(torch_model,
                      dummy_input.to("cpu"),
                      onnx_model,
                      opset_version=11,
                      export_params=True,
                      verbose=True,
                      input_names=['input'],
                      output_names=['output'])

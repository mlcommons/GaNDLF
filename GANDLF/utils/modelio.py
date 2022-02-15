import hashlib, pkg_resources, subprocess
from time import gmtime, strftime
import os
import torch
from openvino.inference_engine import IECore

# these are the base keys for the model dictionary to save
model_dict_base = {
    "epoch": 0,
    "model_state_dict": None,
    "optimizer_state_dict": None,
    "loss": None,
    "timestamp": None,
    "timestamp_hash": None,
    "git_hash": None,
    "version": None,
}


def save_model(model_dict, model, input_shape, path):
    """
    Save the model dictionary to a file.

    Args:
        model_dict (dict): Model dictionary to save.
        model (torch model): trained torch model.
        input_shape (list or triple): input patch size to export the model.
        path (str): The path to save the model dictionary to.
    """
    model_dict["timestamp"] = strftime("%Y%m%d%H%M%S", gmtime())
    model_dict["timestamp_hash"] = hashlib.sha256(
        str(model_dict["timestamp"]).encode("utf-8")
    ).hexdigest()
    model_dict["version"] = pkg_resources.require("GANDLF")[0].version
    try:
        model_dict["git_hash"] = (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .decode("ascii")
            .strip()
        )
    except subprocess.CalledProcessError:
        model_dict["git_hash"] = None
    torch.save(model_dict, path)

    onnx_path = path.replace("pth.tar", "onnx")
    dummy_input = torch.randn((1, 3, input_shape[0], input_shape[1]))

    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input.to("cpu"),
            onnx_path,
            opset_version=11,
            export_params=True,
            verbose=True,
            input_names=["input"],
            output_names=["output"]
        )

    ov_output_dir = os.path.dirname(os.path.abspath(path))
    subprocess.call(
        "mo.py -m {0} --input_shape [1,3,{1},{2}] --output_dir {3}".format(
            onnx_path, input_shape[0], input_shape[1], ov_output_dir
        ),
        shell=True,
    )


def load_model(path):
    """
    Load a model dictionary from a file.

    Args:
        path (str): The path to save the model dictionary to.

    Returns:
        dict: Model dictionary containing model parameters and metadata.
    """
    model_dict = torch.load(path)

    # check if the model dictionary is complete
    incomplete_keys = [
        key for key in model_dict_base.keys() if key not in model_dict.keys()
    ]

    if len(incomplete_keys) > 0:
        print(
            "Model dictionary is incomplete; the following keys are missing:",
            incomplete_keys,
        )

    return model_dict


def load_ov_model(path, device="CPU"):
    """
    Load an OpenVINO IR model from an .xml file.

    Args:
        path (str): The path to the OpenVINO .xml file.
        device (str): The device to run inference, can be "CPU", "GPU" or "MULTI:CPU,GPU". Default to be "CPU".

    Returns:
        exec_net (OpenVINO executable net): executable OpenVINO model.
        input_blob (str): Input name.
        output_blob (str): Output name.
    """

    ie = IECore()
    if device == "GPU":
        ie.set_config(
            config={"CACHE_DIR": os.path.dirname(os.path.abspath(path))},
            device_name=device,
        )

    net = ie.read_network(model=path, weights=path.replace("xml", "bin"))

    input_blob = next(iter(net.input_info))
    out_blob = next(iter(net.outputs))

    exec_net = ie.load_network(network=net, device_name=device)
    return exec_net, input_blob, out_blob

import hashlib, os
from typing import Any, Dict, Optional, Tuple

import torch

from ..version import __version__
from .generic import get_unique_timestamp, get_git_hash

# these are the base keys for the model dictionary to save
model_dict_full = {
    "epoch": 0,
    "model_state_dict": None,
    "optimizer_state_dict": None,
    "loss": None,
    "timestamp": None,
    "timestamp_hash": None,
    "git_hash": None,
    "version": None,
}

model_dict_required = {"model_state_dict": None, "optimizer_state_dict": None}

BEST_MODEL_PATH_END = "_best.pth.tar"
LATEST_MODEL_PATH_END = "_latest.pth.tar"
INITIAL_MODEL_PATH_END = "_initial.pth.tar"


def optimize_and_save_model(
    model: torch.nn.Module,
    params: dict,
    output_path: str,
    onnx_export: Optional[bool] = True,
) -> None:
    """
    Perform post-training optimization and save it to a file.

    Args:
        model (torch.nn.Module): Trained torch model.
        params (dict): The parameter dictionary.
        output_path (str): The path to save the optimized model to.
        onnx_export (Optional[bool]): Whether to export to ONNX and OpenVINO. Defaults to True.
    """
    # Check if ONNX export is enabled in the parameter dictionary
    onnx_export = params["model"].get("onnx_export", onnx_export)

    # Check for incompatible topologies and disable ONNX export
    # Customized imagenet_vgg no longer supported for ONNX export
    if onnx_export:
        architecture = params["model"]["architecture"]
        if architecture in ["sdnet", "brain_age"] or "imagenet_vgg" in architecture:
            onnx_export = False

    if not onnx_export:
        # Print a warning if ONNX export is disabled and not already warned
        if "onnx_print" not in params:
            print("WARNING: Current model is not supported by ONNX/OpenVINO!")
            params["onnx_print"] = True
        return
    else:
        try:
            print("Optimizing the best model.")
            num_channel = params["model"]["num_channels"]
            model_dimension = params["model"]["dimension"]
            input_shape = params["patch_size"]
            onnx_path = output_path.replace(".pth.tar", ".onnx")

            if model_dimension == 2:
                dummy_input = torch.randn(
                    (1, num_channel, input_shape[0], input_shape[1])
                )
            else:
                dummy_input = torch.randn(
                    (1, num_channel, input_shape[0], input_shape[1], input_shape[2])
                )

            # Export the model to ONNX format
            with torch.no_grad():
                torch.onnx.export(
                    model.to("cpu"),
                    dummy_input.to("cpu"),
                    onnx_path,
                    opset_version=11,
                    export_params=True,
                    verbose=True,
                    input_names=["input"],
                    output_names=["output"],
                )
        except RuntimeWarning:
            print("WARNING: Cannot export to ONNX model.")
            return

        # Check if OpenVINO is present and try to convert the ONNX model
        openvino_present = False
        try:
            import openvino as ov
            from openvino.tools.mo import convert_model
            from openvino.runtime import get_version

            # debugging
            openvino_present = True
            # openvino_present = False
            # # check for the correct openvino version to prevent inadvertent api breaks
            # if "2024.1.0" in get_version():
            #     openvino_present = True
        except ImportError:
            print("WARNING: OpenVINO is not present.")

        if openvino_present:
            xml_path = onnx_path.replace("onnx", "xml")
            bin_path = onnx_path.replace("onnx", "bin")
            try:
                if model_dimension == 2:
                    ov_model = convert_model(
                        onnx_path,
                        input_shape=(1, num_channel, input_shape[0], input_shape[1]),
                    )
                else:
                    ov_model = convert_model(
                        onnx_path,
                        input_shape=(
                            1,
                            num_channel,
                            input_shape[0],
                            input_shape[1],
                            input_shape[2],
                        ),
                    )
                ov.runtime.serialize(ov_model, xml_path=xml_path, bin_path=bin_path)
            except Exception as e:
                print("WARNING: OpenVINO Model Optimizer IR conversion failed: " + e)


def save_model(
    model_dict: Dict[str, Any],
    model: torch.nn.Module,
    params: Dict[str, Any],
    path: str,
    onnx_export: Optional[bool] = True,
) -> None:
    """
    Save the model dictionary to a file.

    Args:
        model_dict (dict): Model dictionary to save.
        model (torch.nn.Module): Trained torch model.
        params (dict): The parameter dictionary.
        path (str): The path to save the model dictionary to.
        onnx_export (Optional[bool]): Whether to export to ONNX and OpenVINO. Defaults to True.
    """
    model_dict["timestamp"] = get_unique_timestamp()
    model_dict["timestamp_hash"] = hashlib.sha256(
        str(model_dict["timestamp"]).encode("utf-8")
    ).hexdigest()
    model_dict["version"] = __version__
    model_dict["parameters"] = params
    model_dict["git_hash"] = get_git_hash()

    torch.save(model_dict, path)

    # post-training optimization
    optimize_and_save_model(model, params, path, onnx_export=onnx_export)


def load_model(
    path: str, device: torch.device, full_sanity_check: Optional[bool] = True
) -> Dict[str, Any]:
    """
    Load a model dictionary from a file.

    Args:
        path (str): The path to save the model dictionary to.
        device (torch.device): The device to run the model on.
        full_sanity_check (Optional[bool]): Whether to perform a full sanity check. Defaults to True.

    Returns:
        dict: Model dictionary containing model parameters and metadata.
    """
    model_dict = torch.load(path, map_location=device)

    # check if the model dictionary is complete
    if full_sanity_check:
        incomplete_keys = [
            key for key in model_dict_full.keys() if key not in model_dict.keys()
        ]
        assert (
            len(incomplete_keys) == 0
        ), "Model dictionary is incomplete; the following keys are missing: " + str(
            incomplete_keys
        )

    # check if required keys are absent, and if so raise an error
    incomplete_required_keys = [
        key for key in model_dict_required.keys() if key not in model_dict.keys()
    ]
    assert (
        len(incomplete_required_keys) == 0
    ), "Model dictionary is incomplete; the following keys are missing: " + str(
        incomplete_required_keys
    )

    return model_dict


def load_ov_model(path: str, device: Optional[str] = "CPU") -> Tuple[Any, Any, Any]:
    """
    Load an OpenVINO IR model from an .xml file.

    Args:
        path (str): The path to the OpenVINO .xml file.
        device (Optional[str]): The device to run the model on, can be "CPU", "GPU" or "MULTI:CPU,GPU". Defaults to "CPU".

    Returns:
        Tuple[Any, Any, Any]: The compiled OpenVINO model, input layer name, and output layer name.
    """

    try:
        from openvino import runtime as ov
    except ImportError:
        raise ImportError("OpenVINO inference engine is not configured correctly.")

    core = ov.Core()
    if device.lower() == "cuda":
        device = "GPU"

    if device == "GPU":
        core.set_property({"CACHE_DIR": os.path.dirname(os.path.abspath(path))})

    model = core.read_model(model=path, weights=path.replace("xml", "bin"))
    compiled_model = core.compile_model(model=model, device_name=device.upper())
    input_layer = compiled_model.inputs
    output_layer = compiled_model.outputs

    return compiled_model, input_layer, output_layer

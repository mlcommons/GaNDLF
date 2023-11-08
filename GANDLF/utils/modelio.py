import hashlib
import os
import subprocess
from typing import Any, Dict, Optional

import torch

from ..version import __version__
from .generic import get_unique_timestamp

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

model_dict_required = {
    "model_state_dict": None,
    "optimizer_state_dict": None,
}

best_model_path_end = "_best.pth.tar"
latest_model_path_end = "_latest.pth.tar"
initial_model_path_end = "_initial.pth.tar"


def optimize_and_save_model(model, params, path, onnx_export=True):
    """
    Perform post-training optimization and save it to a file.

    Args:
        model (torch.nn.Module): Trained torch model.
        params (dict): The parameter dictionary.
        path (str): The path to save the model dictionary to.
        onnx_export (bool): Whether to export to ONNX and OpenVINO.
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
            onnx_path = path
            if not onnx_path.endswith(".onnx"):
                onnx_path = onnx_path.replace("pth.tar", "onnx")

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

            openvino_present = False
            # check for the correct openvino version to prevent inadvertent api breaks
            if "2023.0.1" in get_version():
                openvino_present = True
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
    onnx_export: bool = True,
    hf_hub_repo_id: Optional[str] = None,
):
    """
    Save the model dictionary to a file.

    Args:
        model_dict (dict): Model dictionary to save.
        model (torch.nn.Module): Trained torch model.
        params (dict): The parameter dictionary.
        path (str): The path to save the model dictionary to.
        onnx_export (bool): Whether to export to ONNX and OpenVINO.
        hf_hub_repo_id (str): The Hugging Face Hub repo id to push to. Defaults to None (will not push to HF Hub).
    """
    model_dict["timestamp"] = get_unique_timestamp()
    model_dict["timestamp_hash"] = hashlib.sha256(
        str(model_dict["timestamp"]).encode("utf-8")
    ).hexdigest()
    model_dict["version"] = __version__
    model_dict["parameters"] = params

    try:
        model_dict["git_hash"] = (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .decode("ascii")
            .strip()
        )
    except subprocess.CalledProcessError:
        model_dict["git_hash"] = None

    torch.save(model_dict, path)

    # post-training optimization
    optimize_and_save_model(model, params, path, onnx_export=onnx_export)

    if hf_hub_repo_id is not None:
        push_model_to_hf_hub(model_path=path, repo_path=path, repo_id=hf_hub_repo_id)
        #TODO: also push optimized models?

def load_model(
    path: str, device: torch.device, full_sanity_check: bool = True
) -> Dict[str, Any]:
    """
    Load a model dictionary from a file.

    Args:
        path (str): The path to save the model dictionary to.
        device (torch.device): The device to run the model on.
        full_sanity_check (bool): Whether to run full sanity checking on the model.

    Returns:
        dict: Model dictionary containing model parameters and metadata.
    """
    model_dict = torch.load(path, map_location=device)

    # check if the model dictionary is complete
    if full_sanity_check:
        incomplete_keys = [
            key for key in model_dict_full.keys() if key not in model_dict.keys()
        ]
        if len(incomplete_keys) > 0:
            raise RuntimeWarning(
                "Model dictionary is incomplete; the following keys are missing:",
                incomplete_keys,
            )

    # check if required keys are absent, and if so raise an error
    incomplete_required_keys = [
        key for key in model_dict_required.keys() if key not in model_dict.keys()
    ]
    if len(incomplete_required_keys) > 0:
        raise KeyError(
            "Model dictionary is incomplete; the following keys are missing:",
            incomplete_required_keys,
        )

    return model_dict


def load_ov_model(path: str, device: str = "CPU"):
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


def load_model_from_hf_hub(
        repo_id: str,
        model_filename: str,
        revision: str = None,
        local_dir: str = None,
        device: str = "CPU",
    ) -> Dict[str, Any]:
    """
    Download and load model from Hugging Face Hub. If the repo is private, credentials must be set beforehand.

    Args:
        repo_id (str): The Hugging Face Hub repo id.
        model_file_path (str): The model filename in the repo.
        revision (Optional[str]): The revision of the model to load. Defaults to
        the latest revision.
        local_dir (Optional[str]): The local directory to download the model to. Defaults to None.
        device (str): The device to run inference, can be "CPU", "GPU" or "MULTI:CPU,GPU". Default is "CPU".

    Returns:
        path: Path to model file. Can be used in `load_model`.
    """
    try:
        import huggingface_hub
    except ImportError:
        print("WARNING: huggingface_hub is not present.")

    local_model_path = huggingface_hub.hf_hub_download(
        repo_id=repo_id, 
        filename=model_filename,
        revision=revision,
        local_dir=local_dir
    )

    return load_model(local_model_path, device)


def upload_model_repo_to_hf_hub(
        repo_id: str,
        local_path: str,
        path_in_repo: str,
        model_card_path: Optional[str] = None,
        private: Optional[bool] = False,
        upload_onnx: Optional[bool] = False,
        upload_ov: Optional[bool] = False
    ) -> str:
    """
    Upload model to repo on Hugging Face Hub. Must be logged in to Hugging Face Hub.

    Args:
        repo_id (str): The Hugging Face Hub repo id to upload to. Will create if does not already exist.
        local_path (str): The path to the model to upload.
        path_in_repo (str): The path to the model in the repo.
        model_card_path (Optional[str]): The path to the model card to upload. Defaults to None.
        private (Optional[bool]): Whether to make the repo private. Defaults to False.
        upload_onnx (Optional[bool]): Whether to upload the ONNX model. Defaults to False.
        upload_ov (Optional[bool]): Whether to upload the OpenVINO model. Defaults to False.
    
    Returns:
        str: The revision of the model.
    """
    try:
        import huggingface_hub
    except ImportError:
        print("WARNING: huggingface_hub is not present.")

    huggingface_hub.create_repo(repo_id=repo_id, private=private)

    api = huggingface_hub.HfApi()

    api.upload_file(path_or_file_obj=local_path, path_in_repo=path_in_repo, repo_id=repo_id)

    if model_card_path is not None:
        #TODO: upload model card
        pass
    else:
        #TODO: create new model card
        pass
    
    #TODO: upload optimized models?


def push_model_to_hf_hub(
    repo_id: str,
    local_path: str,
    path_in_repo: str
) -> str:
    """
    Push model to repo on Hugging Face Hub. Must be logged in to Hugging Face Hub.

    Args:
        repo_id (str): The Hugging Face Hub repo id to push to.
        model_path (str): The local path to the model to push.
        path_in_repo (str): The path to the model in the repo.

    Returns:
        str: The URL to visualize the uploaded file on the hub.
    """
    
    try:
        import huggingface_hub
    except ImportError:
        print("WARNING: huggingface_hub is not present.")

    api = huggingface_hub.HfApi()
    
    return api.upload_file(
        path_or_file_obj=local_path,
        path_in_repo=path_in_repo,
        repo_id=repo_id
    )
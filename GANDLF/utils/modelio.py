import hashlib, pkg_resources, subprocess
from time import gmtime, strftime

import torch

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


def save_model(model_dict, path):
    """
    Save the model dictionary to a file.

    Args:
        model_dict (dict): Model dictionary to save.
        path (str): The path to save the model dictionary to.
    """
    model_dict["timestamp"] = strftime("%Y%m%d%H%M%S", gmtime())
    model_dict["timestamp_hash"] = hashlib.sha256(
        str(model_dict["timestamp"]).encode("utf-8")
    ).hexdigest()
    model_dict["version"] = pkg_resources.require("GANDLF")[0].version
    try:
        model_dict["git_hash"] = (
            subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
        )
    except subprocess.CalledProcessError:
        model_dict["git_hash"] = None
    torch.save(model_dict, path)


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

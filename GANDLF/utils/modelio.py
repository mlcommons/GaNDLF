import torch

# these are the base keys for the model dictionary to save
model_dict_base = {
    "epoch": 0,
    "model_state_dict": None,
    "optimizer_state_dict": None,
    "loss": None,
    "timestamp": None,
    "hash": None,
}

def save_model(model_dict, path):
    """
    Save the model dictionary to a file.

    Args:
        model_dict (dict): Model dictionary to save.
        path ([type]): [description]
    """
    torch.save(model_dict, path)
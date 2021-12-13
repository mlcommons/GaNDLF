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
        path (str): The path to save the model dictionary to.
    """
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
    incomplete_keys = [key for key in model_dict_base.keys() if key not in model_dict.keys()]
    
    if len(incomplete_keys) > 0:
        print("Model dictionary is incomplete; the following keys are missing:", incomplete_keys)
    
    return model_dict
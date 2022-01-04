from GANDLF.models import global_gan_models_dict


def is_GAN(model_architecture):
    """
    This function checks if the model architecture is a GAN or not.
    Args:
        model_architecture (str): The model architecture to check.
    Returns:
        bool: If the model architecture is a GAN or not.
    """
    if model_architecture in global_gan_models_dict.keys():
        return True
    return False

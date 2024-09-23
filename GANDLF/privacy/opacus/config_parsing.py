from typing import Callable


def parse_opacus_params(params: dict, initialize_key: Callable) -> dict:
    """
    Function to set defaults and augment the parameters related to making a trained model differentially
    private with respect to the training data.

    Args:
        params (dict): Training parameters.
        initialize_key (Callable): Function to fill in value for a missing key.

    Returns:
        dict: Updated training parameters.
    """

    if not isinstance(params["differential_privacy"], dict):
        print(
            "WARNING: Non dictionary value for the key: 'differential_privacy' was used, replacing with default valued dictionary."
        )
        params["differential_privacy"] = {}
    # these are some defaults
    params["differential_privacy"] = initialize_key(
        params["differential_privacy"], "noise_multiplier", 10.0
    )
    params["differential_privacy"] = initialize_key(
        params["differential_privacy"], "max_grad_norm", 1.0
    )
    params["differential_privacy"] = initialize_key(
        params["differential_privacy"], "accountant", "rdp"
    )
    params["differential_privacy"] = initialize_key(
        params["differential_privacy"], "secure_mode", False
    )
    params["differential_privacy"] = initialize_key(
        params["differential_privacy"], "allow_opacus_model_fix", True
    )
    params["differential_privacy"] = initialize_key(
        params["differential_privacy"], "delta", 1e-5
    )
    params["differential_privacy"] = initialize_key(
        params["differential_privacy"], "physical_batch_size", params["batch_size"]
    )

    if params["differential_privacy"]["physical_batch_size"] > params["batch_size"]:
        print(
            f"WARNING: The physical batch size {params['differential_privacy']['physical_batch_size']} is greater"
            f"than the batch size {params['batch_size']}, setting the physical batch size to the batch size."
        )
    params["differential_privacy"]["physical_batch_size"] = params["batch_size"]

    # these keys need to be parsed as floats, not strings
    for key in ["noise_multiplier", "max_grad_norm", "delta", "epsilon"]:
        if key in params["differential_privacy"]:
            params["differential_privacy"][key] = float(
                params["differential_privacy"][key]
            )

    return params

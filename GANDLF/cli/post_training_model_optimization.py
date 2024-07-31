import os
from pathlib import Path
from typing import Optional
from GANDLF.compute import create_pytorch_objects
from GANDLF.config_manager import ConfigManager
from GANDLF.utils import version_check, load_model, optimize_and_save_model


def post_training_model_optimization(
    model_path: str, config_path: Optional[str] = None, output_dir: Optional[str] = None
) -> bool:
    """
    CLI function to optimize a model for deployment.

    Args:
        model_path (str): Path to the model file.
        config_path (str, optional): Path to the configuration file.
        output_dir (str, optional): Output directory to save the optimized model.

    Returns:
        bool: True if successful, False otherwise.
    """
    # Load the model and its parameters from the given paths
    main_dict = load_model(model_path, "cpu")
    parameters = main_dict.get("parameters", None)

    # If parameters are not available in the model file, parse them from the config file
    parameters = (
        ConfigManager(config_path, version_check_flag=False)
        if parameters is None
        else parameters
    )

    output_dir = os.path.dirname(model_path) if output_dir is None else output_dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    optimized_model_path = os.path.join(
        output_dir, os.path.basename(model_path).replace("pth.tar", "onnx")
    )

    # Create PyTorch objects and set onnx_export to True for optimization
    model, _, _, _, _, parameters = create_pytorch_objects(parameters, device="cpu")
    parameters["model"]["onnx_export"] = True

    # Perform version check and load the model's state dictionary
    version_check(parameters["version"], version_to_check=main_dict["version"])
    model.load_state_dict(main_dict["model_state_dict"])

    # Optimize the model and save it to an ONNX file
    optimize_and_save_model(model, parameters, optimized_model_path, onnx_export=True)

    # Check if the optimized model file exists
    if not os.path.exists(optimized_model_path):
        print("Error while optimizing the model.")
        return False

    return True

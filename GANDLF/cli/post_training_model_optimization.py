import argparse
import os
from GANDLF.compute import create_pytorch_objects
from GANDLF.parseConfig import parseConfig
from GANDLF.utils import version_check, load_model, optimize_and_save_model
from .copyright_message import copyrightMessage


def post_training_model_optimization(model_path, config_path):
    """
    CLI function to optimize a model for deployment.

    Args:
        model_path (str): Path to the model file.
        config_path (str): Path to the config file.

    Returns:
        bool: True if successful, False otherwise.
    """

    main_dict = load_model(model_path, "cpu")
    parameters = main_dict.get("parameters", None)
    parameters = (
        parseConfig(config_path, version_check_flag=False)
        if parameters is None
        else parameters
    )
    (
        model,
        _,
        _,
        _,
        _,
        parameters,
    ) = create_pytorch_objects(parameters, device="cpu")
    parameters["model"]["onnx_export"] = True

    version_check(parameters["version"], version_to_check=main_dict["version"])
    model.load_state_dict(main_dict["model_state_dict"])
    optimize_and_save_model(model, parameters, model_path, onnx_export=True)
    optimized_model_path = model_path.replace("pth.tar", "onnx")
    if not os.path.exists(optimized_model_path):
        print("Error while optimizing model.")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(
        prog="GANDLF_OptimizeModel",
        formatter_class=argparse.RawTextHelpFormatter,
        description="Generate optimized versions of trained GaNDLF models.\n\n"
        + copyrightMessage,
    )

    parser.add_argument(
        "-m",
        "--model",
        metavar="",
        type=str,
        help="Path to the model file (ending in '.pth.tar') you wish to optimize.",
        required=True,
    )
    parser.add_argument(
        "-c",
        "--config",
        metavar="",
        type=str,
        default=None,
        required=False,
        help="The configuration file (contains all the information related to the training/inference session).",
    )

    args = parser.parse_args()

    if post_training_model_optimization(args.model, args.config):
        print("Post-training model optimization successful.")
    else:
        print("Post-training model optimization failed.")



if __name__ == "__main__":
    main()
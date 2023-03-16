from GANDLF.compute import create_pytorch_objects
from GANDLF.parseConfig import parseConfig
from GANDLF.utils import version_check, load_model, save_model, optimize_and_save_model


def post_training_model_optimization(model_path, config_path):
    """
    CLI function to optimize a model for deployment.

    Args:
        model_path (str): Path to the model file.
        config_path (str): Path to the config file.

    Returns:
        bool: True if successful, False otherwise.
    """

    parameters = parseConfig(config_path, version_check_flag=False)
    (
        model,
        _,
        _,
        _,
        _,
        parameters,
    ) = create_pytorch_objects(parameters, device="cpu")

    main_dict = load_model(model_path, "cpu")
    version_check(parameters["version"], version_to_check=main_dict["version"])
    model.load_state_dict(main_dict["model_state_dict"])
    try:
        optimize_and_save_model(model, parameters, model_path, onnx_export=True)
        return True
    except Exception as e:
        print("Error while optimizing model: ", e)
        return False

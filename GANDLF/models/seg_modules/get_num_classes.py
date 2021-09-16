
def get_number_of_outputs(parameters):
    """
    Helper function to get the number of outputs of a model.

    Args:
        parameters (dict): The parameters of the model.

    Returns:
        int: The number of outputs for the model.
    """
    if parameters["problem_type"] == "regression":
        num_classes = len(parameters["headers"]["predictionHeaders"])
    elif parameters["problem_type"] == "classification":
        num_classes = len(parameters["model"]["class_list"])
    return num_classes

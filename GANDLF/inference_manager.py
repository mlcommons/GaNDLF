from GANDLF.inference_loop import inference_loop


def InferenceManager(dataframe, outputDir, parameters, device):
    """
    This function takes in a dataframe, with some other parameters and performs the inference
    """
    # get the indeces for kfold splitting
    inferenceData_full = dataframe

    # # initialize parameters for inference
    if not ("weights" in parameters):
        parameters["weights"] = None  # no need for loss weights for inference

    inference_loop(
        inferenceDataFromPickle=inferenceData_full,
        outputDir=outputDir,
        device=device,
        parameters=parameters,
    )

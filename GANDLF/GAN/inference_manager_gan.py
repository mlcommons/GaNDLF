import os
from pathlib import Path
import pandas as pd
import torch
import torch.nn.functional as F

from GANDLF.GAN.compute import inference_loop_gans
from GANDLF.utils import get_unique_timestamp


def InferenceManagerGAN(
    dataframe, modelDir, parameters, device, outputDir=None
):
    """
    The main inference manager for GAN models. For now, it only generates
    the given number of images from the generator network. Dataframe is NOT
    used for now.
    Args:
        dataframe (pandas.DataFrame): The dataframe containing the data to be used for inference.
        modelDir (str): The path to the directory containing the model to be used for inference.
        outputDir (str): The path to the directory where the output of the inference will be stored.
        parameters (dict): The dictionary containing the parameters for the inference.
        device (str): The device type.

    Returns:
        None
    """
    # get the indeces for kfold splitting
    inferenceData_full = dataframe

    # if outputDir is not provided, create a new directory with a unique timestamp
    if outputDir is None:
        outputDir = os.path.join(modelDir, get_unique_timestamp())
        print(
            "Output directory not provided, creating a new directory with a unique timestamp: ",
            outputDir,
        )
    Path(outputDir).mkdir(parents=True, exist_ok=True)

    parameters["output_dir"] = outputDir

    # # initialize parameters for inference
    if not ("weights" in parameters):
        parameters["weights"] = None  # no need for loss weights for inference
    if not ("class_weights" in parameters):
        parameters["class_weights"] = (
            None  # no need for class weights for inference
        )

    # initialize model type for processing: if not defined, default to torch
    if "type" not in parameters["model"]:
        parameters["model"]["type"] = "torch"

    inference_loop_gans(
        dataframe=inferenceData_full,
        modelDir=modelDir,
        device=device,
        parameters=parameters,
        outputDir=outputDir,
    )

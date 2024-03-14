"""Inference loop for GANs. The inference is understood as the generation 
of images from the generator network. For now, user only provides the
number of images to generate. In the future, there can be an option to pass
the test data to compute metrics (also can be useful in conditional generation
or for explainability)."""

from .generic import create_pytorch_objects_gan, generate_latent_vector
import os
import SimpleITK as sitk

# hides torchio citation request, see https://github.com/fepegar/torchio/issues/235
os.environ["TORCHIO_HIDE_CITATION_PROMPT"] = "1"

import torch
from random import seed as random_seed
from pandas import DataFrame
import numpy as np
from tqdm import tqdm
from torch.cuda.amp import autocast
from GANDLF.utils import (
    best_model_path_end,
    latest_model_path_end,
)
from typing import Union
from warnings import warn


## TODO move this function into some utils file
def norm_range(t: torch.Tensor) -> None:
    """
    Normalizes the input tensor to be in the range [0, 1]. Operation is
    performed in place.

    Args:
        t (torch.Tensor): The input tensor to normalize.
    """

    def norm_ip(img: torch.Tensor, low: float, high: float) -> None:
        """
        Utility function to normalize the input image, the same as in
        torchvision. Operation is performed in place.

        Args:
            img (torch.Tensor): The image to normalize.
            low (float): The lower bound of the normalization.
            high (float): The upper bound of the normalization.
        """
        img.clamp_(min=low, max=high)
        img.sub_(low).div_(max(high - low, 1e-5))

    norm_ip(t, float(t.min()), float(t.max()))


def inference_loop_gans(
    dataframe: Union[DataFrame, None],
    device: str,
    parameters: dict,
    modelDir: str,
    outputDir: str = None,
):
    """
    The main inference loop for GAN models. For now, it only generates
    images from the generator network.

    Args:
        dataframe (pandas.DataFrame): The dataframe containing the data to be used for
    inference (NOT USED FOR NOW).
        device (str): The device to perform computations on.
        parameters (dict): The parameters dictionary.
        modelDir (str): The path to the directory containing the model to be
    used for inference.
        outputDir (str): The path to the directory where the output of the
    inference session will be stored.
    """
    # Defining our model here according to parameters mentioned in the configuration file
    print("Current model type : ", parameters["model"]["type"])
    print("Number of dims     : ", parameters["model"]["dimension"])
    if "num_channels" in parameters["model"]:
        print("Number of channels : ", parameters["model"]["num_channels"])
    # ensure outputs are saved properly
    parameters["save_output"] = True

    assert (
        parameters["model"]["type"].lower() == "torch"
    ), f"The model type is not recognized: {parameters['model']['type']}"
    assert (
        "save_format" in parameters["inference_config"]
    ), "The save format is not provided"
    assert (
        "n_generated_samples" in parameters["inference_config"]
    ), "The number of samples to generate is not provided"
    assert (
        "batch_size" in parameters["inference_config"]
    ), "The batch size for inference is not provided"
    assert parameters["inference_config"]["save_format"] in [
        "png",
        "jpeg",
        "jpg",
        "nii.gz",
    ], f"The save format is not recognized: {parameters['inference_config']['save_format']}"
    if dataframe is not None:
        warn(
            "The dataframe was passed, but it's usage is not yet implemented in inference. Ignoring it."
        )

    assert (
        parameters["model"]["type"].lower() != "openvino"
    ), "OpenVINO not yet implemented"

    pytorch_objects = create_pytorch_objects_gan(parameters, device=device)
    model, parameters = pytorch_objects[0], pytorch_objects[-1]
    main_dict = None

    if parameters["model"]["type"].lower() == "torch":
        # Loading the weights into the model
        if os.path.isdir(modelDir):
            files_to_check = [
                os.path.join(
                    modelDir,
                    str(parameters["model"]["architecture"])
                    + best_model_path_end,
                ),
                os.path.join(
                    modelDir,
                    str(parameters["model"]["architecture"])
                    + latest_model_path_end,
                ),
            ]

            file_to_load = None
            for best_file in files_to_check:
                if os.path.isfile(best_file):
                    file_to_load = best_file
                    break

            assert file_to_load != None, "The 'best_file' was not found"

        main_dict = torch.load(file_to_load, map_location=parameters["device"])
        model.load_state_dict(main_dict["model_state_dict"])
        parameters["previous_parameters"] = main_dict.get("parameters", None)
        model.eval()

    n_generated_samples = parameters["inference_config"]["n_generated_samples"]
    latent_vector_size = parameters["model"]["latent_vector_size"]
    batch_size = parameters["inference_config"]["batch_size"]
    # how many iterations to run
    n_iterations = (
        n_generated_samples // batch_size
    )  
    # remaining samples for last iteration
    remaining_samples = (
        n_generated_samples % batch_size
    )  

    print(
        f"Running {n_iterations} generator iterations to generate {n_generated_samples} samples with batch size {batch_size}.",
        flush=True,
    )

    if os.environ.get("HOSTNAME") is not None:
        print("\nHostname     :" + str(os.environ.get("HOSTNAME")), flush=True)

    if not os.path.exists(outputDir):
        os.makedirs(outputDir)
    # set the random seeds for reproducibility if provided
    if "seed" in parameters:
        print(f"Setting random seed to {parameters['seed']}", flush=True)
        torch.manual_seed(parameters["seed"])
        np.random.seed(parameters["seed"])
        random_seed(parameters["seed"])
    file_extension = parameters["inference_config"]["save_format"]
    for iteration in tqdm(range(n_iterations)):
        with torch.no_grad():
            latent_vector = generate_latent_vector(
                batch_size if iteration < n_iterations else remaining_samples,
                latent_vector_size,
                parameters["model"]["dimension"],
                device,
            )
            if parameters["model"]["amp"]:
                with autocast():
                    generated_images = model(latent_vector)
            else:
                generated_images = model(latent_vector)
            generated_images = generated_images.cpu()
            norm_range(generated_images)
            generated_images = generated_images * 255
            if parameters["model"]["dimension"] == 2:
                generated_images = generated_images.permute(0, 2, 3, 1)
                generated_images = generated_images.numpy().astype(np.uint8)
            elif parameters["model"]["dimension"] == 3:
                generated_images = generated_images.permute(0, 2, 3, 4, 1)
            for i in range(generated_images.shape[0]):
                image_to_save = generated_images[i]
                save_path = os.path.join(
                    outputDir,
                    f"batch_num_{iteration}_image_{i}.{file_extension}",
                )

                if parameters["model"]["dimension"] == 2:
                    sitk.WriteImage(
                        sitk.GetImageFromArray(
                            image_to_save,
                            isVector=True,
                        ),
                        save_path,
                    )
                elif parameters["model"]["dimension"] == 3:
                    sitk.WriteImage(
                        sitk.GetImageFromArray(image_to_save),
                        save_path,
                    )

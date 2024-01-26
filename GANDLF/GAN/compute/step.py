import torch
import psutil
from typing import Dict, Union, Tuple
from .loss_and_metric import get_loss_and_metrics_gans
from GANDLF.models.modelBase import ModelBase


def step_gan(
    gan_model: ModelBase,
    primary_images: torch.Tensor,
    label: Union[torch.Tensor, None],
    params: Dict,
    secondary_images: Union[torch.Tensor, None] = None,
    train: bool = True,
) -> Tuple[
    Union[torch.Tensor, None],
    Union[torch.Tensor, None],
    torch.Tensor,
    Union[torch.Tensor, None],
]:
    """
    Function that steps the GAN model for a single batch

    Parameters
    ----------
    gan_model : ModelBase derived class
        The model to process the input image with, it should support
    appropriate dimensions.
    image : torch.Tensor
        The input image stack according to requirements (can be latent
    vector for generator).
    label : torch.Tensor or None
        The input label for the corresponding image label. For generator
    step, this should be None.
    params : dict
        The parameters passed by the user yaml.
    train : bool
        Whether the model is in training mode.

    Returns
    -------
    loss : torch.Tensor or None
        The computed loss from the label and the output.
    metric_output : torch.Tensor or None
        The computed metric from the label and the output.
    output: torch.Tensor
        The final output of the model.
    attention_map: torch.Tensor or None
        The attention map for the output, if available.

    """
    if params["verbose"]:
        if torch.cuda.is_available():
            print(torch.cuda.memory_summary())
        print(
            "|===========================================================================|"
        )
        print(
            "|                              CPU Utilization                              |"
        )
        print("Load_Percent   :", psutil.cpu_percent(interval=None))
        print("MemUtil_Percent:", psutil.virtual_memory()[2])
        print(
            "|===========================================================================|"
        )

    sub_model = gan_model.discriminator

    if params["model"]["dimension"] == 2:
        image = torch.squeeze(primary_images, -1)

    if not (train) and params["model"]["type"].lower() == "openvino":
        output = torch.from_numpy(
            sub_model(
                inputs={params["model"]["IO"][0][0]: image.cpu().numpy()}
            )[params["model"]["IO"][1][0]]
        )
        output = output.to(params["device"])
    else:
        if params["model"]["amp"]:
            with torch.cuda.amp.autocast():
                output = sub_model(image)
        else:
            output = sub_model(image)

    attention_map = None
    if "medcam_enabled" in params and params["medcam_enabled"]:
        output, attention_map = output
    loss, metric_output = get_loss_and_metrics_gans(
        image, secondary_images, label, output, params
    )

    if params["model"]["dimension"] == 2:
        output = torch.unsqueeze(output, -1)
        if "medcam_enabled" in params and params["medcam_enabled"]:
            attention_map = torch.unsqueeze(attention_map, -1)

    return loss, metric_output, output, attention_map

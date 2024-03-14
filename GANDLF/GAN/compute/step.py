import torch
import psutil
from typing import Union, Tuple, Optional
from .loss_and_metric import get_loss_and_metrics_gans


def step_gan(
    gan_model: torch.nn.Module,
    primary_images: torch.Tensor,
    label: Optional[torch.Tensor],
    params: dict,
    secondary_images: Optional[torch.Tensor],
    train: bool = True,
) -> Tuple[
    torch.Tensor,
    Union[torch.Tensor, None],
    torch.Tensor,
    Union[torch.Tensor, None],
]:
    """
    Function that steps the GAN model for a single batch.

    Args:
        gan_model (torch.nn.Module): The GAN model to process the input
    image with, it should support appropriate dimensions.
        primary_images (torch.Tensor): The input image stack according to
    requirements (can be latent vector for generator).
        label (Optional[torch.Tensor]): The input label for the corresponding
    image label (fake or real). When doing validation or inference, this can
    be None.
        params (dict): The parameters passed by the user yaml.
        secondary_images (Optional[torch.Tensor]): The input secondary image
    stack used only when computing metrics.
        train (bool, optional): Whether the model is in training mode.
    Defaults to True.

    Returns:
        loss (torch.Tensor): The computed loss from the label and the output.
        metric_output (Union[torch.Tensor, None]): The computed metric from
    the label and the output. Available only if secondary_images is not None.
        output (torch.Tensor): The final output of the model.
        attention_map (Union[torch.Tensor, None]): The attention map for the
    output, if available.

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
    else:
        image = primary_images

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

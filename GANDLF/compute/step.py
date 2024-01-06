import torch
import psutil
import warnings
from typing import Dict, Any, Union, Tuple
from .loss_and_metric import get_loss_and_metrics
from models.modelBase import ModelBase


def step(model, image, label, params, train=True):
    """
    Function that steps the model for a single batch

    Parameters
    ----------
    model : torch.model
        The model to process the input image with, it should support appropriate dimensions.
    image : torch.Tensor
        The input image stack according to requirements
    label : torch.Tensor
        The input label for the corresponding image label
    params : dict
        The parameters passed by the user yaml

    Returns
    -------
    loss : torch.Tensor
        The computed loss from the label and the output
    metric_output : torch.Tensor
        The computed metric from the label and the output
    output: torch.Tensor
        The final output of the model

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

    # for the weird cases where mask is read as an RGB image, ensure only the first channel is used
    if label is not None:
        if params["problem_type"] == "segmentation":
            if label.shape[1] == 3:
                label = label[:, 0, ...].unsqueeze(1)
                # this warning should only come up once
                if params["print_rgb_label_warning"]:
                    print(
                        "WARNING: The label image is an RGB image, only the first channel will be used.",
                        flush=True,
                    )
                    params["print_rgb_label_warning"] = False

            if params["model"]["dimension"] == 2:
                label = torch.squeeze(label, -1)

    if params["model"]["dimension"] == 2:
        image = torch.squeeze(image, -1)
        if "value_keys" in params:
            if label is not None:
                if len(label.shape) > 1:
                    label = torch.squeeze(label, -1)

    if not (train) and params["model"]["type"].lower() == "openvino":
        output = torch.from_numpy(
            model(inputs={params["model"]["IO"][0][0]: image.cpu().numpy()})[
                params["model"]["IO"][1][0]
            ]
        )
        output = output.to(params["device"])
    else:
        if params["model"]["amp"]:
            with torch.cuda.amp.autocast():
                output = model(image)
        else:
            output = model(image)

    attention_map = None
    if "medcam_enabled" in params and params["medcam_enabled"]:
        output, attention_map = output

    # one-hot encoding of 'label' will probably be needed for segmentation
    if label is not None:
        loss, metric_output = get_loss_and_metrics(
            image, label, output, params
        )
    else:
        loss, metric_output = None, None

    if len(output) > 1:
        output = output[0]

    if params["model"]["dimension"] == 2:
        output = torch.unsqueeze(output, -1)
        if "medcam_enabled" in params and params["medcam_enabled"]:
            attention_map = torch.unsqueeze(attention_map, -1)

    return loss, metric_output, output, attention_map


def step_gan(
    gan_model: ModelBase,
    image: torch.Tensor,
    label: Union[torch.Tensor, None],
    params: Dict,
    step_discriminator: bool = True,
    step_generator: bool = False,
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
    gan_model : BaseModel derived class
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
    step_discriminator : bool
        Whether step is made on a discriminator submodel.
    step_generator : bool
        Whether step is made on a generator submodel.
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

    if step_discriminator and step_generator:
        raise ValueError(
            "Both discriminator and generator cannot be stepped at the same time."
        )
    if not (step_discriminator or step_generator):
        raise ValueError(
            "Either discriminator or generator must be stepped at the same time."
        )
    if step_discriminator and label is None:
        raise ValueError("Label must be provided for discriminator step.")
    if step_generator and label is not None:
        warnings.warn(
            "Label is provided for generator step. This label will be ignored.",
            UserWarning,
        )

    sub_model = (
        gan_model.discriminator if step_discriminator else gan_model.generator
    )
    if params["model"]["dimension"] == 2:
        image = torch.squeeze(image, -1)
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
    if step_discriminator:  # TODO check if the metrics are suitable
        loss, metric_output = get_loss_and_metrics(
            image, label, output, params
        )
    else:
        loss, metric_output = None, None

    if params["model"]["dimension"] == 2:
        output = torch.unsqueeze(output, -1)
        if "medcam_enabled" in params and params["medcam_enabled"]:
            attention_map = torch.unsqueeze(attention_map, -1)

    return loss, metric_output, output, attention_map

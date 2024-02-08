import torch

from ..utils.generic import print_system_utilization
from ..utils.imaging import (
    adjust_dimensions,
    adjust_output_dimensions,
    preprocess_label_for_segmentation,
)
from .loss_and_metric import get_loss_and_metrics


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
        print_system_utilization()

    # Handle label preprocessing for segmentation problems
    if label is not None and params["problem_type"] == "segmentation":
        label = preprocess_label_for_segmentation(label, params)

    # Adjust image and label dimensions if necessary
    if params["model"]["dimension"] == 2:
        image, label = adjust_dimensions(image, label, params)

    # Get model output
    output, attention_map = model_forward_pass(model, image, params, train)

    attention_map = None
    if "medcam_enabled" in params and params["medcam_enabled"]:
        output, attention_map = output

    # Compute loss and metrics if label is provided
    loss, metric_output = (
        (None, None)
        if label is None
        else get_loss_and_metrics(image, label, output, params)
    )

    # Adjust output dimensions if necessary
    if params["model"]["dimension"] == 2:
        output, attention_map = adjust_output_dimensions(output, attention_map, params)

    return loss, metric_output, output, attention_map


def model_forward_pass(model, image, params, train=True):
    """
    Perform a forward pass through the model.

    Parameters
    ----------

    model : torch.model
        The model to process the input image with, it should support appropriate dimensions.
    image : torch.Tensor
        The input image stack according to requirements
    params : dict
        The parameters passed by the user yaml
    train : bool
        Whether the model is being trained or not

    Returns
    -------
    output : torch.Tensor
        The output of the model
    """
    if not train and params["model"]["type"].lower() == "openvino":
        output = torch.from_numpy(
            model(inputs={params["model"]["IO"][0][0]: image.cpu().numpy()})[
                params["model"]["IO"][1][0]
            ]
        )
        output = output.to(params["device"])
    elif params["model"].get("amp", False):
        with torch.cuda.amp.autocast():
            output = model(image)
    else:
        output = model(image)

    attention_map = None
    if "medcam_enabled" in params and params["medcam_enabled"]:
        output, attention_map = output

    if isinstance(output, tuple) and len(output) > 1:  # Handling multiple outputs
        output = output[0]

    return output, attention_map

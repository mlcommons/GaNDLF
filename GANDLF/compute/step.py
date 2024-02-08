import torch

from ..utils.generic import print_system_utilization
from ..utils.imaging import preprocess_label_for_segmentation
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
        loss, metric_output = get_loss_and_metrics(image, label, output, params)
    else:
        loss, metric_output = None, None

    if len(output) > 1:
        output = output[0]

    if params["model"]["dimension"] == 2:
        output = torch.unsqueeze(output, -1)
        if "medcam_enabled" in params and params["medcam_enabled"]:
            attention_map = torch.unsqueeze(attention_map, -1)

    return loss, metric_output, output, attention_map

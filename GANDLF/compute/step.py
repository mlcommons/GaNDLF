from typing import Optional, Tuple
import torch
import psutil
import logging
from .loss_and_metric import get_loss_and_metrics
from GANDLF.utils import setup_logger


def step(
    model: torch.nn.Module,
    image: torch.Tensor,
    label: torch.Tensor,
    params: dict,
    train: Optional[bool] = True,
) -> Tuple[float, dict, torch.Tensor, torch.Tensor]:
    """
    This function performs a single step of training or validation.

    Args:
        model (torch.nn.Module): The model to process the input image with, it should support appropriate dimensions.
        image (torch.Tensor): The input image stack according to requirements.
        label (torch.Tensor): The input label for the corresponding image tensor.
        params (dict): The parameters dictionary.
        train (Optional[bool], optional): Whether the step is for training or validation. Defaults to True.

    Returns:
        Tuple[float, dict, torch.Tensor, torch.Tensor]: The loss, metrics, output, and attention map.
    """

    if "logger_name" in params:
        logger = logging.getLogger(params["logger_name"])
    else:
        logger, params["logs_dir"], params["logger_name"] = setup_logger(
            output_dir=params["output_dir"], 
            verbose=params.get("verbose", False),
        )


    if torch.cuda.is_available():
        logger.debug(torch.cuda.memory_summary())
    logger.debug(
        f"""\n
            |===========================================================================| \n
            |                              CPU Utilization                              | \n
            | Load_Percent   : {psutil.cpu_percent(interval=None)}                      | \n
            | MemUtil_Percent: {psutil.virtual_memory()[2]}                             | \n
            |===========================================================================|"""
    )

    # for the weird cases where mask is read as an RGB image, ensure only the first channel is used
    if label is not None:
        if params["problem_type"] == "segmentation":
            if label.shape[1] == 3:
                label = label[:, 0, ...].unsqueeze(1)
                # this warning should only come up once
                if params["print_rgb_label_warning"]:
                    logger.warning(
                        "The label image is an RGB image, only the first channel will be used."
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

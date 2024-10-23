import warnings
from typing import Optional, Tuple, Union
import torch
import psutil
from .loss_and_metric import get_loss_and_metrics


def step(
    model: torch.nn.Module,
    image: torch.Tensor,
    label: Optional[torch.Tensor],
    params: dict,
    train: Optional[bool] = True,
) -> Tuple[float, dict, Union[torch.Tensor, list[torch.Tensor]], torch.Tensor]:
    """
    This function performs a single step of training or validation.

    Args:
        model (torch.nn.Module): The model to process the input image with, it should support appropriate dimensions.
        image (torch.Tensor): The input image stack according to requirements. (B, C, H, W, D)
        label Optional[torch.Tensor]: The input label for the corresponding image tensor.
            If segmentation, (B, C, H, W, D);
            if classification / regression (not multilabel), (B, 1)
            if classif / reg (multilabel), (B, N_LABELS)

        params (dict): The parameters dictionary.
        train (Optional[bool], optional): Whether the step is for training or validation. Defaults to True.

    Returns:
        Tuple[float, dict, Union[torch.Tensor, list[torch.Tensor]], torch.Tensor]: The loss, metrics, output,
            and attention map.
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
                warnings.warn(
                    "The label image is an RGB image, only the first channel will be used."
                )

        assert len(label) == len(image)

    if params["model"]["dimension"] == 2:
        image = image.squeeze(-1)  # removing depth

    # for segmentation remove the depth dimension from the label.
    # for classification / regression, flattens class / reg label from list (possible in multilabel) to scalar
    # TODO: second condition is crutch - in some cases label is passed as 1-d Tensor (B,) and if Batch size is 1,
    #  it is squeezed to scalar tensor (0-d) and the future logic fails
    if label is not None and len(label.shape) != 1:
        label = label.squeeze(-1)

    if not train and params["model"]["type"].lower() == "openvino":
        output = torch.from_numpy(
            model(inputs={params["model"]["IO"][0][0]: image.cpu().numpy()})[
                params["model"]["IO"][1][0]
            ]
        )
        output = output.to(params["device"])
    elif params["model"]["amp"]:
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

    if params["model"]["dimension"] == 2:
        if "medcam_enabled" in params and params["medcam_enabled"]:
            attention_map = torch.unsqueeze(attention_map, -1)

    if not isinstance(output, torch.Tensor):
        warnings.warn(
            f"Model output is not a Tensor: {type(output)}. Say, `deep_resunet` and `deep_unet` may return "
            f"list of tensors on different scales instead of just one prediction Tensor. However due to "
            f"GaNDLF architecture it is expected that models return only one tensor. For deep_* models "
            f"only the biggest scale is processed. Use these models with caution till fix is implemented."
        )
        output = output[0]

    if params["model"]["dimension"] == 2 and params["problem_type"] == "segmentation":
        # for 2d images where the depth is removed, add it back
        output = output.unsqueeze(-1)

    assert len(output) == len(
        image
    ), f"Error: output({len(output)}) and batch({len(image)}) have different lengths. Both should be equal to batch size!"
    return loss, metric_output, output, attention_map

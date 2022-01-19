import torch
import psutil
from GANDLF.compute.loss_and_metric import get_loss_and_metrics


def step_ov(model, image, label, params):
    """
    Function that steps the model for a single batch

    Parameters
    ----------
    model : if params["model"]["type"] = "Torch" this is a torch.model otherwise this is OV model
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
        if params["device"] == 'gpu':
            print(torch.cuda.memory_summary())
            
        print(
            "|===========================================================================|\n|                             CPU Utilization                            |\n|"
        )
        print("Load_Percent   :", psutil.cpu_percent(interval=None))
        print("MemUtil_Percent:", psutil.virtual_memory()[2])
        print(
            "|===========================================================================|\n|"
        )

    if params["model"]["dimension"] == 2:
        image = torch.squeeze(image, -1)
        if "value_keys" in params:  # squeeze label for segmentation only
            if len(label.shape) > 1:
                label = torch.squeeze(label, -1)
    if params["model"]["type"] == "Torch":
        if params["model"]["amp"]:
            with torch.cuda.amp.autocast():
                output = model(image)
        else:
            output = model(image)
    else:
        output = torch.from_numpy(model.infer( inputs={params["model"]["IO"][0]:image.cpu().numpy()})[params["model"]["IO"][1]])

    if "medcam_enabled" in params and params["medcam_enabled"]:
        output, attention_map = output

    # one-hot encoding of 'label' will be needed for segmentation
    loss, metric_output = get_loss_and_metrics(image, label, output, params)

    if len(output) > 1:
        output = output[0]

    if params["model"]["dimension"] == 2:
        output = torch.unsqueeze(output, -1)
        if "medcam_enabled" in params and params["medcam_enabled"]:
            attention_map = torch.unsqueeze(attention_map, -1)

    if not ("medcam_enabled" in params and params["medcam_enabled"]):
        return loss, metric_output, output
    else:
        return loss, metric_output, output, attention_map

import torch
import psutil
from .loss_and_metric import get_loss_and_metrics


def step(model_main, image, label, params):
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
        print(torch.cuda.memory_summary())
        print(
            "|===========================================================================|\n|                             CPU Utilization                            |\n|"
        )
        print("Load_Percent   :", psutil.cpu_percent(interval=None))
        print("MemUtil_Percent:", psutil.virtual_memory()[2])
        print(
            "|===========================================================================|\n|"
        )

    # for the weird cases where mask is read as an RGB image, ensure only the first channel is used
    if params["problem_type"] == "segmentation":
        if label.shape[1] == 3:
            #label = label[:, 0, ...].unsqueeze(1)
            #Not necessary for GANs
            # this warning should only come up once
            if params["print_rgb_label_warning"]:
                print(
                    "WARNING: The output image is an RGB image.",
                    flush=True,
                )
                params["print_rgb_label_warning"] = False

        if params["model"]["dimension"] == 2:
            label = torch.squeeze(label, -1)

    if params["model"]["dimension"] == 2:
        image = torch.squeeze(image, -1)
        label = torch.squeeze(label, -1)
    
    try:
        style_to_style = params["style_to_style"]
        if style_to_style==False:
            style=False
        else:
            style=True
    except:
        style=True
    if style == False:
        image = torch.zeros(image.shape[0], params["latent_dim"]).normal_(0, 1)
    else:
        image,label=model_main.preprocess(image,label)

    model=model_main.return_generator()
    #image=image/255.0
    #label=label.float()
    #label=label/255.0

    if params["model"]["amp"]:
        with torch.cuda.amp.autocast():
            output = model(image)
    else:
        output = model(image)
    if "medcam_enabled" in params and params["medcam_enabled"]:
        output, attention_map = output
  
    # one-hot encoding of 'label' will probably be needed for segmentation
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
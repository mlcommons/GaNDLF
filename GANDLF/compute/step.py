import torch
import psutil
from .loss_and_metric import get_loss_and_metrics, get_gan_loss

def step_discriminator(generator, discriminator, image_real, params, image=None):
    """
    Function that steps the model for a single batch

    Parameters
    ----------
    generator : torch.model
        The model to process the input image with, it should support appropriate dimensions.
    discriminator : torch.model
        The model to process the input image with, it should support appropriate dimensions.
    image_real : torch.Tensor
        The input image stack according to requirements
    image_fake : torch.Tensor
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

    # create real/fake labels
    real = torch.ones(image_real.shape[0], 1).to(image_real.device)
    fake = torch.zeros(image_real.shape[0], 1).to(image_real.device)

    if params["model"]["amp"]:
        with torch.cuda.amp.autocast():
            output_real = discriminator(image_real)
            noise = torch.zeros(image_real.shape[0], params["latent_dim"]).normal_(0, 1)
            image_fake = generator(noise)
            output_fake = discriminator(image_fake)

    else:
        output_real = discriminator(image_real)
        noise = torch.zeros(image_real.shape[0], params["latent_dim"]).normal_(0, 1)
        image_fake = generator(noise)
        output_fake = discriminator(image_fake)

    dloss_real = get_gan_loss(output_real, real, params)
    dloss_fake = get_gan_loss(output_fake, fake, params)
    dloss = dloss_real + dloss_fake

    return dloss, image_fake


def step_generator(discriminator, image_fake, params):
    """
    Function that steps the model for a single batch

    Parameters
    ----------
    discriminator : torch.model
        The model to process the input image with, it should support appropriate dimensions.
    image_fake : torch.Tensor
        The generated image
    params : dict
        The parameters passed by the user yaml

    Returns
    -------
    loss : torch.Tensor
        The computed loss from the label and the output

    """

    # create real/fake labels
    real = torch.ones(image_fake.shape[0], 1).to(image_fake.device)

    if params["model"]["amp"]:
        with torch.cuda.amp.autocast():
            output_fake = discriminator(image_fake)

    else:
        output_fake = discriminator(image_fake)

    # Measures the generator's ability to fool the discriminator
    gloss = get_gan_loss(output_fake, real, params)

    return gloss


def step(model, image, label, params):
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
            if len(label.shape) > 1:
                label = torch.squeeze(label, -1)

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

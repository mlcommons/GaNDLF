import os
import pathlib
import numpy as np
import SimpleITK as sitk
import torch
import torchio
from GANDLF.GAN.compute.step import step_gan
from GANDLF.utils import (
    get_date_time,
    get_filename_extension_sanitized,
    resample_image,
)
from tqdm import tqdm
from typing import Tuple, Optional
from .generic import get_fixed_latent_vector, generate_latent_vector
from warnings import warn
import torchvision.utils as vutils
from torch.utils.data import DataLoader


def norm_range(t: torch.Tensor):
    """Normalizes the input tensor to be in the range [0, 1]. Operation is
    performed in place.
    Args:
        t (torch.Tensor): The input tensor to normalize.
    """

    def norm_ip(img: torch.Tensor, low: float, high: float):
        """Utility function to normalize the input image, the same as in
        torchvision. Operation is performed in place.
        Args:
            img (torch.Tensor): The image to normalize.
            low (float): The lower bound of the normalization.
            high (float): The upper bound of the normalization.
        """
        img.clamp_(min=low, max=high)
        img.sub_(low).div_(max(high - low, 1e-5))

    norm_ip(t, float(t.min()), float(t.max()))


def validate_network_gan(
    model: torch.nn.Module,
    valid_dataloader: DataLoader,
    scheduler_d: object,
    scheduler_g: object,
    params: dict,
    epoch: Optional[int] = 0,
    mode: Optional[str] = "validation",
) -> Tuple[float, float, dict]:
    """
    Function to validate the network for a single epoch for GANs.

    Args:
        model (torch.nn.Module): The model to process the input image with,
    it should support appropriate dimensions. if parameters["model"]["type"] == torch,
    this is a torch.model, otherwise this is OV exec_net.
        valid_dataloader (torch.utils.data.DataLoader): The dataloader to use.
        scheduler_d (object): The scheduler for the discriminator.
        scheduler_g (object): The scheduler for the generator.
        params (dict): The parameters for the run.
        epoch (int, optional): The epoch number. Defaults to 0.
        mode (str, optional): The mode of operation, either 'validation' or 'inference'.
    Defaults to 'validation'.

    Returns:
        Tuple[float, float, dict]: The average loss for the generator, the average
    loss for the discriminator, and the average metrics for the epoch.
    """
    assert mode in [
        "validation",
        "inference",
    ], "Mode should be 'validation' or 'inference' "

    print("*" * 20)
    print("Starting " + mode + " : ")
    print("*" * 20)
    total_epoch_discriminator_fake_loss = 0.0
    total_epoch_discriminator_real_loss = 0.0
    total_epoch_metrics = {}
    for metric in params["metrics"]:
        total_epoch_metrics[metric] = 0.0
    is_inference = mode == "inference"
    if params["verbose"]:
        if params["model"]["amp"]:
            print("Using Automatic mixed precision", flush=True)
    if scheduler_d is None or scheduler_g is None:
        current_output_dir = params["output_dir"]  # this is in inference mode
    else:  # this is useful for inference
        current_output_dir = os.path.join(params["output_dir"], "output_" + mode)
    pathlib.Path(current_output_dir).mkdir(parents=True, exist_ok=True)

    if ((scheduler_d is None) and (scheduler_g is None)) or is_inference:
        current_output_dir = params["output_dir"]
    else:
        current_output_dir = os.path.join(params["output_dir"], "output_" + mode)

    if not is_inference:
        current_output_dir = os.path.join(current_output_dir, str(epoch))

    # Set the model to valid
    if params["model"]["type"] == "torch":
        model.eval()

    for batch_idx, (subject) in enumerate(
        tqdm(valid_dataloader, desc="Looping over " + mode + " data")
    ):
        if params["verbose"]:
            print("== Current subject:", subject["subject_id"], flush=True)

        # ensure spacing is always present in params and is always subject-specific
        params["subject_spacing"] = None
        if "spacing" in subject:
            params["subject_spacing"] = subject["spacing"]

        # constructing a new dict because torchio.GridSampler requires torchio.Subject,
        # which requires torchio.Image to be present in initial dict, which the loader does not provide
        subject_dict = {}

        for key in params["channel_keys"]:
            subject_dict[key] = torchio.ScalarImage(
                path=subject[key]["path"],
                tensor=subject[key]["data"].squeeze(0),
                affine=subject[key]["affine"].squeeze(0),
            )

        grid_sampler = torchio.inference.GridSampler(
            torchio.Subject(subject_dict),
            params["patch_size"],
            patch_overlap=params["inference_mechanism"]["patch_overlap"],
        )
        # Caution - now, only full image validation is supported
        patch_loader = torch.utils.data.DataLoader(grid_sampler, batch_size=1)
        # aggregator = torchio.inference.GridAggregator(
        #     grid_sampler,
        #     overlap_mode=params["inference_mechanism"][
        #         "grid_aggregator_overlap"
        #     ],
        # )
        # TODO Do we need medcam in this case? I don't think so
        # if params["medcam_enabled"]:
        #     attention_map_aggregator = torchio.inference.GridAggregator(
        #         grid_sampler,
        #         overlap_mode=params["inference_mechanism"][
        #             "grid_aggregator_overlap"
        #         ],
        #     )

        current_patch = 0
        for patches_batch in patch_loader:
            if params["verbose"]:
                print(
                    "=== Current patch:",
                    current_patch,
                    ", time : ",
                    get_date_time(),
                    ", location :",
                    patches_batch[torchio.LOCATION],
                    flush=True,
                )
            current_patch += 1
            image = (
                torch.cat(
                    [
                        patches_batch[key][torchio.DATA]
                        for key in params["channel_keys"]
                    ],
                    dim=1,
                )
                .float()
                .to(params["device"])
            )
            current_batch_size = image.shape[0]

            label_real = torch.full(
                size=(current_batch_size,),
                fill_value=1,
                dtype=torch.float,
                device=params["device"],
            )
            label_fake = torch.full(
                size=(current_batch_size,),
                fill_value=0,
                dtype=torch.float,
                device=params["device"],
            )
            latent_vector = generate_latent_vector(
                current_batch_size,
                params["model"]["latent_vector_size"],
                params["model"]["dimension"],
                params["device"],
            )
            with torch.no_grad():
                fake_images = model.generator(latent_vector)
                loss_disc_fake, _, output_disc_fake, _ = step_gan(
                    model,
                    fake_images,
                    label_fake,
                    params,
                    secondary_images=None,
                )
                # here, we are iterating over the patches
                loss_disc_real, metrics_output, output_disc_real, _ = step_gan(
                    model,
                    image,
                    label_real,
                    params,
                    secondary_images=fake_images,
                )

            for metric in params["metrics"]:
                total_epoch_metrics[metric] += metrics_output[metric] / len(
                    patch_loader
                )
            # accumulating average loss for the real and fake images over the patch loader

            total_epoch_discriminator_real_loss += loss_disc_real.cpu().item() / len(
                patch_loader
            )
            total_epoch_discriminator_fake_loss += loss_disc_fake.cpu().item() / len(
                patch_loader
            )

    average_epoch_metrics = {
        metric_name: total_epoch_metrics[metric_name] / len(valid_dataloader)
        for metric_name in total_epoch_metrics
    }
    # average the losses over all validation batches
    total_epoch_discriminator_fake_loss /= len(valid_dataloader)
    total_epoch_discriminator_real_loss /= len(valid_dataloader)
    # TODO Aggregator is currently not used and invalid - no way
    # to get the generator to output spatially consistent results, to be implemented

    # aggregator.add_batch(fake_images.cpu(), patches_batch[torchio.LOCATION])
    # TODO do we use medcam in this case ever?
    # if params["medcam_enabled"]:
    #     _, _, output, attention_map = result
    #     attention_map_aggregator.add_batch(
    #         attention_map, patches_batch[torchio.LOCATION]
    #     )
    # output_prediction = aggregator.get_output_tensor()
    # output_prediction = output_prediction.unsqueeze(0)

    if params["save_output"]:
        img_for_metadata = torchio.ScalarImage(
            tensor=subject["1"]["data"].squeeze(-1, 1),
            affine=subject["1"]["affine"].squeeze(0),
        ).as_sitk()
        ext = get_filename_extension_sanitized(subject["1"]["path"][0])
        ## TODO dirty bypass - for some reason, the extension is empty
        if ext == "":
            ext = ".png"

        fixed_latent_vector = get_fixed_latent_vector(
            batch_size=params["validation_config"]["n_generated_samples"],
            latent_vector_size=params["model"]["latent_vector_size"],
            dimension=params["model"]["dimension"],
            device=params["device"],
            seed=params.get("seed", 0),
        )

        with torch.no_grad():
            fake_images_to_save = model.generator(fixed_latent_vector).cpu()
        if params["save_grid"] and params["model"]["dimension"] == 2:
            fake_images_tensor = fake_images_to_save.clone()
        # TODO - think if this is correct or no
        norm_range(fake_images_to_save)
        fake_images_to_save *= 255
        if params["model"]["dimension"] == 2:
            fake_images_to_save = fake_images_to_save.permute(0, 2, 3, 1).numpy()
        else:
            fake_images_to_save = fake_images_to_save.permute(0, 2, 3, 4, 1).numpy()
        if ext in [
            ".jpg",
            ".jpeg",
            ".png",
        ]:
            # for optional later save as grid
            # fake_images_tensor = fake_images_to_save.clone()
            # Rescale the images 0 - 1. Think if this is the best way to do it
            fake_images_to_save = fake_images_to_save.astype(np.uint8)
        is_2d_rgb = (
            params["model"]["dimension"] == 2 and fake_images_to_save.shape[-1] == 3
        )
        for i, fake_image_to_save in enumerate(fake_images_to_save):
            if is_2d_rgb:
                result_image = sitk.GetImageFromArray(fake_image_to_save, isVector=True)
            else:
                result_image = sitk.GetImageFromArray(fake_image_to_save)
            # TODO - think about proper metadata handling, for now
            # it gives an error
            # result_image.CopyInformation(img_for_metadata)

            # this handles cases that need resampling/resizing
            if "resample" in params["data_preprocessing"]:
                result_image = resample_image(
                    result_image,
                    img_for_metadata.GetSpacing(),
                    interpolator=sitk.sitkNearestNeighbor,
                )

            # Create the subject directory if it doesn't exist in the

            os.makedirs(
                os.path.join(
                    current_output_dir,
                    "testing",
                    subject["subject_id"][0],
                ),
                exist_ok=True,
            )

            path_to_save = os.path.join(
                current_output_dir,
                "testing",
                subject["subject_id"][0],
                subject["subject_id"][0] + f"_gen_{i}" + ext,
            )

            sitk.WriteImage(
                result_image,
                path_to_save,
            )
        # for convenience, user can save the grid of images as well
        if params["save_grid"] and params["model"]["dimension"] == 2:
            os.makedirs(
                os.path.join(current_output_dir, "testing"),
                exist_ok=True,
            )
            os.makedirs(
                os.path.join(
                    current_output_dir,
                    "testing",
                    subject["subject_id"][0],
                ),
                exist_ok=True,
            )

            path_to_save = os.path.join(
                current_output_dir,
                "testing",
                subject["subject_id"][0],
                subject["subject_id"][0] + f"array" + ext,
            )

            vutils.save_image(
                fake_images_tensor,
                path_to_save,
                normalize=True,
                scale_each=False,
            )
        elif params["save_grid"] and params["model"]["dimension"] == 3:
            warn("Cannot save grid for 3D images, this step will be omitted.")
    if scheduler_d is not None:
        assert params["scheduler_d"]["type"] not in [
            "reduce_on_plateau",
            "reduce-on-plateau",
            "plateau",
            "reduceonplateau",
        ], "Reduce on plateau scheduler not implemented for GAN, but passed for discriminator"

        scheduler_d.step()
    if scheduler_g is not None:
        assert params["scheduler_g"]["type"] not in [
            "reduce_on_plateau",
            "reduce-on-plateau",
            "plateau",
            "reduceonplateau",
        ], "Reduce on plateau scheduler not implemented for GAN, but passed for generator"
        scheduler_g.step()
    return (
        total_epoch_discriminator_fake_loss,
        total_epoch_discriminator_real_loss,
        average_epoch_metrics,
    )

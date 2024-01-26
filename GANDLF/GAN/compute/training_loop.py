import os, time, psutil
import torch
from tqdm import tqdm
import numpy as np
import torchio
from medcam import medcam
from pandas import DataFrame
from GANDLF.data import get_testing_loader
from GANDLF.grad_clipping.grad_scaler import (
    GradScaler,
    model_parameters_exclude_head,
)
from GANDLF.grad_clipping.clip_gradients import dispatch_clip_grad_
from GANDLF.utils import (
    get_date_time,
    best_model_path_end,
    latest_model_path_end,
    initial_model_path_end,
    save_model,
    optimize_and_save_model,
    load_model,
    version_check,
    write_training_patches,
    print_model_summary,
    get_ground_truths_and_predictions_tensor,
    get_model_dict,
    print_and_format_metrics,
)
from GANDLF.metrics import overall_stats
from GANDLF.logger import LoggerGAN
from .step import step_gan
from .forward_pass import validate_network
from .generic import create_pytorch_objects_gan
from typing import Union
from pathlib import Path


def train_network_gan(
    model, train_dataloader, optimizer_g, optimizer_d, params
):
    """
    Function to train a GAN network for a single epoch.
    This function is a modified version of train_network() to support
    usage of two optimizers for the generator and discriminator.

    Parameters
    ----------
    model : torch.model
        The model to process the input image with, it should support appropriate dimensions.
    train_dataloader : torch.DataLoader
        The dataloader for the training epoch
    optimizer_g : torch.optim
        Optimizer for optimizing generator network
    optimizer_d : torch.optim
        Optimizer for optimizing discriminator network
    params : dict
        the parameters passed by the user yaml

    Returns
    -------
    average_epoch_train_loss_gen : float
        Train loss for the current epoch for generator
    average_epoch_train_loss_disc : float
        Train loss for the current epoch for discriminator
    average_epoch_train_metric : dict
        Train metrics for the current epoch
    """

    print("*" * 20)
    print("Starting Training : ")
    print("*" * 20)
    # Initialize a few things
    total_epoch_train_loss_gen = 0
    total_epoch_train_loss_disc = 0
    total_epoch_train_metric = {}
    average_epoch_train_metric = {}
    for metric in params["metrics"]:
        if "per_label" in metric:  # conditional generation not yet implemented
            total_epoch_train_metric[metric] = []
        else:
            total_epoch_train_metric[metric] = 0

    # automatic mixed precision - https://pytorch.org/docs/stable/amp.html
    if params["model"]["amp"]:
        scaler = GradScaler()
        if params["verbose"]:
            print("Using Automatic mixed precision", flush=True)

    # Set the model to train
    model.train()
    for batch_idx, (subject) in enumerate(
        tqdm(train_dataloader, desc="Looping over training data")
    ):
        #### DISCRIMINATOR STEP WITH ALL REAL LABELS ####
        optimizer_d.zero_grad()
        image_real = (
            torch.cat(
                [subject[key][torchio.DATA] for key in params["channel_keys"]],
                dim=1,
            )
            .float()
            .to(params["device"])
        )
        current_batch_size = image_real.shape[0]

        label_real = torch.full(
            current_batch_size,
            1,
            dtype=torch.float,
            device=params["device"],
        )

        if params["save_training"]:
            write_training_patches(
                subject,
                params,
            )
        # ensure spacing is always present in params and is always subject-specific
        if "spacing" in subject:
            params["subject_spacing"] = subject["spacing"]
        else:
            params["subject_spacing"] = None
        loss_disc_real, calculated_metrics, output_disc_real, _ = step_gan(
            model, image_real, label_real, params, secondary_images=None
        )
        nan_loss = torch.isnan(loss_disc_real)
        second_order = (
            hasattr(optimizer_d, "is_second_order")
            and optimizer_d.is_second_order
        )
        if params["model"]["amp"]:
            with torch.cuda.amp.autocast():
                # if loss is nan, don't backprop and don't step optimizer
                if not nan_loss:
                    scaler(
                        loss=loss_disc_real,
                        optimizer=optimizer_d,
                        clip_grad=params["clip_grad"],
                        clip_mode=params["clip_mode"],
                        parameters=model_parameters_exclude_head(
                            model, clip_mode=params["clip_mode"]
                        ),
                        create_graph=second_order,
                    )
        else:
            if not nan_loss:
                loss_disc_real.backward(create_graph=second_order)
                if params["clip_grad"] is not None:
                    dispatch_clip_grad_(
                        parameters=model_parameters_exclude_head(
                            model, clip_mode=params["clip_mode"]
                        ),
                        value=params["clip_grad"],
                        mode=params["clip_mode"],
                    )

        #### DISCRIMINATOR STEP WITH ALL FAKE LABELS ####
        latent_vector = torch.randn(
            current_batch_size,
            params["model"]["latent_vector_size"],
            device=params["device"],
        )
        label_fake = label_real.fill_(0)
        fake_images = model.generator(latent_vector)
        loss_disc_fake, calculated_metrics, output_disc_fake, _ = step_gan(
            model,
            fake_images,
            label_fake,
            params,
            secondary_images=None,
        )

        nan_loss = torch.isnan(loss_disc_fake)
        if params["model"]["amp"]:
            with torch.cuda.amp.autocast():
                # if loss is nan, don't backprop and don't step optimizer
                if not nan_loss:
                    scaler(
                        loss=loss_disc_fake,
                        optimizer=optimizer_d,
                        clip_grad=params["clip_grad"],
                        clip_mode=params["clip_mode"],
                        parameters=model_parameters_exclude_head(
                            model, clip_mode=params["clip_mode"]
                        ),
                        create_graph=second_order,
                    )
        else:
            if not nan_loss:
                loss_disc_fake.backward(create_graph=second_order)
                if params["clip_grad"] is not None:
                    dispatch_clip_grad_(
                        parameters=model_parameters_exclude_head(
                            model, clip_mode=params["clip_mode"]
                        ),
                        value=params["clip_grad"],
                        mode=params["clip_mode"],
                    )
        loss_disc = (loss_disc_real + loss_disc_fake) / 2
        if not nan_loss:
            total_epoch_train_loss_disc += loss_disc.detach().cpu().item()
        optimizer_d.step()
        ### GENERATOR STEP ###
        optimizer_g.zero_grad()
        label_fake = label_real.fill_(1)
        loss_gen, calculated_metrics, output_gen_step, _ = step_gan(
            model, fake_images, label_fake, params, secondary_images=image_real
        )

        nan_loss = torch.isnan(loss_gen)
        if params["model"]["amp"]:
            with torch.cuda.amp.autocast():
                # if loss is nan, don't backprop and don't step optimizer
                if not nan_loss:
                    scaler(
                        loss=loss_gen,
                        optimizer=optimizer_d,
                        clip_grad=params["clip_grad"],
                        clip_mode=params["clip_mode"],
                        parameters=model_parameters_exclude_head(
                            model, clip_mode=params["clip_mode"]
                        ),
                        create_graph=second_order,
                    )
        else:
            if not nan_loss:
                loss_gen.backward(create_graph=second_order)
                if params["clip_grad"] is not None:
                    dispatch_clip_grad_(
                        parameters=model_parameters_exclude_head(
                            model, clip_mode=params["clip_mode"]
                        ),
                        value=params["clip_grad"],
                        mode=params["clip_mode"],
                    )
        optimizer_g.step()
        if not nan_loss:
            total_epoch_train_loss_gen += loss_gen.detach().cpu().item()
    average_epoch_train_loss_gen = total_epoch_train_loss_gen / len(
        train_dataloader
    )
    print(
        "     Epoch Final generator train loss : ",
        average_epoch_train_loss_gen,
    )
    average_epoch_train_loss_disc = total_epoch_train_loss_disc / len(
        train_dataloader
    )
    print(
        "     Epoch Final discriminator train loss : ",
        average_epoch_train_loss_disc,
    )
    if params["verbose"]:
        # For printing information at halftime during an epoch
        if ((batch_idx + 1) % (len(train_dataloader) / 2) == 0) and (
            (batch_idx + 1) < len(train_dataloader)
        ):
            print(
                "\nHalf-Epoch Average generator train loss : ",
                total_epoch_train_loss_gen / (batch_idx + 1),
            )
            print(
                "\nHalf-Epoch Average discriminator train loss : ",
                total_epoch_train_loss_gen / (batch_idx + 1),
            )
            for metric in params["metrics"]:
                if isinstance(total_epoch_train_metric[metric], np.ndarray):
                    to_print = (
                        total_epoch_train_metric[metric] / (batch_idx + 1)
                    ).tolist()
                else:
                    to_print = total_epoch_train_metric[metric] / (
                        batch_idx + 1
                    )
                print(
                    "Half-Epoch Average train " + metric + " : ",
                    to_print,
                )
    average_epoch_train_metric = print_and_format_metrics(
        average_epoch_train_metric,
        total_epoch_train_metric,
        params["metrics"],
        "train",
        len(train_dataloader),
    )
    return (
        average_epoch_train_loss_gen,
        average_epoch_train_loss_disc,
        average_epoch_train_metric,
    )


def training_loop_gans(
    training_data: DataFrame,
    validation_data: DataFrame,
    device: str,
    params: dict,
    output_dir: Union[str, Path],
    testing_data=Union[DataFrame, None],
    epochs=Union[int, None],
):
    """
    The main training loop for GANs.

    Args:
        training_data (pandas.DataFrame): The data to use for training.
        validation_data (pandas.DataFrame): The data to use for validation.
        device (str): The device to perform computations on.
        params (dict): The parameters dictionary.
        output_dir (str): The output directory.
        testing_data (pandas.DataFrame): The data to use for testing.
        epochs (int): The number of epochs to train; if None, take from params.
    """
    if epochs is None:
        epochs = params["num_epochs"]
    params["device"] = device
    params["output_dir"] = output_dir
    params["training_data"] = training_data
    params["validation_data"] = validation_data
    params["testing_data"] = testing_data
    testingDataDefined = True
    if params["testing_data"] is None:
        # testing_data = validation_data
        testingDataDefined = False

    # Setup a few variables for tracking
    best_loss_disc = 1e7
    best_loss_gen = -1e7
    patience, start_epoch = 0, 0
    first_model_saved = False
    model_paths = {
        "best": os.path.join(
            output_dir, params["model"]["architecture"] + best_model_path_end
        ),
        "initial": os.path.join(
            output_dir,
            params["model"]["architecture"] + initial_model_path_end,
        ),
        "latest": os.path.join(
            output_dir, params["model"]["architecture"] + latest_model_path_end
        ),
    }
    main_dict = None
    if os.path.exists(model_paths["best"]):
        main_dict = load_model(model_paths["best"], params["device"])
        version_check(params["version"], version_to_check=main_dict["version"])
        params["previous_parameters"] = main_dict.get("parameters", None)

    # Defining our model here according to parameters mentioned in the configuration file
    print("Number of channels : ", params["model"]["num_channels"])

    (
        model,
        optimizer_g,
        optimizer_d,
        train_dataloader,
        val_dataloader,
        scheduler_g,
        scheduler_d,
        params,
    ) = create_pytorch_objects_gan(
        params, training_data, validation_data, device
    )
    # save the initial model
    if not os.path.exists(model_paths["initial"]):
        save_model(
            {
                "epoch": 0,
                "model_state_dict": get_model_dict(model, params["device_id"]),
                "optimizer_gen_state_dict": optimizer_g.state_dict(),
                "optimizer_disc_state_dict": optimizer_d.state_dict(),
                "loss_gen": best_loss_gen,
                "loss_disc": best_loss_disc,
            },
            model,
            params,
            model_paths["initial"],
            onnx_export=False,
        )
        print("Initial model saved.")
    # if previous model file is present, load it up
    if main_dict is not None:
        try:
            model.load_state_dict(main_dict["model_state_dict"])
            start_epoch = main_dict["epoch"]
            optimizer_g.load_state_dict(main_dict["optimizer_gen_state_dict"])
            optimizer_d.load_state_dict(main_dict["optimizer_disc_state_dict"])
            best_loss_gen = main_dict["loss_gen"]
            best_loss_disc = main_dict["loss_disc"]
            params["previous_parameters"] = main_dict.get("parameters", None)
            print("Previous model successfully loaded.")
        except RuntimeWarning:
            RuntimeWarning(
                "Previous model could not be loaded, initializing model"
            )
    if params["model"]["print_summary"]:
        print_model_summary(
            model,
            params["batch_size"],
            params["model"]["num_channels"],
            params["patch_size"],
            params["device"],
        )

    if testingDataDefined:
        test_dataloader = get_testing_loader(params)
    # Start training time here
    start_time = time.time()

    if not (os.environ.get("HOSTNAME") is None):
        print("Hostname :", os.environ.get("HOSTNAME"))

    # datetime object containing current date and time
    print("Initializing training at :", get_date_time(), flush=True)

    ## TODO add metrics part

    # Setup a few loggers for tracking
    train_logger = LoggerGAN(
        logger_csv_filename=os.path.join(output_dir, "logs_training.csv"),
        metrics=metrics_log,
    )
    valid_logger = LoggerGAN(
        logger_csv_filename=os.path.join(output_dir, "logs_validation.csv"),
        metrics=metrics_log,
    )
    if testingDataDefined:
        test_logger = LoggerGAN(
            logger_csv_filename=os.path.join(output_dir, "logs_testing.csv"),
            metrics=metrics_log,
        )
    train_logger.write_header(mode="train")
    valid_logger.write_header(mode="valid")
    if testingDataDefined:
        test_logger.write_header(mode="test")
    # TODO Do we have medcam?
    # if "medcam" in params:
    #     model = medcam.inject(
    #         model,
    #         output_dir=os.path.join(
    #             output_dir, "attention_maps", params["medcam"]["backend"]
    #         ),
    #         backend=params["medcam"]["backend"],
    #         layer=params["medcam"]["layer"],
    #         save_maps=False,
    #         return_attention=True,
    #         enabled=False,
    #     )
    #     params["medcam_enabled"] = False

    print("Using device:", device, flush=True)
    # Iterate for number of epochs
    for epoch in range(start_epoch, epochs):
        if params["track_memory_usage"]:
            file_to_write_mem = os.path.join(output_dir, "memory_usage.csv")
            if os.path.exists(file_to_write_mem):
                # append to previously generated file
                file_mem = open(file_to_write_mem, "a")
                outputToWrite_mem = ""
            else:
                # if file was absent, write header information
                file_mem = open(file_to_write_mem, "w")
                outputToWrite_mem = "Epoch,Memory_Total,Memory_Available,Memory_Percent_Free,Memory_Usage,"  # used to write output
                if params["device"] == "cuda":
                    outputToWrite_mem += "CUDA_active.all.peak,CUDA_active.all.current,CUDA_active.all.allocated"
                outputToWrite_mem += "\n"

            mem = psutil.virtual_memory()
            outputToWrite_mem += (
                str(epoch)
                + ","
                + str(mem[0])
                + ","
                + str(mem[1])
                + ","
                + str(mem[2])
                + ","
                + str(mem[3])
            )
            if params["device"] == "cuda":
                mem_cuda = torch.cuda.memory_stats()
                outputToWrite_mem += (
                    ","
                    + str(mem_cuda["active.all.peak"])
                    + ","
                    + str(mem_cuda["active.all.current"])
                    + ","
                    + str(mem_cuda["active.all.allocated"])
                )
            outputToWrite_mem += ",\n"
            file_mem.write(outputToWrite_mem)
            file_mem.close()

        # Printing times
        epoch_start_time = time.time()
        print("*" * 20)
        print("*" * 20)
        print("Starting Epoch : ", epoch)
        if params["verbose"]:
            print("Epoch start time : ", get_date_time())

        params["current_epoch"] = epoch
        (
            epoch_train_loss_gen,
            epoch_train_loss_disc,
            epoch_train_metric,
        ) = train_network_gan(
            model, train_dataloader, optimizer_g, optimizer_d, params
        )
        (
            epoch_valid_loss_gen,
            epoch_valid_loss_disc,
            epoch_valid_metric,
        ) = validate_network_gan(
            model,
            val_dataloader,
            scheduler_g,
            scheduler_d,
            params,
            epoch,
            mode="validation",
        )
        patience += 1

        # Write the losses to a logger
        train_logger.write(
            epoch,
            epoch_train_loss_disc,
            epoch_train_loss_gen,
            epoch_train_metric,
        )
        valid_logger.write(
            epoch,
            epoch_valid_loss_disc,
            epoch_valid_loss_gen,
            epoch_valid_metric,
        )
        if testingDataDefined:
            (
                epoch_test_loss_gen,
                epoch_test_loss_disc,
                epoch_test_metric,
            ) = validate_network_gan(
                model,
                test_dataloader,
                scheduler_d,
                scheduler_g,
                params,
                epoch,
                mode="testing",
            )
            test_logger.write(
                epoch,
                epoch_test_loss_disc,
                epoch_test_loss_gen,
                epoch_test_metric,
            )
        if params["verbose"]:
            print("Epoch end time : ", get_date_time())
        epoch_end_time = time.time()
        print(
            "Time taken for epoch : ",
            (epoch_end_time - epoch_start_time) / 60,
            " mins",
            flush=True,
        )

        model_dict = get_model_dict(model, params["device_id"])

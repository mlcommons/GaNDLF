import os, time, psutil
from typing import Tuple, Union
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import torchio
from medcam import medcam

from GANDLF.data import get_testing_loader
from GANDLF.grad_clipping.grad_scaler import GradScaler, model_parameters_exclude_head
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
    get_model_dict,
    print_and_format_metrics,
)
from GANDLF.metrics import overall_stats
from GANDLF.logger import Logger
from .step import step
from .forward_pass import validate_network
from .generic import create_pytorch_objects

# hides torchio citation request, see https://github.com/fepegar/torchio/issues/235
os.environ["TORCHIO_HIDE_CITATION_PROMPT"] = "1"


def train_network(
    model: torch.nn.Module,
    train_dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    params: dict,
) -> Tuple[float, dict]:
    """
    This function performs the training of the network.

    Args:
        model (torch.nn.Module): The model to process the input image with, it should support appropriate dimensions.
        train_dataloader (DataLoader): The dataloader for the training epoch.
        optimizer (torch.optim.Optimizer): Optimizer for optimizing network.
        params (dict): The parameters dictionary.

    Returns:
        Tuple[float, dict]: The average epoch training loss and metrics.
    """
    print("*" * 20)
    print("Starting Training : ")
    print("*" * 20)
    # Initialize a few things
    total_epoch_train_loss = 0
    total_epoch_train_metric: dict[str, Union[float, np.array]] = {}
    average_epoch_train_metric = {}
    # TODO: calculate metrics for segmentation and other problems. btw what are possible problem types?
    calculate_overall_metrics = params["problem_type"] in {
        "classification",
        "regression",
    }

    # get ground truths
    if calculate_overall_metrics:
        ground_truth_array = []
        predictions_array = []

    for metric in params["metrics"]:
        # TODO: can it be per-label for non-classif?
        if "per_label" in metric:
            total_epoch_train_metric[metric] = np.zeros(
                1
            )  # real shape would be defined during execution
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
        optimizer.zero_grad()
        image = (  # 5D tensor: (B, C, H, W, D)
            torch.cat(
                [subject[key][torchio.DATA] for key in params["channel_keys"]], dim=1
            )
            .float()
            .to(params["device"])
        )
        if (
            "value_keys" in params
        ):  # classification / regression (when label is scalar) or multilabel classif/regression
            label = torch.cat([subject[key] for key in params["value_keys"]], dim=0)
            # min is needed because for certain cases, batch size becomes smaller than the total remaining labels
            label = label.reshape(
                min(params["batch_size"], len(label)), len(params["value_keys"])
            )
        else:
            label = subject["label"][
                torchio.DATA
            ]  # segmentation; label is (B, C, H, W, D) image
        label = label.to(params["device"])

        if params["save_training"]:
            write_training_patches(subject, params)

        # ensure spacing is always present in params and is always subject-specific
        if "spacing" in subject:
            params["subject_spacing"] = subject["spacing"]
        else:
            params["subject_spacing"] = None
        loss, calculated_metrics, output, _ = step(model, image, label, params)
        # store predictions for classification
        if calculate_overall_metrics:
            # TODO: smelly code. if segmentation, in some models output may be a list of tensors rather then a one
            #  tensor. This is not handled here. However, `calculate_overall_metrics` is set to False for segmentation
            ground_truth_array.extend(label.detach().cpu())
            # TODO: output is BATCH_SIZE x N_CLASSES. What if not?
            batch_predictions = torch.argmax(output, 1).cpu()
            assert len(batch_predictions) == len(label)
            predictions_array.extend(batch_predictions.detach().cpu())

        nan_loss = torch.isnan(loss)
        # loss backward
        second_order = (
            hasattr(optimizer, "is_second_order") and optimizer.is_second_order
        )
        if params["model"]["amp"]:
            with torch.cuda.amp.autocast():
                # if loss is nan, don't backprop and don't step optimizer
                if not nan_loss:
                    scaler(
                        loss=loss,
                        optimizer=optimizer,
                        clip_grad=params["clip_grad"],
                        clip_mode=params["clip_mode"],
                        parameters=model_parameters_exclude_head(
                            model, clip_mode=params["clip_mode"]
                        ),
                        create_graph=second_order,
                    )
        else:
            if not nan_loss:
                loss.backward(create_graph=second_order)
                if params["clip_grad"] is not None:
                    dispatch_clip_grad_(
                        parameters=model_parameters_exclude_head(
                            model, clip_mode=params["clip_mode"]
                        ),
                        value=params["clip_grad"],
                        mode=params["clip_mode"],
                    )
                optimizer.step()

        # Non network training related
        if not nan_loss:
            total_epoch_train_loss += loss.detach().cpu().item()
        for metric, metric_val in calculated_metrics.items():
            total_epoch_train_metric[metric] = (
                total_epoch_train_metric[metric] + metric_val
            )

        if params["verbose"]:
            # For printing information at halftime during an epoch
            if ((batch_idx + 1) % (len(train_dataloader) / 2) == 0) and (
                (batch_idx + 1) < len(train_dataloader)
            ):
                print(
                    "\nHalf-Epoch Average train loss : ",
                    total_epoch_train_loss / (batch_idx + 1),
                )
                for metric in params["metrics"]:
                    if isinstance(total_epoch_train_metric[metric], np.ndarray):
                        to_print = (
                            total_epoch_train_metric[metric] / (batch_idx + 1)
                        ).tolist()
                    else:
                        to_print = total_epoch_train_metric[metric] / (batch_idx + 1)
                    print("Half-Epoch Average train " + metric + " : ", to_print)

    average_epoch_train_loss = total_epoch_train_loss / len(train_dataloader)
    print("     Epoch Final   train loss : ", average_epoch_train_loss)

    # get overall stats for classification
    if calculate_overall_metrics:
        average_epoch_train_metric = overall_stats(
            torch.Tensor(predictions_array), torch.Tensor(ground_truth_array), params
        )
    # TODO: the following not just prints and formats, but updates the dict also. Clean this code
    #  1. average_epoch_train_metric and total_epoch_train_metric are combined
    #  2. list values in total_epoch_train_metric are converted to strings by some logic (but not in avg_ep_tr_metr)
    average_epoch_train_metric = print_and_format_metrics(
        average_epoch_train_metric,
        total_epoch_train_metric,
        params["metrics"],
        "train",
        len(train_dataloader),
    )

    return average_epoch_train_loss, average_epoch_train_metric


def training_loop(
    training_data: pd.DataFrame,
    validation_data: pd.DataFrame,
    device: str,
    params: dict,
    output_dir: str,
    testing_data: bool = None,
    epochs: bool = None,
) -> None:
    """
    The main training loop.

    Args:
        training_data (pd.DataFrame): The data to use for training.
        validation_data (pd.DataFrame): The data to use for validation.
        device (str): The device to perform computations on.
        params (dict): The parameters dictionary.
        output_dir (str): The output directory.
        testing_data (pd.DataFrame): The data to use for testing.
        epochs (int): The number of epochs to train; if None, take from params.
    """
    # Some autodetermined factors
    if epochs is None:
        epochs = params["num_epochs"]
    params["device"] = device
    params["output_dir"] = output_dir
    params["training_data"] = training_data
    params["validation_data"] = validation_data
    params["testing_data"] = testing_data
    testingDataDefined = True
    if not isinstance(testing_data, pd.DataFrame):
        if params["testing_data"] is None:
            testingDataDefined = False

    # Setup a few variables for tracking
    best_loss = 1e7
    patience, start_epoch = 0, 0
    first_model_saved = False
    model_paths = {
        "best": os.path.join(
            output_dir, params["model"]["architecture"] + best_model_path_end
        ),
        "initial": os.path.join(
            output_dir, params["model"]["architecture"] + initial_model_path_end
        ),
        "latest": os.path.join(
            output_dir, params["model"]["architecture"] + latest_model_path_end
        ),
    }

    # if previous model file is present, load it up for sanity checks
    main_dict = None
    if os.path.exists(model_paths["best"]):
        main_dict = load_model(model_paths["best"], params["device"])
        version_check(params["version"], version_to_check=main_dict["version"])
        params["previous_parameters"] = main_dict.get("parameters", None)

    # Defining our model here according to parameters mentioned in the configuration file
    print("Number of channels : ", params["model"]["num_channels"])

    (
        model,
        optimizer,
        train_dataloader,
        val_dataloader,
        scheduler,
        params,
    ) = create_pytorch_objects(params, training_data, validation_data, device)

    # save the initial model
    if not os.path.exists(model_paths["initial"]):
        save_model(
            {
                "epoch": 0,
                "model_state_dict": get_model_dict(model, params["device_id"]),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": best_loss,
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
            optimizer.load_state_dict(main_dict["optimizer_state_dict"])
            best_loss = main_dict["loss"]
            params["previous_parameters"] = main_dict.get("parameters", None)
            print("Previous model successfully loaded.")
        except RuntimeWarning:
            RuntimeWarning("Previous model could not be loaded, initializing model")

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

    metrics_log = list(params["metrics"])

    calculate_overall_metrics = params["problem_type"] in {
        "classification",
        "regression",
    }

    if calculate_overall_metrics:
        # get the overall metrics that are calculated automatically for classification/regression problems
        if params["problem_type"] == "regression":
            overall_metrics = overall_stats(
                torch.Tensor([1]), torch.Tensor([1]), params
            )
        elif params["problem_type"] == "classification":
            # this is just used to generate the headers for the overall stats
            temp_tensor = torch.randint(0, params["model"]["num_classes"], (5,))
            overall_metrics = overall_stats(
                temp_tensor.to(dtype=torch.int32),
                temp_tensor.to(dtype=torch.int32),
                params,
            )
        else:
            raise NotImplementedError("Problem type not implemented for overall stats")

        for metric in overall_metrics:
            if metric not in metrics_log:
                metrics_log.append(metric)

    # Setup a few loggers for tracking
    train_logger = Logger(
        logger_csv_filename=os.path.join(output_dir, "logs_training.csv"),
        metrics=metrics_log,
        mode="train",
    )
    valid_logger = Logger(
        logger_csv_filename=os.path.join(output_dir, "logs_validation.csv"),
        metrics=metrics_log,
        mode="valid",
    )
    if testingDataDefined:
        test_logger = Logger(
            logger_csv_filename=os.path.join(output_dir, "logs_testing.csv"),
            metrics=metrics_log,
            mode="test",
        )

    if "medcam" in params:
        model = medcam.inject(
            model,
            output_dir=os.path.join(
                output_dir, "attention_maps", params["medcam"]["backend"]
            ),
            backend=params["medcam"]["backend"],
            layer=params["medcam"]["layer"],
            save_maps=False,
            return_attention=True,
            enabled=False,
        )
        params["medcam_enabled"] = False

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

        epoch_train_loss, epoch_train_metric = train_network(
            model, train_dataloader, optimizer, params
        )
        epoch_valid_loss, epoch_valid_metric = validate_network(
            model, val_dataloader, scheduler, params, epoch, mode="validation"
        )

        patience += 1

        # Write the losses to a logger
        train_logger.write(epoch, epoch_train_loss, epoch_train_metric)
        valid_logger.write(epoch, epoch_valid_loss, epoch_valid_metric)

        if testingDataDefined:
            epoch_test_loss, epoch_test_metric = validate_network(
                model, test_dataloader, scheduler, params, epoch, mode="testing"
            )
            test_logger.write(epoch, epoch_test_loss, epoch_test_metric)

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

        # Start to check for loss
        if not (first_model_saved) or (epoch_valid_loss <= torch.tensor(best_loss)):
            best_loss = epoch_valid_loss
            best_train_idx = epoch
            patience = 0

            model.eval()

            save_model(
                {
                    "epoch": best_train_idx,
                    "model_state_dict": model_dict,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": best_loss,
                },
                model,
                params,
                model_paths["best"],
                onnx_export=False,
            )
            model.train()
            first_model_saved = True

        if params["model"]["save_at_every_epoch"]:
            save_model(
                {
                    "epoch": epoch,
                    "model_state_dict": model_dict,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": epoch_valid_loss,
                },
                model,
                params,
                os.path.join(
                    output_dir,
                    params["model"]["architecture"]
                    + "_epoch_"
                    + str(epoch)
                    + ".pth.tar",
                ),
                onnx_export=False,
            )
            model.train()

        # save the latest model
        if os.path.exists(model_paths["latest"]):
            os.remove(model_paths["latest"])
        save_model(
            {
                "epoch": epoch,
                "model_state_dict": model_dict,
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": best_loss,
            },
            model,
            params,
            model_paths["latest"],
            onnx_export=False,
        )
        print("Latest model saved.")
        print("Current Best epoch: ", best_train_idx)

        if patience > params["patience"]:
            print(
                "Performance Metric has not improved for %d epochs, exiting training loop!"
                % (patience),
                flush=True,
            )
            break

    # End train time
    end_time = time.time()

    print(
        "Total time to finish Training : ",
        (end_time - start_time) / 60,
        " mins",
        flush=True,
    )

    # once the training is done, optimize the best model
    if os.path.exists(model_paths["best"]):
        optimize_and_save_model(model, params, model_paths["best"], onnx_export=True)


if __name__ == "__main__":
    import argparse, pickle

    torch.multiprocessing.freeze_support()
    # parse the cli arguments here
    parser = argparse.ArgumentParser(description="Training Loop of GANDLF")
    parser.add_argument(
        "-train_loader_pickle", type=str, help="Train loader pickle", required=True
    )
    parser.add_argument(
        "-val_loader_pickle", type=str, help="Validation loader pickle", required=True
    )
    parser.add_argument(
        "-testing_loader_pickle",
        type=str,
        help="Testing loader pickle",
        required=False,
        default=None,
    )
    parser.add_argument(
        "-parameter_pickle", type=str, help="Parameters pickle", required=True
    )
    parser.add_argument("-outputDir", type=str, help="Output directory", required=True)
    parser.add_argument("-device", type=str, help="Device to train on", required=True)

    args = parser.parse_args()

    # # write parameters to pickle - this should not change for the different folds, so keeping is independent
    parameters = pickle.load(open(args.parameter_pickle, "rb"))
    trainingDataFromPickle = pd.read_pickle(args.train_loader_pickle)
    validationDataFromPickle = pd.read_pickle(args.val_loader_pickle)
    testingData_str = args.testing_loader_pickle
    testingDataFromPickle = pd.read_pickle(testingData_str) if testingData_str else None

    training_loop(
        training_data=trainingDataFromPickle,
        validation_data=validationDataFromPickle,
        output_dir=args.outputDir,
        device=args.device,
        params=parameters,
        testing_data=testingDataFromPickle,
    )

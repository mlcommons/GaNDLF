import os
import torch

os.environ["TORCHIO_HIDE_CITATION_PROMPT"] = "1"  # hides torchio citation request


def training_loop(
    training_data,
    validation_data,
    device,
    params,
    output_dir,
    testing_data=None,
):
    """
    The main training loop.

    Args:
        training_data (pandas.DataFrame): The data to use for training.
        validation_data (pandas.DataFrame): The data to use for validation.
        device (str): The device to perform computations on.
        params (dict): The parameters dictionary.
        output_dir (str): The output directory.
        testing_data (pandas.DataFrame): The data to use for testing.
    """
    # Some autodetermined factors
    epochs = params["num_epochs"]
    params["device"] = device
    params["output_dir"] = output_dir

    # Defining our model here according to parameters mentioned in the configuration file
    print("Number of channels : ", params["model"]["num_channels"])

    # Fetch the model according to params mentioned in the configuration file
    model = global_models_dict[params["model"]["architecture"]](parameters=params)

    # Set up the dataloaders
    training_data_for_torch = ImagesFromDataFrame(training_data, params, train=True)

    validation_data_for_torch = ImagesFromDataFrame(
        validation_data, params, train=False
    )

    testingDataDefined = True
    if testing_data is None:
        # testing_data = validation_data
        testingDataDefined = False

    if testingDataDefined:
        test_data_for_torch = ImagesFromDataFrame(testing_data, params, train=False)

    train_dataloader = DataLoader(
        training_data_for_torch,
        batch_size=params["batch_size"],
        shuffle=True,
        pin_memory=False,  # params["pin_memory_dataloader"], # this is going OOM if True - needs investigation
    )
    params["training_samples_size"] = len(train_dataloader.dataset)

    val_dataloader = DataLoader(
        validation_data_for_torch,
        batch_size=1,
        pin_memory=False,  # params["pin_memory_dataloader"], # this is going OOM if True - needs investigation
    )

    if testingDataDefined:
        test_dataloader = DataLoader(
            test_data_for_torch,
            batch_size=1,
            pin_memory=False,  # params["pin_memory_dataloader"], # this is going OOM if True - needs investigation
        )

    # Fetch the appropriate channel keys
    # Getting the channels for training and removing all the non numeric entries from the channels
    params = populate_channel_keys_in_params(validation_data_for_torch, params)

    # Calculate the weights here
    if params["weighted_loss"]:
        # Set up the dataloader for penalty calculation
        penalty_data = ImagesFromDataFrame(
            training_data,
            parameters=params,
            train=False,
        )
        penalty_loader = DataLoader(
            penalty_data,
            batch_size=1,
            shuffle=True,
            pin_memory=False,
        )

        params["weights"], params["class_weights"] = get_class_imbalance_weights(
            penalty_loader, params
        )
    else:
        params["weights"], params["class_weights"] = None, None

    # Fetch the optimizers
    params["model_parameters"] = model.parameters()
    optimizer = global_optimizer_dict[params["optimizer"]["type"]](params)
    params["optimizer_object"] = optimizer

    if not ("step_size" in params["scheduler"]):
        params["scheduler"]["step_size"] = (
            params["training_samples_size"] / params["learning_rate"]
        )

    scheduler = global_schedulers_dict[params["scheduler"]["type"]](params)

    # these keys contain generators, and are not needed beyond this point in params
    generator_keys_to_remove = ["optimizer_object", "model_parameters"]
    for key in generator_keys_to_remove:
        params.pop(key, None)

    # Start training time here
    start_time = time.time()
    print("\n\n")

    if not (os.environ.get("HOSTNAME") is None):
        print("Hostname :", os.environ.get("HOSTNAME"))

    # datetime object containing current date and time
    print("Initializing training at :", get_date_time(), flush=True)

    # Setup a few loggers for tracking
    train_logger = Logger(
        logger_csv_filename=os.path.join(output_dir, "logs_training.csv"),
        metrics=params["metrics"],
    )
    valid_logger = Logger(
        logger_csv_filename=os.path.join(output_dir, "logs_validation.csv"),
        metrics=params["metrics"],
    )
    test_logger = Logger(
        logger_csv_filename=os.path.join(output_dir, "logs_testing.csv"),
        metrics=params["metrics"],
    )
    train_logger.write_header(mode="train")
    valid_logger.write_header(mode="valid")
    test_logger.write_header(mode="valid")

    model, params["model"]["amp"], device = send_model_to_device(
        model, amp=params["model"]["amp"], device=params["device"], optimizer=optimizer
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

    # Setup a few variables for tracking
    best_loss = 1e7
    patience, start_epoch = 0, 0
    first_model_saved = False
    best_model_path = os.path.join(
        output_dir, params["model"]["architecture"] + "_best.pth.tar"
    )

    # if previous model file is present, load it up
    if os.path.exists(best_model_path):
        print("Previous model found. Loading it up.")
        try:
            main_dict = torch.load(best_model_path)
            model.load_state_dict(main_dict["model_state_dict"])
            start_epoch = main_dict["epoch"]
            optimizer.load_state_dict(main_dict["optimizer_state_dict"])
            best_loss = main_dict["best_loss"]
            print("Previous model loaded successfully.")
        except Exception as e:
            print("Previous model could not be loaded, error: ", e)

    print("Using device:", device, flush=True)

    # Iterate for number of epochs
    for epoch in range(start_epoch, epochs):

        # Printing times
        epoch_start_time = time.time()
        print("*" * 20)
        print("Starting Epoch : ", epoch)
        print("Epoch start time : ", get_date_time())

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

        print("Epoch end time : ", get_date_time())
        epoch_end_time = time.time()
        print(
            "Time taken for epoch : ",
            (epoch_end_time - epoch_start_time) / 60,
            " mins",
            flush=True,
        )

        # Start to check for loss
        if not (first_model_saved) or (epoch_valid_loss <= torch.tensor(best_loss)):
            best_loss = epoch_valid_loss
            best_train_idx = epoch
            patience = 0
            torch.save(
                {
                    "epoch": best_train_idx,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_loss": best_loss,
                },
                best_model_path,
            )
            first_model_saved = True

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


if __name__ == "__main__":

    import argparse, pickle, pandas

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
        "-testing_loader_pickle", type=str, help="Testing loader pickle", required=True
    )
    parser.add_argument(
        "-parameter_pickle", type=str, help="Parameters pickle", required=True
    )
    parser.add_argument("-outputDir", type=str, help="Output directory", required=True)
    parser.add_argument("-device", type=str, help="Device to train on", required=True)

    args = parser.parse_args()

    # # write parameters to pickle - this should not change for the different folds, so keeping is independent
    parameters = pickle.load(open(args.parameter_pickle, "rb"))
    trainingDataFromPickle = pandas.read_pickle(args.train_loader_pickle)
    validationDataFromPickle = pandas.read_pickle(args.val_loader_pickle)
    testingData_str = args.testing_loader_pickle
    if testingData_str == "None":
        testingDataFromPickle = None
    else:
        testingDataFromPickle = pandas.read_pickle(testingData_str)

    training_loop(
        training_data=trainingDataFromPickle,
        validation_data=validationDataFromPickle,
        output_dir=args.outputDir,
        device=args.device,
        params=parameters,
        testing_data=testingDataFromPickle,
    )

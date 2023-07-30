import os
import argparse
import ast
import sys
import traceback

from GANDLF import version
from pathlib import Path

from GANDLF.training_manager import TrainingManager, TrainingManager_split
from GANDLF.inference_manager import InferenceManager
from GANDLF.parseConfig import parseConfig
from GANDLF.utils import (
    populate_header_in_parameters,
    parseTrainingCSV,
    parseTestingCSV,

)
from .copyright_message import copyrightMessage


def main_run(
    data_csv, config_file, model_dir, train_mode, device, resume, reset, output_dir=None
):
    """
    Main function that runs the training and inference.

    Args:
        data_csv (str): The CSV file of the training data.
        config_file (str): The YAML file of the training configuration.
        model_dir (str): The model directory; for training, model is written out here, and for inference, trained model is expected here.
        train_mode (bool): Whether to train or infer.
        device (str): The device type.
        resume (bool): Whether the previous run will be resumed or not.
        reset (bool): Whether the previous run will be reset or not.
        output_dir (str): The output directory for the inference session.

    Returns:
        None
    """
    file_data_full = data_csv
    model_parameters = config_file
    device = device
    parameters = parseConfig(model_parameters)
    parameters["device_id"] = -1

    if train_mode:
        if resume:
            print(
                "Trying to resume training without changing any parameters from previous run.",
                flush=True,
            )
        parameters["output_dir"] = model_dir
        Path(parameters["output_dir"]).mkdir(parents=True, exist_ok=True)

    # if the output directory is not specified, then use the model directory even for the testing data
    # default behavior
    parameters["output_dir"] = parameters.get("output_dir", output_dir)
    if parameters["output_dir"] is None:
        parameters["output_dir"] = model_dir
    Path(parameters["output_dir"]).mkdir(parents=True, exist_ok=True)

    if "-1" in device:
        device = "cpu"

    # parse training CSV
    if "," in file_data_full:
        # training and validation pre-split
        data_full = None
        all_csvs = file_data_full.split(",")
        data_train, headers_train = parseTrainingCSV(all_csvs[0], train=train_mode)
        data_validation, headers_validation = parseTrainingCSV(
            all_csvs[1], train=train_mode
        )
        assert (
            headers_train == headers_validation
        ), "The training and validation CSVs do not have the same header information."

        # testing data is present
        data_testing = None
        headers_testing = headers_train
        if len(all_csvs) == 3:
            data_testing, headers_testing = parseTrainingCSV(
                all_csvs[2], train=train_mode
            )
        assert (
            headers_train == headers_testing
        ), "The training and testing CSVs do not have the same header information."

        parameters = populate_header_in_parameters(parameters, headers_train)
        # if we are here, it is assumed that the user wants to do training
        if train_mode:
            TrainingManager_split(
                dataframe_train=data_train,
                dataframe_validation=data_validation,
                dataframe_testing=data_testing,
                outputDir=parameters["output_dir"],
                parameters=parameters,
                device=device,
                resume=resume,
                reset=reset,
            )
    else:
        data_full, headers = parseTrainingCSV(file_data_full, train=train_mode)
        parameters = populate_header_in_parameters(parameters, headers)
        if train_mode:
            TrainingManager(
                dataframe=data_full,
                outputDir=parameters["output_dir"],
                parameters=parameters,
                device=device,
                resume=resume,
                reset=reset,
            )
        else:
            _, data_full, headers = parseTestingCSV(
                file_data_full, parameters["output_dir"]
            )
            InferenceManager(
                dataframe=data_full,
                modelDir=model_dir,
                outputDir=output_dir,
                parameters=parameters,
                device=device,
            )


def main():
    parser = argparse.ArgumentParser(
        prog="GANDLF",
        formatter_class=argparse.RawTextHelpFormatter,
        description="Semantic segmentation, regression, and classification for medical images using Deep Learning.\n\n"
        + copyrightMessage,
    )
    parser.add_argument(
        "-c",
        "--config",
        "--parameters_file",
        metavar="",
        type=str,
        required=True,
        help="The configuration file (contains all the information related to the training/inference session)",
    )
    parser.add_argument(
        "-i",
        "--inputdata",
        "--data_path",
        metavar="",
        type=str,
        required=True,
        help="Data CSV file that is used for training/inference; can also take comma-separated training-validation pre-split CSVs",
    )
    parser.add_argument(
        "-t",
        "--train",
        metavar="",
        type=ast.literal_eval,
        required=True,
        help="True: training and False: inference; for inference, there needs to be a compatible model saved in '-modeldir'",
    )
    parser.add_argument(
        "-m",
        "--modeldir",
        metavar="",
        type=str,
        help="Training: Output directory to save intermediate files and model weights; inference: location of previous training session output",
    )
    parser.add_argument(
        "-d",
        "--device",
        default="cuda",
        metavar="",
        type=str,
        required=True,
        help="Device to perform requested session on 'cpu' or 'cuda'; for cuda, ensure CUDA_VISIBLE_DEVICES env var is set",
    )
    parser.add_argument(
        "-rt",
        "--reset",
        metavar="",
        default=False,
        type=ast.literal_eval,
        help="Completely resets the previous run by deleting 'modeldir'",
    )
    parser.add_argument(
        "-rm",
        "--resume",
        metavar="",
        default=False,
        type=ast.literal_eval,
        help="Resume previous training by only keeping model dict in 'modeldir'",
    )

    parser.add_argument(
        "-o",
        "--outputdir",
        "--output_path",
        metavar="",
        type=str,
        help="Location to save the output of the inference session. Not used for training.",
    )

    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version="%(prog)s v{}".format(version) + "\n\n" + copyrightMessage,
        help="Show program's version number and exit.",
    )

    # This is a dummy argument that exists to trigger MLCube mounting requirements.
    # Do not remove.
    parser.add_argument("-rawinput", "--rawinput", help=argparse.SUPPRESS)

    args = parser.parse_args()
    if args.modeldir is None and args.outputdir:
        args.modeldir = args.outputdir

    assert args.modeldir is not None, "Missing required parameter: modeldir"

    if os.path.isdir(args.inputdata):
        # Is this a fine assumption to make?
        # Medperf models receive the data generated by the data preparator mlcube
        # We can therefore ensure the output of that mlcube contains a data.csv file
        filename = "data.csv"
        args.inputdata = os.path.join(args.inputdata, filename)

    if not args.train:
        # if inference mode, then no need to check for reset/resume
        args.reset, args.resume = False, False

    if args.reset and args.resume:
        print(
            "WARNING: 'reset' and 'resume' are mutually exclusive; 'resume' will be used."
        )
        args.reset = False

    # config file should always be present
    assert os.path.isfile(args.config), "Configuration file not found!"

    try:
        main_run(
            args.inputdata,
            args.config,
            args.modeldir,
            args.train,
            args.device,
            args.resume,
            args.reset,
            args.outputdir,
        )
    except Exception:
        sys.exit("ERROR: " + traceback.format_exc())

    print("Finished.")


if __name__ == "__main__":
    main()
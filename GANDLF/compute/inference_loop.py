from cv2 import transform
from matplotlib import transforms
from .forward_pass import validate_network
import os
from pathlib import Path

# hides torchio citation request, see https://github.com/fepegar/torchio/issues/235
os.environ["TORCHIO_HIDE_CITATION_PROMPT"] = "1"

import pickle, argparse, torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from skimage.io import imsave
from tqdm import tqdm
from torch.cuda.amp import autocast
import tiffslide as openslide

from GANDLF.data.ImagesFromDataFrame import ImagesFromDataFrame
from GANDLF.utils import populate_channel_keys_in_params, send_model_to_device
from GANDLF.models import global_models_dict
from GANDLF.data.inference_dataloader_histopath import InferTumorSegDataset
from GANDLF.data.preprocessing import get_transforms_for_preprocessing


def inference_loop(inferenceDataFromPickle, device, parameters, outputDir):
    """
    The main training loop.

    Args:
        inferenceDataFromPickle (pandas.DataFrame): The data to use for inference.
        device (str): The device to perform computations on.
        parameters (dict): The parameters dictionary.
        outputDir (str): The output directory.
    """
    # Defining our model here according to parameters mentioned in the configuration file
    print("Number of dims     : ", parameters["model"]["dimension"])
    if "num_channels" in parameters["model"]:
        print("Number of channels : ", parameters["model"]["num_channels"])
    print("Number of classes  : ", len(parameters["model"]["class_list"]))

    # Fetch the model according to params mentioned in the configuration file
    model = global_models_dict[parameters["model"]["architecture"]](
        parameters=parameters
    )

    # Loading the weights into the model
    main_dict = outputDir
    if os.path.isdir(outputDir):
        file_to_check = os.path.join(
            outputDir, str(parameters["model"]["architecture"]) + "_best.pth.tar"
        )
        if not os.path.isfile(file_to_check):
            raise ValueError("The model specified model was not found:", file_to_check)

    parameters["save_output"] = True

    main_dict = torch.load(file_to_check, map_location=torch.device(device))
    model.load_state_dict(main_dict["model_state_dict"])
    model, parameters["model"]["amp"], parameters["device"] = send_model_to_device(
        model, parameters["model"]["amp"], device, optimizer=None
    )
    print("Using device:", parameters["device"], flush=True)

    # ensure batch_size is 1, so that output can be written properly
    parameters["batch_size"] = 1

    if not (os.environ.get("HOSTNAME") is None):
        print("\nHostname     :" + str(os.environ.get("HOSTNAME")), flush=True)

    # radiology inference
    if parameters["modality"] == "rad":

        # Setting up the inference loader
        inferenceDataForTorch = ImagesFromDataFrame(
            inferenceDataFromPickle, parameters, train=False, loader_type="inference"
        )
        inference_loader = DataLoader(inferenceDataForTorch, batch_size=1)

        # get the channel keys for concatenation later (exclude non numeric channel keys)
        parameters = populate_channel_keys_in_params(inference_loader, parameters)

        print("Data Samples: ", len(inference_loader.dataset), flush=True)

        average_epoch_valid_loss, average_epoch_valid_metric = validate_network(
            model, inference_loader, None, parameters, mode="inference"
        )
        print(average_epoch_valid_loss, average_epoch_valid_metric)
    elif parameters["modality"] in ["path", "histo"]:
        # set some defaults
        if not "slide_level" in parameters:
            parameters["slide_level"] = 0
        if not "stride_size" in parameters:
            parameters["stride_size"] = parameters["patch_size"]

        parameters["stride_size"] = np.array(parameters["stride_size"])

        if parameters["stride_size"].size == 1:
            parameters["stride_size"] = np.append(
                parameters["stride_size"], parameters["stride_size"]
            )

        if not "mask_level" in parameters:
            parameters["mask_level"] = parameters["slide_level"]

        if parameters["problem_type"] != "segmentation":
            output_to_write = "SubjectID,x_coords,y_coords,output\n"

        # actual computation
        pbar = tqdm(inferenceDataFromPickle.iterrows())
        for _, row in pbar:
            subject_name = row[parameters["headers"]["subjectIDHeader"]]
            os_image = openslide.open_slide(
                row[parameters["headers"]["channelHeaders"]].values[0]
            )
            level_width, level_height = os_image.level_dimensions[
                int(parameters["slide_level"])
            ]
            subject_dest_dir = os.path.join(outputDir, str(subject_name))
            Path(subject_dest_dir).mkdir(parents=True, exist_ok=True)

            if parameters["problem_type"] == "segmentation":
                probs_map = np.zeros((level_height, level_width), dtype=np.float16)
                count_map = np.zeros((level_height, level_width), dtype=np.uint8)

            patch_size = parameters["patch_size"]

            transform = get_transforms_for_preprocessing(parameters, [], False, False)

            pbar.set_description(
                "Constructing loader for subject: " + str(subject_name)
            )

            patient_dataset_obj = InferTumorSegDataset(
                row[parameters["headers"]["channelHeaders"]].values[0],
                patch_size=patch_size,
                stride_size=parameters["stride_size"],
                selected_level=parameters["slide_level"],
                mask_level=parameters["mask_level"],
                transform=transform,
            )

            dataloader = DataLoader(
                patient_dataset_obj,
                batch_size=int(parameters["batch_size"]),
                shuffle=False,
                num_workers=parameters["q_num_workers"],
            )

            pbar.set_description(
                "Looping over patches for subject: " + str(subject_name)
            )

            for image_patches, (x_coords, y_coords) in dataloader:
                x_coords, y_coords = y_coords.numpy(), x_coords.numpy()
                if parameters["model"]["amp"]:
                    with autocast():
                        output = model(image_patches.float().to(parameters["device"]))
                else:
                    output = model(image_patches.float().to(parameters["device"]))
                output = output.detach().cpu().numpy()

                for i in range(int(output.shape[0])):
                    if parameters["problem_type"] == "segmentation":
                        count_map[
                            x_coords[i] : x_coords[i] + patch_size[0],
                            y_coords[i] : y_coords[i] + patch_size[1],
                        ] += 1
                        probs_map[
                            x_coords[i] : x_coords[i] + patch_size[0],
                            y_coords[i] : y_coords[i] + patch_size[1],
                        ] += output[i][0]
                    else:
                        output_to_write += (
                            str(subject_name)
                            + ","
                            + str(x_coords[i])
                            + ","
                            + str(y_coords[i])
                            + ","
                            + str(output[i][0])
                            + "\n"
                        )

            if parameters["problem_type"] == "segmentation":
                probs_map = probs_map / count_map
                count_map = count_map / count_map.max()
                out = count_map * probs_map
                count_map = np.array(count_map * 255, dtype=np.uint16)
                out_thresh = np.array((out > 0.5) * 255, dtype=np.uint16)
                imsave(
                    os.path.join(
                        subject_dest_dir,
                        str(row[parameters["headers"]["subjectIDHeader"]])
                        + "_prob.png",
                    ),
                    out,
                )
                imsave(
                    os.path.join(
                        subject_dest_dir,
                        str(row[parameters["headers"]["subjectIDHeader"]]) + "_seg.png",
                    ),
                    out_thresh,
                )
                imsave(
                    os.path.join(
                        subject_dest_dir,
                        str(row[parameters["headers"]["subjectIDHeader"]])
                        + "_count.png",
                    ),
                    count_map,
                )
            else:
                output_file = os.path.join(
                    subject_dest_dir,
                    "predictions.csv",
                )
                with open(output_file, "w") as f:
                    f.write(output_to_write)


if __name__ == "__main__":

    # parse the cli arguments here
    parser = argparse.ArgumentParser(description="Inference Loop of GANDLF")
    parser.add_argument(
        "-inference_loader_pickle",
        type=str,
        help="Inference loader pickle",
        required=True,
    )
    parser.add_argument(
        "-parameter_pickle", type=str, help="Parameters pickle", required=True
    )
    parser.add_argument(
        "-headers_pickle", type=str, help="Header pickle", required=True
    )
    parser.add_argument("-outputDir", type=str, help="Output directory", required=True)
    parser.add_argument("-device", type=str, help="Device to train on", required=True)

    args = parser.parse_args()

    # # write parameters to pickle - this should not change for the different folds, so keeping is independent
    patch_size = pickle.load(open(args.patch_size_pickle, "rb"))
    headers = pickle.load(open(args.headers_pickle, "rb"))
    label_header = pickle.load(open(args.label_header_pickle, "rb"))
    parameters = pickle.load(open(args.parameter_pickle, "rb"))
    inferenceDataFromPickle = pd.read_pickle(args.inference_loader_pickle)

    inference_loop(
        inferenceDataFromPickle=inferenceDataFromPickle,
        parameters=parameters,
        outputDir=args.outputDir,
        device=args.device,
    )

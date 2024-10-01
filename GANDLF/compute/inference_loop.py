from .forward_pass import validate_network
from .generic import create_pytorch_objects
import os, sys
from typing import Optional
from pathlib import Path
import pandas as pd

# hides torchio citation request, see https://github.com/fepegar/torchio/issues/235
os.environ["TORCHIO_HIDE_CITATION_PROMPT"] = "1"

import torch
import cv2
import numpy as np
from torch.utils.data import DataLoader
from skimage.io import imsave
from tqdm import tqdm
from torch.cuda.amp import autocast
import tiffslide as openslide
from GANDLF.data import get_testing_loader
from GANDLF.utils import (
    best_model_path_end,
    latest_model_path_end,
    load_ov_model,
    print_model_summary,
    applyCustomColorMap,
)

from GANDLF.data.inference_dataloader_histopath import InferTumorSegDataset
from GANDLF.data.preprocessing import get_transforms_for_preprocessing


def inference_loop(
    inferenceDataFromPickle: pd.DataFrame,
    device: str,
    parameters: dict,
    modelDir: str,
    outputDir: Optional[str] = None,
) -> None:
    """
    The main training loop.

    Args:
        inferenceDataFromPickle (pandas.DataFrame): The data to use for inference.
        device (str): The device to perform computations on.
        parameters (dict): The parameters dictionary.
        modelDir (str): The path to the directory containing the model to be used for inference.
        outputDir (str): The path to the directory where the output of the inference session will be stored.
    """
    # Defining our model here according to parameters mentioned in the configuration file
    print("Current model type : ", parameters["model"]["type"])
    print("Number of dims     : ", parameters["model"]["dimension"])
    if "num_channels" in parameters["model"]:
        print("Number of channels : ", parameters["model"]["num_channels"])
    print("Number of classes  : ", len(parameters["model"]["class_list"]))
    parameters["testing_data"] = inferenceDataFromPickle

    # ensure outputs are saved properly
    parameters["save_output"] = True

    assert (
        parameters["model"]["type"].lower() == "torch"
        or parameters["model"]["type"].lower() == "openvino"
    ), f"The model type is not recognized: {parameters['model']['type']}"

    (model, _, _, _, _, parameters) = create_pytorch_objects(parameters, device=device)

    # Loading the weights into the model
    main_dict = None
    if parameters["model"]["type"].lower() == "torch":
        # Loading the weights into the model
        if os.path.isdir(modelDir):
            files_to_check = [
                os.path.join(
                    modelDir,
                    str(parameters["model"]["architecture"]) + best_model_path_end,
                ),
                os.path.join(
                    modelDir,
                    str(parameters["model"]["architecture"]) + latest_model_path_end,
                ),
            ]

            file_to_load = None
            for best_file in files_to_check:
                if os.path.isfile(best_file):
                    file_to_load = best_file
                    break

            assert file_to_load != None, "The 'best_file' was not found"

        main_dict = torch.load(file_to_load, map_location=parameters["device"])
        state_dict = main_dict["model_state_dict"]
        if parameters.get("differential_privacy"):
            # this is required for torch==1.11 and for DP inference
            new_state_dict = {}
            for key, val in state_dict.items():
                new_key = key.replace("_module.", "")
                new_state_dict[new_key] = val  # remove `module.`
            state_dict = new_state_dict

        model.load_state_dict(state_dict)
        parameters["previous_parameters"] = main_dict.get("parameters", None)
        model.eval()
    elif parameters["model"]["type"].lower() == "openvino":
        # Loading the executable OpenVINO model
        if os.path.isdir(modelDir):
            xml_to_check = os.path.join(
                modelDir, str(parameters["model"]["architecture"]) + "_best.xml"
            )
            bin_to_check = os.path.join(
                modelDir, str(parameters["model"]["architecture"]) + "_best.bin"
            )
            if not os.path.isfile(xml_to_check):
                raise ValueError(
                    "The specified model IR was not found: {0}.".format(xml_to_check)
                )
            if not os.path.isfile(bin_to_check):
                raise ValueError(
                    "The model specified model weights was not found: {0}.".format(
                        bin_to_check
                    )
                )
            model, input_blob, output_blob = load_ov_model(xml_to_check, device.upper())
            parameters["model"]["IO"] = [input_blob, output_blob]

    if not (os.environ.get("HOSTNAME") is None):
        print("\nHostname     :" + str(os.environ.get("HOSTNAME")), flush=True)

    # radiology inference
    if parameters["modality"] == "rad":
        if parameters["model"]["print_summary"]:
            print_model_summary(
                model,
                parameters["batch_size"],
                parameters["model"]["num_channels"],
                parameters["patch_size"],
                parameters["device"],
            )

        # Setting up the inference loader
        inference_loader = get_testing_loader(parameters)

        print("Data Samples: ", len(inference_loader.dataset), flush=True)

        average_epoch_valid_loss, average_epoch_valid_metric = validate_network(
            model, inference_loader, None, parameters, mode="inference"
        )
        print(average_epoch_valid_loss, average_epoch_valid_metric)
    elif parameters["modality"] in ["path", "histo"]:
        # set some defaults
        parameters["stride_size"] = parameters.get("stride_size", None)
        parameters["slide_level"] = parameters.get("slide_level", 0)
        parameters["mask_level"] = parameters.get(
            "mask_level", parameters["slide_level"]
        )
        parameters["blending_alpha"] = float(parameters.get("blending_alpha", 0.5))

        output_to_write = "SubjectID,x_coords,y_coords"
        if parameters["problem_type"] == "regression":
            output_to_write += ",output"
        elif parameters["problem_type"] == "classification":
            for n in range(parameters["model"]["num_classes"]):
                output_to_write += ",probability_" + str(n)
        output_to_write += "\n"

        # actual computation
        pbar = tqdm(inferenceDataFromPickle.iterrows())
        for _, row in pbar:
            subject_name = row[parameters["headers"]["subjectIDHeader"]]
            os_image = openslide.open_slide(
                row[parameters["headers"]["channelHeaders"]].values[0]
            )
            max_defined_slide_level = os_image.level_count - 1
            parameters["slide_level"] = min(
                parameters["slide_level"], max_defined_slide_level
            )
            parameters["slide_level"] = max(parameters["slide_level"], 0)
            level_width, level_height = os_image.level_dimensions[
                parameters["slide_level"]
            ]
            subject_dest_dir = os.path.join(outputDir, str(subject_name))
            Path(subject_dest_dir).mkdir(parents=True, exist_ok=True)

            try:
                count_map, probs_map = None, None
                count_map = np.zeros((level_height, level_width), dtype=np.uint8)
                # this can probably be made into a single multi-class probability map that functions for all workloads
                probs_map = np.zeros(
                    (parameters["model"]["num_classes"], level_height, level_width),
                    dtype=np.float16,
                )
            except Exception as e:
                print(
                    "Could not initialize count and probability maps for subject ID:",
                    subject_name,
                    "; Error:",
                    e,
                    flush=True,
                    file=sys.stderr,
                )

            patch_size = parameters["patch_size"]
            # patch size should be 2D for histology
            if len(patch_size) == 3:
                patch_size = patch_size[:-1]

            transform_requested = get_transforms_for_preprocessing(
                parameters, [], False, False
            )

            pbar.set_description(
                "Constructing loader for subject: " + str(subject_name)
            )

            patient_dataset_obj = InferTumorSegDataset(
                row[parameters["headers"]["channelHeaders"]].values[0],
                patch_size=patch_size,
                stride_size=parameters["stride_size"],
                selected_level=parameters["slide_level"],
                mask_level=parameters["mask_level"],
                transform=transform_requested,
            )

            dataloader = DataLoader(
                patient_dataset_obj,
                batch_size=1,
                shuffle=False,
                num_workers=parameters["q_num_workers"],
            )
            # update patch_size in case microns were requested
            patch_size = patient_dataset_obj.get_patch_size()

            if parameters["model"]["print_summary"]:
                print_model_summary(
                    model,
                    parameters["batch_size"],
                    parameters["model"]["num_channels"],
                    patch_size,
                    parameters["device"],
                )

            pbar.set_description(
                "Looping over patches for subject: " + str(subject_name)
            )

            for image_patches, (x_coords, y_coords) in dataloader:
                x_coords, y_coords = x_coords.numpy(), y_coords.numpy()
                if parameters["model"]["type"] == "torch":
                    if parameters["model"]["amp"]:
                        with autocast():
                            output = model(
                                image_patches.float().to(parameters["device"])
                            )
                    else:
                        output = model(image_patches.float().to(parameters["device"]))
                    output = output.detach().cpu().numpy()
                else:
                    output = model(
                        inputs={
                            parameters["model"]["IO"][0][0]: image_patches.float()
                            .cpu()
                            .numpy()
                        }
                    )[parameters["model"]["IO"][1][0]]

                for i in range(int(output.shape[0])):
                    if count_map is not None:
                        count_map[
                            y_coords[i] : y_coords[i] + patch_size[1],
                            x_coords[i] : x_coords[i] + patch_size[0],
                        ] += 1
                    output_to_write += (
                        str(subject_name)
                        + ","
                        + str(x_coords[i])
                        + ","
                        + str(y_coords[i])
                    )
                    for n in range(parameters["model"]["num_classes"]):
                        # This is a temporary fix for the segmentation problem for single class
                        if probs_map is not None:
                            probs_map[
                                n,
                                y_coords[i] : y_coords[i] + patch_size[1],
                                x_coords[i] : x_coords[i] + patch_size[0],
                            ] += output[i][n]
                        if parameters["problem_type"] != "segmentation":
                            output_to_write += "," + str(output[i][n])
                    output_to_write += "\n"

            # ensure probability map is scaled
            # reusing variables to save memory
            if probs_map is not None:
                probs_map = np.divide(probs_map, count_map)

                # Check if out_probs_map is greater than 1, print a warning
                if np.max(probs_map) > 1:
                    # Print a warning
                    print(
                        "Warning: Probability map is greater than 1, report the images to GaNDLF developers"
                    )

            if count_map is not None:
                count_map = np.array(count_map * 255, dtype=np.uint16)
                imsave(
                    os.path.join(
                        subject_dest_dir,
                        str(row[parameters["headers"]["subjectIDHeader"]])
                        + "_count.png",
                    ),
                    count_map,
                )

            if parameters["problem_type"] != "segmentation":
                output_file = os.path.join(subject_dest_dir, "predictions.csv")
                with open(output_file, "w") as f:
                    f.write(output_to_write)

            heatmaps = {}
            if probs_map is not None:
                try:
                    for n in range(parameters["model"]["num_classes"]):
                        heatmap_gray = np.array(probs_map[n, ...] * 255, dtype=np.uint8)
                        heatmaps[str(n) + "_jet"] = cv2.applyColorMap(
                            heatmap_gray, cv2.COLORMAP_JET
                        )
                        heatmaps[str(n) + "_turbo"] = cv2.applyColorMap(
                            heatmap_gray, cv2.COLORMAP_TURBO
                        )
                        heatmaps[str(n) + "_agni"] = applyCustomColorMap(heatmap_gray)

                        # save the segmentation maps
                        file_to_write = os.path.join(
                            subject_dest_dir, "seg_map_" + str(n) + ".png"
                        )

                        segmap = ((probs_map[n, ...] > 0.5).astype(np.uint8)) * 255

                        cv2.imwrite(file_to_write, segmap)

                    for key in heatmaps:
                        file_to_write = os.path.join(
                            subject_dest_dir, "probability_map" + key + ".png"
                        )
                        cv2.imwrite(file_to_write, heatmaps[key])

                        os_image_array = os_image.read_region(
                            (0, 0),
                            parameters["slide_level"],
                            (level_width, level_height),
                            as_array=True,
                        )
                        blended_image = cv2.addWeighted(
                            os_image_array,
                            parameters["blending_alpha"],
                            heatmaps[key],
                            1 - parameters["blending_alpha"],
                            0,
                        )

                        file_to_write = os.path.join(
                            subject_dest_dir, "probability_map_blended_" + key + ".png"
                        )
                        cv2.imwrite(file_to_write, blended_image)
                except Exception as ex:
                    print("Could not write heatmaps; error:", ex)

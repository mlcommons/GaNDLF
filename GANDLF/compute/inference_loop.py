from .forward_pass import validate_network
from .generic import create_pytorch_objects
import os
from pathlib import Path

# hides torchio citation request, see https://github.com/fepegar/torchio/issues/235
os.environ["TORCHIO_HIDE_CITATION_PROMPT"] = "1"

import pickle, argparse, torch
import numpy as np
from torch.utils.data import DataLoader
from skimage.io import imsave
from tqdm import tqdm
from torch.cuda.amp import autocast
import tiffslide as openslide
from GANDLF.data import get_testing_loader
from GANDLF.utils import (
    populate_channel_keys_in_params,
    get_dataframe,
    best_model_path_end,
    load_ov_model,
)

from GANDLF.data.inference_dataloader_histopath import InferTumorSegDataset
from GANDLF.data.preprocessing import get_transforms_for_preprocessing


def inference_loop(
    inferenceDataFromPickle, device, parameters, outputDir_or_optimizedModel
):
    """
    The main training loop.

    Args:
        inferenceDataFromPickle (pandas.DataFrame): The data to use for inference.
        device (str): The device to perform computations on.
        parameters (dict): The parameters dictionary.
        outputDir_or_optimizedModel (str): The output directory or optimized model file.
    """
    # Defining our model here according to parameters mentioned in the configuration file
    print("Current model type : ", parameters["model"]["type"])
    print("Number of dims     : ", parameters["model"]["dimension"])
    if "num_channels" in parameters["model"]:
        print("Number of channels : ", parameters["model"]["num_channels"])
    print("Number of classes  : ", len(parameters["model"]["class_list"]))
    parameters["testing_data"] = inferenceDataFromPickle

    (
        model,
        _,
        _,
        _,
        _,
        parameters,
    ) = create_pytorch_objects(parameters, device=device)

    # ensure outputs are saved properly
    parameters["save_output"] = True

    # Loading the weights into the model
    main_dict = None
    if parameters["model"]["type"] == "torch":
        # Loading the weights into the model
        if os.path.isdir(outputDir_or_optimizedModel):
            file_to_check = os.path.join(
                outputDir_or_optimizedModel,
                str(parameters["model"]["architecture"]) + best_model_path_end,
            )
            if not os.path.isfile(file_to_check):
                raise ValueError(
                    "The specified model was not found: {0}.".format(file_to_check)
                )

        main_dict = torch.load(file_to_check)
        model.load_state_dict(main_dict["model_state_dict"])
        model.eval()
    elif parameters["model"]["type"].lower() == "openvino":
        # Loading the executable OpenVINO model
        if os.path.isdir(outputDir_or_optimizedModel):
            xml_to_check = os.path.join(
                outputDir_or_optimizedModel,
                str(parameters["model"]["architecture"]) + "_best.xml",
            )
            bin_to_check = os.path.join(
                outputDir_or_optimizedModel,
                str(parameters["model"]["architecture"]) + "_best.bin",
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
    else:
        raise ValueError(
            "The model type is not recognized: ", parameters["model"]["type"]
        )

    if not (os.environ.get("HOSTNAME") is None):
        print("\nHostname     :" + str(os.environ.get("HOSTNAME")), flush=True)

    # radiology inference
    if parameters["modality"] == "rad":
        # Setting up the inference loader
        inference_loader = get_testing_loader(parameters)

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
            level_width, level_height = os_image.level_dimensions[
                int(parameters["slide_level"])
            ]
            subject_dest_dir = os.path.join(
                outputDir_or_optimizedModel, str(subject_name)
            )
            Path(subject_dest_dir).mkdir(parents=True, exist_ok=True)

            count_map = np.zeros((level_height, level_width), dtype=np.uint8)
            ## this can probably be made into a single multi-class probability map that functions for all workloads
            if parameters["problem_type"] == "segmentation":
                probs_map = np.zeros((level_height, level_width), dtype=np.float16)
            else:
                probs_map = np.zeros(
                    (parameters["model"]["num_classes"], level_height, level_width),
                    dtype=np.float16,
                )

            patch_size = parameters["patch_size"]

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

            pbar.set_description(
                "Looping over patches for subject: " + str(subject_name)
            )

            for image_patches, (x_coords, y_coords) in dataloader:
                x_coords, y_coords = y_coords.numpy(), x_coords.numpy()
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
                    count_map[
                        x_coords[i] : x_coords[i] + patch_size[0],
                        y_coords[i] : y_coords[i] + patch_size[1],
                    ] += 1
                    if parameters["problem_type"] == "segmentation":
                        for n in range(parameters["model"]["num_classes"]):
                            probs_map[
                                x_coords[i] : x_coords[i] + patch_size[0],
                                y_coords[i] : y_coords[i] + patch_size[1],
                            ] += output[i][n] # This is a temporary fix for the segmentation problem for single class
                    else:
                        output_to_write += (
                            str(subject_name)
                            + ","
                            + str(x_coords[i])
                            + ","
                            + str(y_coords[i])
                        )
                        for n in range(parameters["model"]["num_classes"]):
                            probs_map[
                                n,
                                x_coords[i] : x_coords[i] + patch_size[0],
                                y_coords[i] : y_coords[i] + patch_size[1],
                            ] += output[i][n]
                            output_to_write += "," + str(output[i][n])
                        # ensure csv row terminates in new line
                        output_to_write += "\n"

            # ensure probability map is scaled
            # Updating variables to save memory
            probs_map = np.divide(probs_map, count_map)

            # Check if out_probs_map is greater than 1, print a warning
            if np.max(probs_map) > 1:
                # Print a warning
                print("Warning: Probability map is greater than 1, report the images to GANDLF developers")

            count_map = np.array(count_map * 255, dtype=np.uint16)
            imsave(
                os.path.join(
                    subject_dest_dir,
                    str(row[parameters["headers"]["subjectIDHeader"]])
                    + "_count.png",
                ),
                count_map,
            )

            import cv2

            if parameters["problem_type"] == "segmentation":
                out_thresh = np.array((probs_map > 0.5) * 255, dtype=np.uint16)
                output_file = os.path.join(
                    subject_dest_dir,
                    "predictions.csv",
                )
                with open(output_file, "w") as f:
                    f.write(output_to_write)
                
                for n in range(parameters["model"]["num_classes"]):
                    file_to_write = os.path.join(
                        subject_dest_dir, "probability_map_" + str(n) + ".png"
                    )
                    heatmap = cv2.applyColorMap(np.array(
                        probs_map[n, ...] * 255,
                        dtype=np.uint8,
                    ), cv2.COLORMAP_JET)
                    cv2.imwrite(file_to_write, heatmap)

                    file_to_write = os.path.join(
                        subject_dest_dir, "seg_map_" + str(n) + ".png"
                    )
                    
                    segmap = cv2.applyColorMap(np.array(
                        ((probs_map[n, ...] > 0.5).astype(np.uint8)) * 255),
                        dtype=np.uint8,
                    )
                    cv2.imwrite(file_to_write, segmap)
            else:
                output_file = os.path.join(
                    subject_dest_dir,
                    "predictions.csv",
                )
                with open(output_file, "w") as f:
                    f.write(output_to_write)

                for n in range(parameters["model"]["num_classes"]):
                    file_to_write = os.path.join(
                        subject_dest_dir, "probability_map_" + str(n) + ".png"
                    )
                    heatmap = cv2.applyColorMap(np.array(
                        probs_map[n, ...] * 255,
                        dtype=np.uint8,
                    ), cv2.COLORMAP_JET)
                    cv2.imwrite(file_to_write, heatmap)

                    file_to_write = os.path.join(
                        subject_dest_dir, "seg_map_" + str(n) + ".png"
                    )
                    
                    segmap = cv2.applyColorMap(np.array(
                        ((probs_map[n, ...] > 0.5).astype(np.uint8)) * 255),
                        dtype=np.uint8,
                    )
                    cv2.imwrite(file_to_write, segmap)


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
    inferenceDataFromPickle = get_dataframe(args.inference_loader_pickle)

    inference_loop(
        inferenceDataFromPickle=inferenceDataFromPickle,
        parameters=parameters,
        outputDir_or_optimizedModel=args.outputDir,
        device=args.device,
    )

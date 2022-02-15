from .forward_pass import validate_network
import os

# hides torchio citation request, see https://github.com/fepegar/torchio/issues/235
os.environ["TORCHIO_HIDE_CITATION_PROMPT"] = "1"

import pickle, argparse, torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from skimage.io import imsave
from tqdm import tqdm
from torch.cuda.amp import autocast
from GANDLF.data.ImagesFromDataFrame import ImagesFromDataFrame
from GANDLF.utils import populate_channel_keys_in_params, send_model_to_device, load_ov_model
from GANDLF.models import global_models_dict
from GANDLF.utils import (
        populate_channel_keys_in_params, 
        send_model_to_device, 
        load_ov_model,
        )

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
    print("Current model type : ", parameters["model"]["type"])
    print("Number of dims     : ", parameters["model"]["dimension"])
    if "num_channels" in parameters["model"]:
        print("Number of channels : ", parameters["model"]["num_channels"])
    print("Number of classes  : ", len(parameters["model"]["class_list"]))

    # Fetch the model according to params mentioned in the configuration file
    model = global_models_dict[parameters["model"]["architecture"]](
        parameters=parameters
    )

    # Setting up the inference loader
    inferenceDataForTorch = ImagesFromDataFrame(
        inferenceDataFromPickle, parameters, train=False, loader_type="inference"
    )
    inference_loader = DataLoader(inferenceDataForTorch, batch_size=1)

    if parameters["model"]["type"] == "Torch":
        # Loading the weights into the model
        main_dict = outputDir
        if os.path.isdir(outputDir):
            file_to_check = os.path.join(
                outputDir, str(parameters["model"]["architecture"]) + "_best.pth.tar"
            )
            if not os.path.isfile(file_to_check):
                raise ValueError(
                        "The model specified model was not found:", file_to_check
                        )

        main_dict = torch.load(file_to_check, map_location=torch.device(device))
        model.load_state_dict(main_dict["model_state_dict"])
    elif parameters["model"]["type"].lower() == "openvino":
        # Loading the executable OpenVINO model
        main_dict = outputDir
        if os.path.isdir(outputDir):
            xml_to_check = os.path.join(
                outputDir, str(parameters["model"]["architecture"]) + "_best.xml"
            )
            bin_to_check = os.path.join(
                outputDir, str(parameters["model"]["architecture"]) + "_best.bin"
            )
            if not os.path.isfile(xml_to_check):
                raise ValueError("The model specified model IR was not found:", xml_to_check)
            if not os.path.isfile(bin_to_check):
                raise ValueError("The model specified model weights was not found:", bin_to_check)
            model, input_blob, output_blob = load_ov_model(xml_to_check, device.upper())
            parameters['model']['IO'] = [input_blob, output_blob]
    else:
        raise ValueError(
                "The model type is not recognized: ", parameters["model"]["type"]
                )
        
    if not (os.environ.get("HOSTNAME") is None):
        print("\nHostname     :" + str(os.environ.get("HOSTNAME")), flush=True)

    # get the channel keys for concatenation later (exclude non numeric channel keys)
    parameters = populate_channel_keys_in_params(inference_loader, parameters)
    parameters["save_output"] = True

    print("Data Samples: ", len(inference_loader.dataset), flush=True)
    if parameters["model"]["type"] == "Torch":
        model, parameters["model"]["amp"], parameters["device"] = send_model_to_device(
            model, parameters["model"]["amp"], device, optimizer=None
        )

    print("Using device:", parameters["device"], flush=True)

    # radiology inference
    if parameters["modality"] == "rad":
        average_epoch_valid_loss, average_epoch_valid_metric = validate_network(
            model, inference_loader, None, parameters, mode="inference"
        )
        print(average_epoch_valid_loss, average_epoch_valid_metric)
    elif (parameters["modality"] == "path") or (parameters["modality"] == "histo"):
        # histology inference
        if os.name != "nt":
            """
            path inference is Linux-only because openslide for Windows works only for Python-3.8  whereas pickle5 works only for 3.6 and 3.7
            """
            from GANDLF.data.inference_dataloader_histopath import InferTumorSegDataset
            from openslide import OpenSlide

            # actual computation
            for _, row in inferenceDataForTorch.iterrows():
                subject_name = row[parameters["headers"]["subjectIDHeader"]]
                print(
                    "Patient Slide       : ",
                    row[parameters["headers"]["subjectIDHeader"]],
                )
                print(
                    "Patient Location    : ",
                    row[parameters["headers"]["channelHeaders"]],
                )
                print(row[parameters["headers"]["channelHeaders"]].values[0])
                os_image = OpenSlide(
                    row[parameters["headers"]["channelHeaders"]].values[0]
                )
                level_width, level_height = os_image.level_dimensions[
                    int(parameters["slide_level"])
                ]
                subject_dest_dir = os.path.join(outputDir, subject_name)
                os.makedirs(subject_dest_dir, exist_ok=True)

                probs_map = np.zeros((level_height, level_width), dtype=np.float16)
                count_map = np.zeros((level_height, level_width), dtype=np.uint8)

                patient_dataset_obj = InferTumorSegDataset(
                    row[parameters["headers"]["channelHeaders"]].values[0],
                    patch_size=patch_size,
                    stride_size=parameters["stride_size"],
                    selected_level=parameters["slide_level"],
                    mask_level=4,
                )

                dataloader = DataLoader(
                    patient_dataset_obj,
                    batch_size=int(parameters["batch_size"]),
                    shuffle=False,
                    num_workers=parameters["q_num_workers"],
                )
                for image_patches, (x_coords, y_coords) in tqdm(dataloader):
                    x_coords, y_coords = y_coords.numpy(), x_coords.numpy()
                    if parameters["model"]["type"] == "Torch":
                        if parameters["model"]["amp"]:
                            with autocast():
                                output = model(
                                    image_patches.float().to(parameters["device"])
                                )
                        else:
                            output = model(
                                    image_patches.float().to(parameters["device"])
                                    )
                        output = output.detach().cpu().numpy()
                    else:
                        output = model.infer( 
                                inputs={
                                    params["model"]["IO"][0]:image_patches.float()
                                    .cpu()
                                    .numpy()
                                    }
                                )[params["model"]["IO"][1]]

                    for i in range(int(output.shape[0])):
                        count_map[
                            x_coords[i] : x_coords[i] + patch_size[0],
                            y_coords[i] : y_coords[i] + patch_size[1],
                        ] += 1
                        probs_map[
                            x_coords[i] : x_coords[i] + patch_size[0],
                            y_coords[i] : y_coords[i] + patch_size[1],
                        ] += output[i][0]
                probs_map = probs_map / count_map
                count_map = count_map / count_map.max()
                out = count_map * probs_map
                count_map = np.array(count_map * 255, dtype=np.uint16)
                out_thresh = np.array((out > 0.5) * 255, dtype=np.uint16)
                imsave(
                    os.path.join(
                        subject_dest_dir,
                        row[parameters["headers"]["subjectIDHeader"]] + "_prob.png",
                    ),
                    out,
                )
                imsave(
                    os.path.join(
                        subject_dest_dir,
                        row[parameters["headers"]["subjectIDHeader"]] + "_seg.png",
                    ),
                    out_thresh,
                )
                imsave(
                    os.path.join(
                        subject_dest_dir,
                        row[parameters["headers"]["subjectIDHeader"]] + "_count.png",
                    ),
                    count_map,
                )
        else:
            print(
                "ERROR: histo/path inference is Linux-only because openslide for Windows works only for Python-3.8, whereas pickle5 works only for 3.6 and 3.7"
            )


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

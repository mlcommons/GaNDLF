import os, argparse, sys, pickle
from pathlib import Path
from datetime import date
import numpy as np
import SimpleITK as sitk
import pandas as pd

from GANDLF.utils import (
    get_filename_extension_sanitized,
    parseTrainingCSV,
    populate_header_in_parameters,
)
from GANDLF.parseConfig import parseConfig
from GANDLF.data.ImagesFromDataFrame import ImagesFromDataFrame
from torch.utils.data import DataLoader
from tqdm import tqdm

import torchio


def preprocess_and_save(data_csv, config_file, output_dir, label_pad_mode="constant"):
    """
    This function performs preprocessing based on parameters provided and saves the output.

    Args:
        data_csv (str): The CSV file of the training data.
        config_file (str): The YAML file of the training configuration.
        output_dir (str): The output directory.
        label_pad_mode (str): The padding strategy for the label. Defaults to "constant".

    Raises:
        ValueError: Parameter check from previous run.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # read the csv
    # don't care if the dataframe gets shuffled or not
    dataframe, headers = parseTrainingCSV(data_csv, train=False)
    parameters = parseConfig(config_file)

    # save the parameters so that the same compute doesn't happen once again
    parameter_file = os.path.join(output_dir, "parameters.pkl")
    if os.path.exists(parameter_file):
        parameters_prev = pickle.load(open(parameter_file, "rb"))
        if parameters != parameters_prev:
            raise ValueError(
                "The parameters are not the same as the ones stored in the previous run, please re-check."
            )
    else:
        with open(parameter_file, "wb") as handle:
            pickle.dump(parameters, handle, protocol=pickle.HIGHEST_PROTOCOL)

    parameters = populate_header_in_parameters(parameters, headers)

    data_for_processing = ImagesFromDataFrame(dataframe, parameters, train=False)

    dataloader_for_processing = DataLoader(
        data_for_processing,
        batch_size=1,
        pin_memory=False,
    )

    # initialize a new dict for the preprocessed data
    base_df = pd.read_csv(data_csv)
    # ensure csv only contains lower case columns
    base_df.columns = base_df.columns.str.lower()
    # only store the column names
    output_columns_to_write = base_df.to_dict()
    for key in output_columns_to_write.keys():
        output_columns_to_write[key] = []

    # keep a record of the keys which contains only images
    keys_with_images = parameters["headers"]["channelHeaders"]
    keys_with_images = [str(x) for x in keys_with_images]

    ## to-do
    # use dataloader_for_processing to loop through all images
    # if padding is enabled, ensure that it gets applied to the images
    # save the images to disk, but keep a record that these images are preprocessed.
    # create new csv that contains new files.

    # give warning if label sampler is present but number of patches to extract is > 1
    if (
        (parameters["patch_sampler"] == "label")
        or (isinstance(parameters["patch_sampler"], dict))
    ) and parameters["q_samples_per_volume"] > 1:
        print(
            "[WARNING] Label sampling has been enabled but q_samples_per_volume > 1; this has been known to cause issues, so q_samples_per_volume will be hard-coded to 1 during preprocessing. Please contact GaNDLF developers for more information",
            file=sys.stderr,
            flush=True,
        )

    for _, (subject) in enumerate(
        tqdm(dataloader_for_processing, desc="Looping over data")
    ):
        # initialize the current_output_dir
        current_output_dir = os.path.join(output_dir, str(subject["subject_id"][0]))
        Path(current_output_dir).mkdir(parents=True, exist_ok=True)

        output_columns_to_write["subjectid"].append(subject["subject_id"][0])

        subject_dict_to_write, subject_process = {}, {}

        # start constructing the torchio.Subject object
        for channel in parameters["headers"]["channelHeaders"]:
            # the "squeeze" is needed because the dataloader automatically
            # constructs 5D tensor considering the batch_size as first
            # dimension, but the constructor needs 4D tensor.
            subject_process[str(channel)] = torchio.Image(
                tensor=subject[str(channel)]["data"].squeeze(0),
                type=torchio.INTENSITY,
                path=subject[str(channel)]["path"],
            )
        if parameters["headers"]["labelHeader"] is not None:
            subject_process["label"] = torchio.Image(
                tensor=subject["label"]["data"].squeeze(0),
                type=torchio.LABEL,
                path=subject["label"]["path"],
            )
        subject_dict_to_write = torchio.Subject(subject_process)

        # apply a different padding mode to image and label (so that label information is not duplicated)
        if (parameters["patch_sampler"] == "label") or (
            isinstance(parameters["patch_sampler"], dict)
        ):
            # get the padding size from the patch_size
            psize_pad = list(
                np.asarray(np.ceil(np.divide(parameters["patch_size"], 2)), dtype=int)
            )
            # initialize the padder for images
            padder = torchio.transforms.Pad(
                psize_pad, padding_mode="symmetric", include=keys_with_images
            )
            subject_dict_to_write = padder(subject_dict_to_write)

            if parameters["headers"]["labelHeader"] is not None:
                # initialize the padder for label
                padder_label = torchio.transforms.Pad(
                    psize_pad, padding_mode=label_pad_mode, include="label"
                )
                subject_dict_to_write = padder_label(subject_dict_to_write)

                sampler = torchio.data.LabelSampler(parameters["patch_size"])
                generator = sampler(subject_dict_to_write, num_patches=1)
                for patch in generator:
                    for channel in parameters["headers"]["channelHeaders"]:
                        subject_dict_to_write[str(channel)] = patch[str(channel)]

                    subject_dict_to_write["label"] = patch["label"]

        # write new images
        common_ext = get_filename_extension_sanitized(subject["1"]["path"][0])
        for channel in parameters["headers"]["channelHeaders"]:
            image_file = Path(
                os.path.join(
                    current_output_dir,
                    subject["subject_id"][0] + "_" + str(channel) + common_ext,
                )
            ).as_posix()
            output_columns_to_write["channel_" + str(channel - 1)].append(image_file)
            image_to_write = subject_dict_to_write[str(channel)].as_sitk()
            if not os.path.isfile(image_file):
                try:
                    sitk.WriteImage(image_to_write, image_file)
                except IOError:
                    IOError(
                        "Could not write image file: {}. Make sure that the file is not open and try again.".format(
                            image_file
                        )
                    )
                    sys.exit(1)

        # now try to write the label
        if "label" in subject_dict_to_write:
            image_file = Path(
                os.path.join(
                    current_output_dir, subject["subject_id"][0] + "_label" + common_ext
                )
            ).as_posix()
            output_columns_to_write["label"].append(image_file)
            image_to_write = subject_dict_to_write["label"].as_sitk()
            if not os.path.isfile(image_file):
                try:
                    sitk.WriteImage(image_to_write, image_file)
                except IOError:
                    IOError(
                        "Could not write image file: {}. Make sure that the file is not open and try again.".format(
                            image_file
                        )
                    )
                    sys.exit(1)

        # ensure prediction headers are getting saved, as well
        if len(parameters["headers"]["predictionHeaders"]) > 1:
            for key in parameters["headers"]["predictionHeaders"]:
                output_columns_to_write["valuetopredict_" + str(key)].append(
                    str(subject["value_" + str(key)].numpy()[0])
                )
        elif len(parameters["headers"]["predictionHeaders"]) == 1:
            output_columns_to_write["valuetopredict"].append(
                str(subject["value_0"].numpy()[0])
            )

    path_for_csv = Path(os.path.join(output_dir, "data_processed.csv")).as_posix()
    print("Writing final csv for subsequent training: ", path_for_csv)
    pd.DataFrame.from_dict(data=output_columns_to_write, orient="index").to_csv(
        path_for_csv, header=False
    )

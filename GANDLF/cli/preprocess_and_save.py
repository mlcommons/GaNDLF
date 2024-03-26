import os, sys, pickle
from typing import Optional
from pathlib import Path
import SimpleITK as sitk

from GANDLF.utils import (
    get_filename_extension_sanitized,
    parseTrainingCSV,
    populate_header_in_parameters,
    get_dataframe,
    get_correct_padding_size,
)
from GANDLF.config_manager import ConfigManager
from GANDLF.data.ImagesFromDataFrame import ImagesFromDataFrame
from torch.utils.data import DataLoader
from tqdm import tqdm

import torchio


def preprocess_and_save(
    data_csv: str,
    config_file: str,
    output_dir: str,
    label_pad_mode: Optional[str] = "constant",
    applyaugs: Optional[bool] = False,
    apply_zero_crop: Optional[bool] = False,
) -> None:
    """
    This function performs preprocessing based on parameters provided and saves the output.

    Args:
        data_csv (str): The CSV file of the training data.
        config_file (str): The YAML file of the training configuration.
        output_dir (str): The output directory.
        label_pad_mode (Optional[str], optional): The padding mode for the label. Defaults to "constant".
        applyaugs (Optional[bool], optional): Whether to apply augmentations. Defaults to False.
        apply_zero_crop (Optional[bool], optional): Whether to apply zero crop. Defaults to False.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # read the csv
    # don't care if the dataframe gets shuffled or not
    dataframe, headers = parseTrainingCSV(data_csv, train=False)
    parameters = ConfigManager(config_file)

    # save the parameters so that the same compute doesn't happen once again
    parameter_file = os.path.join(output_dir, "parameters.pkl")
    if os.path.exists(parameter_file):
        parameters_prev = pickle.load(open(parameter_file, "rb"))
        assert (
            parameters == parameters_prev
        ), "The parameters are not the same as the ones stored in the previous run, please re-check."
    else:
        with open(parameter_file, "wb") as handle:
            pickle.dump(parameters, handle, protocol=pickle.HIGHEST_PROTOCOL)

    parameters = populate_header_in_parameters(parameters, headers)

    data_for_processing = ImagesFromDataFrame(
        dataframe,
        parameters,
        train=applyaugs,
        apply_zero_crop=apply_zero_crop,
        loader_type="full",
    )

    dataloader_for_processing = DataLoader(
        data_for_processing, batch_size=1, pin_memory=False
    )

    # initialize a new dict for the preprocessed data
    base_df = get_dataframe(data_csv)
    # ensure csv only contains lower case columns
    base_df.columns = base_df.columns.str.lower()
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
        current_output_dir = os.path.abspath(
            os.path.join(output_dir, str(subject["subject_id"][0]))
        )
        Path(current_output_dir).mkdir(parents=True, exist_ok=True)

        subject_dict_to_write, subject_process = {}, {}

        # start constructing the torchio.Subject object
        for channel in parameters["headers"]["channelHeaders"]:
            # the "squeeze" is needed because the dataloader automatically
            # constructs 5D tensor considering the batch_size as first
            # dimension, but the constructor needs 4D tensor.
            subject_process[str(channel)] = torchio.ScalarImage(
                tensor=subject[str(channel)]["data"].squeeze(0),
                path=subject[str(channel)]["path"],
            )
        if parameters["headers"]["labelHeader"] is not None:
            subject_process["label"] = torchio.LabelMap(
                tensor=subject["label"]["data"].squeeze(0),
                path=subject["label"]["path"],
            )
        subject_dict_to_write = torchio.Subject(subject_process)

        # apply a different padding mode to image and label (so that label information is not duplicated)
        if parameters["patch_sampler"]["type"] == "label":
            # get the padding size from the patch_size
            psize_pad = get_correct_padding_size(
                parameters["patch_size"], parameters["model"]["dimension"]
            )
            # initialize the padder for images
            padder = torchio.transforms.Pad(
                psize_pad,
                padding_mode=label_pad_mode,
                include=keys_with_images + ["label"],
            )
            subject_dict_to_write = padder(subject_dict_to_write)

        # write new images
        common_ext = get_filename_extension_sanitized(subject["path_to_metadata"][0])
        # in cases where the original image has a file format that does not support
        # RGB floats, use the "vtk" format
        if common_ext in [".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"]:
            common_ext = ".vtk"

        image_for_info_copy = subject_dict_to_write[
            str(parameters["headers"]["channelHeaders"][0])
        ].as_sitk()
        for index, channel in enumerate(parameters["headers"]["channelHeaders"]):
            image_file = Path(
                os.path.join(
                    current_output_dir,
                    subject["subject_id"][0] + "_" + str(index) + common_ext,
                )
            ).as_posix()
            base_df["channel_" + str(index)] = image_file
            image_to_write = subject_dict_to_write[str(channel)].as_sitk()
            image_to_write.SetOrigin(image_for_info_copy.GetOrigin())
            image_to_write.SetDirection(image_for_info_copy.GetDirection())
            image_to_write.SetSpacing(image_for_info_copy.GetSpacing())
            if not os.path.isfile(image_file):
                try:
                    sitk.WriteImage(image_to_write, image_file)
                except IOError:
                    raise IOError(
                        "Could not write image file: {}. Make sure that the file is not open and try again.".format(
                            image_file
                        )
                    )

        # now try to write the label
        if "label" in subject_dict_to_write:
            image_file = Path(
                os.path.join(
                    current_output_dir, subject["subject_id"][0] + "_label" + common_ext
                )
            ).as_posix()
            base_df["label"] = image_file
            image_to_write = subject_dict_to_write["label"].as_sitk()
            image_to_write.SetOrigin(image_for_info_copy.GetOrigin())
            image_to_write.SetDirection(image_for_info_copy.GetDirection())
            image_to_write.SetSpacing(image_for_info_copy.GetSpacing())
            if not os.path.isfile(image_file):
                try:
                    sitk.WriteImage(image_to_write, image_file)
                except IOError:
                    raise IOError(
                        "Could not write image file: {}. Make sure that the file is not open and try again.".format(
                            image_file
                        )
                    )

        # ensure prediction headers are getting saved, as well
        if len(parameters["headers"]["predictionHeaders"]) > 1:
            for key in parameters["headers"]["predictionHeaders"]:
                base_df["valuetopredict_" + str(key)] = str(
                    subject["value_" + str(key)].numpy()[0]
                )
        elif len(parameters["headers"]["predictionHeaders"]) == 1:
            base_df["valuetopredict"] = str(subject["value_0"].numpy()[0])

    path_for_csv = Path(os.path.join(output_dir, "data_processed.csv")).as_posix()
    print("Writing final csv for subsequent training: ", path_for_csv)
    base_df.to_csv(path_for_csv, header=True, index=False)

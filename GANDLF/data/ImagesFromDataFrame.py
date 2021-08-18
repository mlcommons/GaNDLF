import os, sys
import numpy as np

import torchio
from torchio.transforms import (
    Resample,
    Compose,
    Pad,
    ToCanonical,
)
from torchio import Image, Subject
import SimpleITK as sitk

from GANDLF.utils import (
    perform_sanity_check_on_subject,
    get_tensor_for_dataloader,
    resize_image,
)
from .preprocessing import global_preprocessing_dict
from .augmentation import global_augs_dict

global_sampler_dict = {
    "uniform": torchio.data.UniformSampler,
    "uniformsampler": torchio.data.UniformSampler,
    "uniformsample": torchio.data.UniformSampler,
    "label": torchio.data.LabelSampler,
    "labelsampler": torchio.data.LabelSampler,
    "labelsample": torchio.data.LabelSampler,
    "weighted": torchio.data.WeightedSampler,
    "weightedsampler": torchio.data.WeightedSampler,
    "weightedsample": torchio.data.WeightedSampler,
}

# This function takes in a dataframe, with some other parameters and returns the dataloader
def ImagesFromDataFrame(dataframe, parameters, train):
    """
    Reads the pandas dataframe and gives the dataloader to use for training/validation/testing

    Parameters
    ----------
    dataframe : pandas.DataFrame
        The main input dataframe which is calculated after splitting the data CSV
    parameters : dict
        The parameters dictionary
    train : bool
        If the dataloader is for training or not. For training, the patching infrastructure and data augmentation is applied.

    Returns
    -------
    subjects_dataset: torchio.SubjectsDataset
        This is the output for validation/testing, where patching and data augmentation is disregarded
    patches_queue: torchio.Queue
        This is the output for training, which is the subjects_dataset queue after patching and data augmentation is taken into account
    """
    # store in previous variable names
    patch_size = parameters["patch_size"]
    headers = parameters["headers"]
    q_max_length = parameters["q_max_length"]
    q_samples_per_volume = parameters["q_samples_per_volume"]
    q_num_workers = parameters["q_num_workers"]
    q_verbose = parameters["q_verbose"]
    sampler = parameters["patch_sampler"]
    augmentations = parameters["data_augmentation"]
    preprocessing = parameters["data_preprocessing"]
    in_memory = parameters["in_memory"]
    enable_padding = parameters["enable_padding"]

    # Finding the dimension of the dataframe for computational purposes later
    num_row, num_col = dataframe.shape
    # changing the column indices to make it easier
    dataframe.columns = range(0, num_col)
    dataframe.index = range(0, num_row)
    # This list will later contain the list of subjects
    subjects_list = []

    channelHeaders = headers["channelHeaders"]
    labelHeader = headers["labelHeader"]
    predictionHeaders = headers["predictionHeaders"]
    subjectIDHeader = headers["subjectIDHeader"]

    # this basically means that label sampler is selected with padding
    if isinstance(sampler, dict):
        sampler_padding = sampler["label"]["padding_type"]
        sampler = "label"
    else:
        sampler = sampler.lower()  # for easier parsing
        sampler_padding = "symmetric"

    resize_images = False
    # if resize has been defined but resample is not (or is none)
    if not (preprocessing is None) and ("resize" in preprocessing):
        if preprocessing["resize"] is not None:
            if not ("resample" in preprocessing):
                resize_images = True
            else:
                print(
                    "WARNING: 'resize' is ignored as 'resample' is defined under 'data_processing', this will be skipped",
                    file=sys.stderr,
                )
    # iterating through the dataframe
    for patient in range(num_row):
        # We need this dict for storing the meta data for each subject
        # such as different image modalities, labels, any other data
        subject_dict = {}
        subject_dict["subject_id"] = str(dataframe[subjectIDHeader][patient])
        skip_subject = False
        # iterating through the channels/modalities/timepoints of the subject
        for channel in channelHeaders:
            # sanity check for malformed csv
            if not os.path.isfile(str(dataframe[channel][patient])):
                skip_subject = True

            # assigning the dict key to the channel
            subject_dict[str(channel)] = Image(
                type=torchio.INTENSITY,
                path=dataframe[channel][patient],
            )

            # store image spacing information if not present
            if "spacing" not in subject_dict:
                file_reader = sitk.ImageFileReader()
                file_reader.SetFileName(dataframe[channel][patient])
                file_reader.ReadImageInformation()
                subject_dict["spacing"] = file_reader.GetSpacing()

            # if resize is requested, the perform per-image resize with appropriate interpolator
            if resize_images:
                img = subject_dict[str(channel)].as_sitk()
                img_resized = resize_image(img, preprocessing["resize"])
                # always ensure resized image spacing is used
                subject_dict["spacing"] = img_resized.GetSpacing()
                img_tensor = get_tensor_for_dataloader(img_resized)
                subject_dict[str(channel)] = Image(
                    tensor=img_tensor,
                    type=torchio.INTENSITY,
                    path=dataframe[channel][patient],
                )

        # # for regression
        # if predictionHeaders:
        #     # get the mask
        #     if (subject_dict['label'] is None) and (class_list is not None):
        #         sys.exit('The \'class_list\' parameter has been defined but a label file is not present for patient: ', patient)

        if labelHeader is not None:
            if not os.path.isfile(str(dataframe[labelHeader][patient])):
                skip_subject = True

            subject_dict["label"] = Image(
                type=torchio.LABEL,
                path=dataframe[labelHeader][patient],
            )

            # if resize is requested, the perform per-image resize with appropriate interpolator
            if resize_images:
                img = sitk.ReadImage(str(dataframe[labelHeader][patient]))
                img_resized = resize_image(img, preprocessing["resize"])
                img_tensor = get_tensor_for_dataloader(img_resized)
                subject_dict["label"] = Image(
                    tensor=img_tensor,
                    type=torchio.LABEL,
                    path=dataframe[channel][patient],
                )

            subject_dict["path_to_metadata"] = str(dataframe[labelHeader][patient])
        else:
            subject_dict["label"] = "NA"
            subject_dict["path_to_metadata"] = str(dataframe[channel][patient])

        # iterating through the values to predict of the subject
        valueCounter = 0
        for values in predictionHeaders:
            # assigning the dict key to the channel
            subject_dict["value_" + str(valueCounter)] = np.array(
                dataframe[values][patient]
            )
            valueCounter = valueCounter + 1

        # skip subject the condition was tripped
        if not skip_subject:
            # Initializing the subject object using the dict
            subject = Subject(subject_dict)
            # https://github.com/fepegar/torchio/discussions/587#discussioncomment-928834
            # this is causing memory usage to explode, see https://github.com/CBICA/GaNDLF/issues/128
            if parameters["verbose"]:
                print(
                    "Checking consistency of images in subject '"
                    + subject["subject_id"]
                    + "'"
                )
            perform_sanity_check_on_subject(subject, parameters)

            # # padding image, but only for label sampler, because we don't want to pad for uniform
            if "label" in sampler or "weight" in sampler:
                if enable_padding:
                    psize_pad = list(
                        np.asarray(np.ceil(np.divide(patch_size, 2)), dtype=int)
                    )
                    # for modes: https://numpy.org/doc/stable/reference/generated/numpy.pad.html
                    padder = Pad(psize_pad, padding_mode=sampler_padding)
                    subject = padder(subject)

            # load subject into memory: https://github.com/fepegar/torchio/discussions/568#discussioncomment-859027
            if in_memory:
                subject.load()

            # Appending this subject to the list of subjects
            subjects_list.append(subject)

    augmentation_list = []

    # first, we want to do thresholding, followed by clipping, if it is present - required for inference as well
    if not (preprocessing is None):
        if "to_canonical" in preprocessing:
            augmentation_list.append(ToCanonical())

        if train:  # we want the crop to only happen during training
            if "crop_external_zero_planes" in preprocessing:
                augmentation_list.append(
                    global_preprocessing_dict["crop_external_zero_planes"](patch_size)
                )
        for key in ["threshold", "clip"]:
            if key in preprocessing:
                augmentation_list.append(
                    global_preprocessing_dict[key](
                        min_thresh=preprocessing[key]["min"],
                        max_thresh=preprocessing[key]["max"],
                    )
                )

        # first, we want to do the resampling, if it is present - required for inference as well
        if "resample" in preprocessing:
            if "resolution" in preprocessing["resample"]:
                # resample_split = str(aug).split(':')
                resample_values = tuple(
                    np.array(preprocessing["resample"]["resolution"]).astype(np.float)
                )
                if len(resample_values) == 2:
                    resample_values = tuple(np.append(resample_values, 1))
                augmentation_list.append(Resample(resample_values))

        # next, we want to do the intensity normalize - required for inference as well
        if "normalize" in preprocessing:
            augmentation_list.append(global_preprocessing_dict["normalize"])
        elif "normalize_nonZero" in preprocessing:
            augmentation_list.append(global_preprocessing_dict["normalize_nonZero"])
        elif "normalize_nonZero_masked" in preprocessing:
            augmentation_list.append(
                global_preprocessing_dict["normalize_nonZero_masked"]
            )

    # other augmentations should only happen for training - and also setting the probabilities
    # for the augmentations
    if train and not (augmentations == None):
        for aug in augmentations:
            if aug != "default_probability":
                actual_function = None

                if aug == "flip":
                    if "axes_to_flip" in augmentations[aug]:
                        print(
                            "WARNING: 'flip' augmentation needs the key 'axis' instead of 'axes_to_flip'",
                            file=sys.stderr,
                        )
                        augmentations[aug]["axis"] = augmentations[aug]["axes_to_flip"]
                    actual_function = global_augs_dict[aug](
                        axes=augmentations[aug]["axis"],
                        p=augmentations[aug]["probability"],
                    )
                elif aug in ["rotate_90", "rotate_180"]:
                    for axis in augmentations[aug]["axis"]:
                        augmentation_list.append(
                            global_augs_dict[aug](
                                axis=axis, p=augmentations[aug]["probability"]
                            )
                        )
                elif aug in ["swap", "elastic"]:
                    actual_function = global_augs_dict[aug](
                        patch_size=patch_size, p=augmentations[aug]["probability"]
                    )
                elif aug == "blur":
                    actual_function = global_augs_dict[aug](
                        std=augmentations[aug]["std"],
                        p=augmentations[aug]["probability"],
                    )
                elif aug == "noise":
                    actual_function = global_augs_dict[aug](
                        mean=augmentations[aug]["mean"],
                        std=augmentations[aug]["std"],
                        p=augmentations[aug]["probability"],
                    )
                elif aug == "anisotropic":
                    actual_function = global_augs_dict[aug](
                        axes=augmentations[aug]["axis"],
                        downsampling=augmentations[aug]["downsampling"],
                        p=augmentations[aug]["probability"],
                    )
                else:
                    actual_function = global_augs_dict[aug](
                        p=augmentations[aug]["probability"]
                    )
                if actual_function is not None:
                    augmentation_list.append(actual_function)

    if augmentation_list:
        transform = Compose(augmentation_list)
    else:
        transform = None
    subjects_dataset = torchio.SubjectsDataset(subjects_list, transform=transform)
    if not train:
        return subjects_dataset
    if sampler in ("weighted", "weightedsampler", "weightedsample"):
        sampler = global_sampler_dict[sampler](patch_size, probability_map="label")
    else:
        sampler = global_sampler_dict[sampler](patch_size)
    # all of these need to be read from model.yaml
    patches_queue = torchio.Queue(
        subjects_dataset,
        max_length=q_max_length,
        samples_per_volume=q_samples_per_volume,
        sampler=sampler,
        num_workers=q_num_workers,
        shuffle_subjects=True,
        shuffle_patches=True,
        verbose=q_verbose,
    )
    return patches_queue

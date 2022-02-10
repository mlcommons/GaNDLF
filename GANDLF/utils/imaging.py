import sys, math, os, pathlib
import SimpleITK as sitk
import numpy as np
import torchio

from .generic import get_filename_extension_sanitized


def perform_sanity_check_on_subject(subject, parameters):
    """
    This function performs sanity check on the subject to ensure presence of consistent header information WITHOUT loading images into memory.

    Args:
        subject (torchio.Subject): The input subject.
        parameters (dict): The parameters passed by the user yaml.

    Returns:
        bool: True if everything is okay.

    Raises:
        ValueError: Dimension mismatch in the images.
        ValueError: Origin mismatch in the images.
        ValueError: Orientation mismatch in the images.
    """
    # read the first image and save that for comparison
    file_reader_base = None

    import copy

    list_for_comparison = copy.deepcopy(parameters["headers"]["channelHeaders"])
    if parameters["headers"]["labelHeader"] is not None:
        list_for_comparison.append("label")

    if len(list_for_comparison) > 1:
        for key in list_for_comparison:
            if file_reader_base is None:
                if subject[str(key)]["path"] != "":
                    file_reader_base = sitk.ImageFileReader()
                    file_reader_base.SetFileName(subject[str(key)]["path"])
                    file_reader_base.ReadImageInformation()
                else:
                    # this case is required if any tensor/imaging operation has been applied in dataloader
                    file_reader_base = subject[str(key)].as_sitk()
            else:
                if subject[str(key)]["path"] != "":
                    # in this case, file_reader_base is ready
                    file_reader_current = sitk.ImageFileReader()
                    file_reader_current.SetFileName(subject[str(key)]["path"])
                    file_reader_current.ReadImageInformation()
                else:
                    # this case is required if any tensor/imaging operation has been applied in dataloader
                    file_reader_current = subject[str(key)].as_sitk()

                if (
                    file_reader_base.GetDimension()
                    != file_reader_current.GetDimension()
                ):
                    raise ValueError(
                        "Dimensions for Subject '"
                        + subject["subject_id"]
                        + "' are not consistent."
                    )

                if file_reader_base.GetOrigin() != file_reader_current.GetOrigin():
                    raise ValueError(
                        "Origin for Subject '"
                        + subject["subject_id"]
                        + "' are not consistent."
                    )

                if (
                    file_reader_base.GetDirection()
                    != file_reader_current.GetDirection()
                ):
                    raise ValueError(
                        "Orientation for Subject '"
                        + subject["subject_id"]
                        + "' are not consistent."
                    )

                if file_reader_base.GetSpacing() != file_reader_current.GetSpacing():
                    raise ValueError(
                        "Spacing for Subject '"
                        + subject["subject_id"]
                        + "' are not consistent."
                    )

    return True


def write_training_patches(subject, params):
    """
    This function writes the training patches to disk.

    Args:
        subject (torchio.Subject): The input subject.
        params (dict): The parameters passed by the user yaml; needs the "output_dir" and "current_epoch" keys to be present.
    """
    # create folder tree for saving the patches
    training_output_dir = os.path.join(params["output_dir"], "training_patches")
    pathlib.Path(training_output_dir).mkdir(parents=True, exist_ok=True)
    training_output_dir_epoch = os.path.join(
        training_output_dir, str(params["current_epoch"])
    )
    pathlib.Path(training_output_dir_epoch).mkdir(parents=True, exist_ok=True)
    training_output_dir_current_subject = os.path.join(
        training_output_dir_epoch, subject["subject_id"][0]
    )
    pathlib.Path(training_output_dir_current_subject).mkdir(parents=True, exist_ok=True)

    # write the training patches to disk
    ext = get_filename_extension_sanitized(subject["path_to_metadata"][0])
    for key in params["channel_keys"]:
        img_to_write = torchio.ScalarImage(
            tensor=subject[key][torchio.DATA][0], affine=subject[key]["affine"][0]
        ).as_sitk()
        sitk.WriteImage(
            img_to_write,
            os.path.join(training_output_dir_current_subject, "modality_" + key + ext),
        )

    if params["label_keys"] is not None:
        img_to_write = torchio.ScalarImage(
            tensor=subject[params["label_keys"][0]][torchio.DATA][0],
            affine=subject[key]["affine"][0],
        ).as_sitk()
        sitk.WriteImage(
            img_to_write,
            os.path.join(training_output_dir_current_subject, "label_" + key + ext),
        )

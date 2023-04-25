import os, pathlib, sys, math
import numpy as np
import SimpleITK as sitk
import torchio

from .generic import get_filename_extension_sanitized


def resample_image(
    img, spacing, size=None, interpolator=sitk.sitkLinear, outsideValue=0
):
    """
    Resample image to certain spacing and size.
    Args:
        img (SimpleITK.Image): The input image to resample.
        spacing (list): List of length 3 indicating the voxel spacing as [x, y, z].
        size (list, optional): List of length 3 indicating the number of voxels per dim [x, y, z], which will use compute the appropriate size based on the spacing. Defaults to [].
        interpolator (SimpleITK.InterpolatorEnum, optional): The interpolation type to use. Defaults to SimpleITK.sitkLinear.
        origin (list, optional): The location in physical space representing the [0,0,0] voxel in the input image.  Defaults to [0,0,0].
        outsideValue (int, optional): value used to pad are outside image.  Defaults to 0.
    Raises:
        Exception: Spacing/resolution mismatch.
        Exception: Size mismatch.
    Returns:
        SimpleITK.Image: The resampled input image.
    """
    if len(spacing) != img.GetDimension():
        raise Exception("len(spacing) != " + str(img.GetDimension()))

    # Set Size
    if size is None:
        inSpacing = img.GetSpacing()
        inSize = img.GetSize()
        size = [
            int(math.ceil(inSize[i] * (inSpacing[i] / spacing[i])))
            for i in range(img.GetDimension())
        ]
    else:
        if len(size) != img.GetDimension():
            raise Exception("len(size) != " + str(img.GetDimension()))

    # Resample input image
    return sitk.Resample(
        img,
        size,
        sitk.Transform(),
        interpolator,
        img.GetOrigin(),
        spacing,
        img.GetDirection(),
        outsideValue,
    )


def resize_image(input_image, output_size, interpolator=sitk.sitkLinear):
    """
    This function resizes the input image based on the output size and interpolator.
    Args:
        input_image (SimpleITK.Image): The input image to be resized.
        output_size (Union[numpy.ndarray, list, tuple]): The output size to resample input_image to.
        interpolator (SimpleITK.sitkInterpolator): The desired interpolator.
    Returns:
        SimpleITK.Image: The output image after resizing.
    """
    output_size_parsed = None
    inputSize = input_image.GetSize()
    inputSpacing = np.array(input_image.GetSpacing())
    outputSpacing = np.array(inputSpacing)

    output_size_parsed = output_size
    if isinstance(output_size, dict):
        if "resize" in output_size:
            output_size_parsed = output_size["resize"]

    assert len(output_size_parsed) == len(
        inputSpacing
    ), "The output size dimension is inconsistent with the input dataset, please check parameters."

    for i, n in enumerate(output_size_parsed):
        outputSpacing[i] = outputSpacing[i] * (inputSize[i] / n)

    return resample_image(
        input_image,
        outputSpacing,
        interpolator=interpolator,
    )


def softer_sanity_check(base_property, new_property, threshold=0.00001):
    """
    This function checks if the new property is within the threshold of the base property.
    Args:
        base_property (float): The base property to check.
        new_property (float): The new property to check
        threshold (float, optional): The threshold to check if the new property is within the base property. Defaults to 0.00001.
    Returns:
        bool: Whether the new property is within the threshold of the base property.
    """
    arr_1 = np.array(base_property)
    arr_2 = np.array(new_property)
    diff = np.sum(arr_1 - arr_2)

    result = False
    if diff <= threshold:
        result = True

    return result


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

    def _get_itkimage_or_filereader(subject_str_key):
        """
        Helper function to get the itk image or file reader from the subject.

        Args:
            subject_str_key (Union[str, sitk.Image]): The subject string key.

        Returns:
            Union[sitk.ImageFileReader, sitk.Image]: The itk image or file reader.
        """
        if subject_str_key["path"] != "":
            file_reader = sitk.ImageFileReader()
            file_reader.SetFileName(subject_str_key["path"])
            file_reader.ReadImageInformation()
            return file_reader
        else:
            # this case is required if any tensor/imaging operation has been applied in dataloader
            file_reader = subject_str_key.as_sitk()

    if len(list_for_comparison) > 1:
        for key in list_for_comparison:
            if file_reader_base is None:
                file_reader_base = _get_itkimage_or_filereader(subject[str(key)])
            else:
                file_reader_current = _get_itkimage_or_filereader(subject[str(key)])

                # this check needs to be absolute
                if (
                    file_reader_base.GetDimension()
                    != file_reader_current.GetDimension()
                ):
                    raise ValueError(
                        "Dimensions for Subject '"
                        + subject["subject_id"]
                        + "' are not consistent."
                    )

                # other checks can be softer
                if not softer_sanity_check(
                    file_reader_base.GetOrigin(), file_reader_current.GetOrigin()
                ):
                    raise ValueError(
                        "Origin for Subject '"
                        + subject["subject_id"]
                        + "' are not consistent."
                    )

                if not softer_sanity_check(
                    file_reader_base.GetDirection(), file_reader_current.GetDirection()
                ):
                    raise ValueError(
                        "Orientation for Subject '"
                        + subject["subject_id"]
                        + "' are not consistent."
                    )

                if not softer_sanity_check(
                    file_reader_base.GetSpacing(), file_reader_current.GetSpacing()
                ):
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

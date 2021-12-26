import sys, math, os
import SimpleITK as sitk
import numpy as np
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
    if size == None:
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
        output_size (numpy.array | list): The output size to resample input_image to.
        interpolator (SimpleITK.sitkInterpolator): The desired interpolator.

    Returns:
        SimpleITK.Image: The output image after resizing.
    """
    output_size_parsed = None
    inputSize = input_image.GetSize()
    inputSpacing = np.array(input_image.GetSpacing())
    outputSpacing = np.array(inputSpacing)

    if isinstance(output_size, dict):
        if "resize" in output_size:
            output_size_parsed = output_size["resize"]
    elif isinstance(output_size, list) or isinstance(output_size, np.array):
        output_size_parsed = output_size

    if len(output_size_parsed) != len(inputSpacing):
        sys.exit(
            "The output size dimension is inconsistent with the input dataset, please check parameters."
        )

    for i, n in enumerate(output_size_parsed):
        outputSpacing[i] = outputSpacing[i] * (inputSize[i] / n)

    return resample_image(
        input_image,
        outputSpacing,
        interpolator=interpolator,
    )


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


def write_training_patches(subject, params, output_dir):
    """
    This function writes the training patches to disk.

    Args:
        subject (torchio.Subject): The input subject.
        params (dict): The parameters passed by the user yaml.
        output_dir (str): The output directory to write the patches to.
    """
    # write the training patches to disk
    ext = get_filename_extension_sanitized(subject["path_to_metadata"][0])
    for key in params["channel_keys"]:
        img_to_write = torchio.ScalarImage(tensor=subject[key][torchio.DATA][0], affine=subject[key]["affine"][0]).as_sitk()
        sitk.WriteImage(
            img_to_write,
            os.path.join(
                output_dir, "modality_" + key + ext
            ),
        )

    if params["label_keys"] is not None:
        img_to_write = torchio.ScalarImage(tensor=subject[params["label_keys"][0]][torchio.DATA][0], affine=subject[key]["affine"][0]).as_sitk()
        sitk.WriteImage(
            img_to_write,
            os.path.join(
                output_dir, "label_" + key + ext
            ),
        )

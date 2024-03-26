from typing import List, Optional, Tuple, Union
import os, pathlib, math, copy
from enum import Enum
import numpy as np
import SimpleITK as sitk
import torchio
import cv2

from .generic import get_filename_extension_sanitized


def resample_image(
    input_image: sitk.Image,
    spacing: Union[np.ndarray, List[float], Tuple[float]],
    size: Optional[Union[np.ndarray, List[float], Tuple[float]]] = None,
    interpolator: Optional[Enum] = sitk.sitkLinear,
    outsideValue: Optional[int] = 0,
) -> sitk.Image:
    """
    This function resamples the input image based on the spacing and size.

    Args:
        input_image (sitk.Image): The input image to be resampled.
        spacing (Union[np.ndarray, List[float], Tuple[float]]): The desired spacing for the resampled image.
        size (Optional[Union[np.ndarray, List[float], Tuple[float]]], optional): The desired size for the resampled image. Defaults to None.
        interpolator (Optional[Enum], optional): The desired interpolator. Defaults to sitk.sitkLinear.
        outsideValue (Optional[int], optional): The value to be used for the outside of the image. Defaults to 0.

    Returns:
        sitk.Image: The resampled image.
    """
    assert (
        len(spacing) == input_image.GetDimension()
    ), "The spacing dimension is inconsistent with the input dataset, please check parameters."

    # Set Size
    if size is None:
        inSpacing = input_image.GetSpacing()
        inSize = input_image.GetSize()
        size = [
            int(math.ceil(inSize[i] * (inSpacing[i] / spacing[i])))
            for i in range(input_image.GetDimension())
        ]

    assert (
        len(size) == input_image.GetDimension()
    ), "The size dimension is inconsistent with the input dataset, please check parameters."

    # Resample input image
    return sitk.Resample(
        input_image,
        size,
        sitk.Transform(),
        interpolator,
        input_image.GetOrigin(),
        spacing,
        input_image.GetDirection(),
        outsideValue,
    )


def resize_image(
    input_image: sitk.Image,
    output_size: Union[np.ndarray, list, tuple],
    interpolator: Optional[Enum] = sitk.sitkLinear,
) -> sitk.Image:
    """
    This function resizes the input image based on the output size.

    Args:
        input_image (sitk.Image): The input image to be resized.
        output_size (Union[np.ndarray, list, tuple]): The desired output size for the resized image.
        interpolator (Optional[Enum], optional): The desired interpolator. Defaults to sitk.sitkLinear.

    Returns:
        sitk.Image: The resized image.
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

    return resample_image(input_image, outputSpacing, interpolator=interpolator)


def softer_sanity_check(
    base_property: Union[np.ndarray, List[float], Tuple[float]],
    new_property: Union[np.ndarray, List[float], Tuple[float]],
    threshold: Optional[float] = 0.00001,
) -> bool:
    """
    This function performs a softer sanity check on the input properties.

    Args:
        base_property (Union[np.ndarray, List[float], Tuple[float]]): The base property.
        new_property (Union[np.ndarray, List[float], Tuple[float]]): The new property.
        threshold (Optional[float], optional): The threshold for comparison. Defaults to 0.00001.

    Returns:
        bool: True if the properties are consistent within the threshold.
    """
    arr_1 = np.array(base_property)
    arr_2 = np.array(new_property)
    diff = np.sum(arr_1 - arr_2)

    result = False
    if diff <= threshold:
        result = True

    return result


def perform_sanity_check_on_subject(subject: torchio.Subject, parameters: dict) -> bool:
    """
    This function performs a sanity check on the image modalities in input subject to ensure that they are consistent.

    Args:
        subject (torchio.Subject): The input subject.
        parameters (dict): The parameters passed by the user yaml.

    Returns:
        bool: True if the sanity check passes.
    """
    # read the first image and save that for comparison
    file_reader_base = None

    list_for_comparison = copy.deepcopy(parameters["headers"]["channelHeaders"])
    if parameters["headers"]["labelHeader"] is not None:
        list_for_comparison.append("label")

    def _get_itkimage_or_filereader(
        subject_str_key: Union[str, sitk.Image]
    ) -> Union[sitk.ImageFileReader, sitk.Image]:
        """
        Helper function to get the itk image or file reader from the subject.

        Args:
            subject_str_key (Union[str, sitk.Image]): The subject string key or itk image.

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
                assert (
                    file_reader_base.GetDimension()
                    == file_reader_current.GetDimension()
                ), (
                    "Dimensions for Subject '"
                    + subject["subject_id"]
                    + "' are not consistent."
                )

                # other checks can be softer
                assert softer_sanity_check(
                    file_reader_base.GetOrigin(), file_reader_current.GetOrigin()
                ), (
                    "Origin for Subject '"
                    + subject["subject_id"]
                    + "' are not consistent."
                )

                assert softer_sanity_check(
                    file_reader_base.GetDirection(), file_reader_current.GetDirection()
                ), (
                    "Orientation for Subject '"
                    + subject["subject_id"]
                    + "' are not consistent."
                )

                assert softer_sanity_check(
                    file_reader_base.GetSpacing(), file_reader_current.GetSpacing()
                ), (
                    "Spacing for Subject '"
                    + subject["subject_id"]
                    + "' are not consistent."
                )

    return True


def write_training_patches(subject: torchio.Subject, params: dict) -> None:
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


def get_correct_padding_size(
    patch_size: Union[List[int], Tuple[int]], model_dimension: int
):
    """
    This function returns the correct padding size based on the patch size and overlap.

    Args:
        patch_size (Union[List[int], Tuple[int]]): The patch size.
        model_dimension (int): The model dimension.

    Returns:
        Union[list, tuple]: The correct padding size.
    """
    psize_pad = list(np.asarray(np.ceil(np.divide(patch_size, 2)), dtype=int))
    # ensure that the patch size for z-axis is not 1 for 2d images
    if model_dimension == 2:
        psize_pad[-1] = 0 if psize_pad[-1] == 1 else psize_pad[-1]

    return psize_pad


def applyCustomColorMap(im_gray: np.ndarray) -> np.ndarray:
    """
    Internal function to apply a custom color map to the input image.

    Args:
        im_gray (np.ndarray): The input image.

    Returns:
        np.ndarray: The image with the custom color map applied.
    """
    img_bgr = cv2.cvtColor(im_gray.astype(np.uint8), cv2.COLOR_BGR2RGB)
    lut = np.zeros((256, 1, 3), dtype=np.uint8)
    lut[:, 0, 0] = np.zeros((256)).tolist()
    lut[:, 0, 1] = np.zeros((256)).tolist()
    lut[:, 0, 2] = np.arange(0, 256, 1).tolist()
    return cv2.LUT(img_bgr, lut)

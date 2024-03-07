from typing import List, Tuple
import numpy as np
import torch
import nibabel as nib

from torchio.transforms import SpatialTransform


# adapted from https://codereview.stackexchange.com/questions/132914/crop-black-border-of-image-using-numpy/132933#132933
def crop_image_outside_zeros(
    array: np.ndarray, patch_size: List[int]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    This function rotates an image by 90 degrees around the specified axis.

    Args:
        array (numpy.array): The input array.
        patch_size (List[int]): The patch size.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The new corner indices and the new array.
    """
    dimensions = len(array.shape)
    assert dimensions == 4, "Array expected to be 4D but got {} dimensions.".format(
        dimensions
    )

    # collapse to single channel and get the mask of non-zero voxels
    mask = array.sum(axis=0) > 0

    # get the small and large corners

    m0 = mask.any(1).any(1)
    m1 = mask.any(0)
    m2 = m1.any(0)
    m1 = m1.any(1)

    small = [m0.argmax(), m1.argmax(), m2.argmax()]
    large = [m0[::-1].argmax(), m1[::-1].argmax(), m2[::-1].argmax()]
    large = [m - l for m, l in zip(mask.shape, large)]

    # ensure we have a full patch
    # for each axis
    for i in range(3):
        # if less than patch size, extend the small corner out
        if large[i] - small[i] < patch_size[i]:
            small[i] = large[i] - patch_size[i]

        # if bottom fell off array, extend the large corner and set small to 0
        if small[i] < 0:
            small[i] = 0
            large[i] = patch_size[i]

    # calculate pixel location of new bounding box corner (will use to update the reference of the image to physical space)
    new_corner_idxs = np.array([small[0], small[1], small[2]])
    # Get the contents of the bounding box from the array
    new_array = array[:, small[0] : large[0], small[1] : large[1], small[2] : large[2]]

    return new_corner_idxs, new_array


# adapted from https://github.com/fepegar/torchio/blob/master/torchio/transforms/preprocessing/spatial/crop.py
class CropExternalZeroplanes(SpatialTransform):
    """
    Transformation class to enable taking the whole image stack (including segmentation) and removing
    (starting from edges) physical-coordinate planes with all zero voxels until you reach a non-zero voxel.
    Args:
        patch_size: patch size (used to ensure we do not crop to smaller size than this)
        **kwargs: See :class:`~torchio.transforms.Transform` for additional
            keyword arguments.
    """

    def __init__(self, patch_size, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.args_names = ("patch_size",)

    def apply_transform(self, subject):
        # get dictionary of images
        images_dict = subject.get_images_dict(intensity_only=False)

        # make sure shapes are consistent across images, and get this shape
        subject.check_consistent_spatial_shape()
        example_image_affine = list(images_dict.values())[0].affine

        # create stack of all images (including segmentation)
        numpy_stack_list = []
        names_list = []
        for name, image in images_dict.items():
            numpy_stack_list.append(image.data.numpy().copy())
            names_list.append(name)
        numpy_stack = np.concatenate(numpy_stack_list, axis=0)

        # crop away the external zero-planes on the whole stack
        new_corner_idxs, new_stack = crop_image_outside_zeros(
            array=numpy_stack, patch_size=self.patch_size
        )

        # recompute origin of affine matrix using initial image shape
        new_origin = nib.affines.apply_affine(example_image_affine, new_corner_idxs)
        new_affine = example_image_affine.copy()
        new_affine[:3, 3] = new_origin

        # repopulate the subject data and shape
        for idx, array in enumerate(new_stack):
            images_dict[names_list[idx]]["data"] = torch.tensor(
                np.expand_dims(array, axis=0)
            )
            images_dict[names_list[idx]]["affine"] = new_affine

        return subject

    def is_invertible(self):
        return False

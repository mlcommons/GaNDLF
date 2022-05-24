#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 20:03:35 2019

@author: siddhesh
"""

import os
import numpy as np
from torch.utils.data.dataset import Dataset

import tiffslide as openslide
from skimage.transform import resize
from skimage.filters import threshold_otsu, median
from skimage.morphology import binary_closing, disk
from scipy.ndimage import binary_fill_holes
from GANDLF.OPM.opm.utils import tissue_mask


def get_tissue_mask(image):
    """
    This function is used to generate tissue masks
    works for patches too i guess

    Args:
        img_rgb (numpy.array): Input image.
        rgb_min (int, optional): The minimum threshold. Defaults to 50.

    Returns:
        numpy.array: The tissue mask.
    """
    resized_image = resize(image, (512, 512), anti_aliasing=True)
    mask = tissue_mask(resized_image)

    # upsample the mask to original size with nearest neighbor interpolation
    mask = resize(mask, (image.shape[0], image.shape[1]), order=0, mode="constant")

    return mask


class InferTumorSegDataset(Dataset):
    def __init__(
        self,
        wsi_path,
        patch_size,
        stride_size,
        selected_level,
        mask_level,
        transform=None,
    ):
        self.transform = transform
        self._wsi_path = wsi_path
        self._patch_size = patch_size
        if self._patch_size[-1] == 1:
            self._patch_size = self._patch_size[:-1]
        self._stride_size = stride_size
        self._selected_level = selected_level
        self._mask_level = mask_level
        self._os_image = openslide.open_slide(os.path.join(self._wsi_path))
        self._points = []
        self._basic_preprocessing()

    def _basic_preprocessing(self):
        mask_xdim, mask_ydim = self._os_image.level_dimensions[self._mask_level]
        extracted_image = self._os_image.read_region(
            (0, 0),
            self._mask_level,
            (mask_xdim, mask_ydim),
            as_array=True,
        )
        mask = get_tissue_mask(extracted_image)
        del extracted_image

        # For some reason, tiffslide x, y coordinates are flipped
        # Fix is definitely needed
        width, height = self._os_image.level_dimensions[str(self._selected_level)]
        if not (self._selected_level == self._mask_level):
            mask = resize(mask, (height, width))
        mask = (mask > 0).astype(np.uint8)

        # This is bugged because currently if mask_level is not equal to selected_level,
        # then this logic straight up does not work
        # You would have to scale the patch size appropriately for this to work correctly
        # Remove all the points which are closer to the boundary of the wsi
        # by accsesing the WSI level properties with
        # self._os_image.level_dimensions[self._selected_level]
        # Logic as if point + self.patch_size > wsi_dimensions
        # The move the point by the wsi_dimensions - (patch_size + self.points)
        # This is because the patch is not going to be extracted if it is
        # outside the wsi
        for i in range(0, width - self._patch_size[0], self._stride_size[0]):
            for j in range(0, height - self._patch_size[1], self._stride_size[1]):
                # If point goes beyond the wsi in y_dim, then move so that we can extract the patch
                if i + self._patch_size[0] > width:
                    i = width - self._patch_size[0]
                # If point goes beyond the wsi in x_dim, then move so that we can extract the patch
                if j + self._patch_size[1] > height:
                    j = height - self._patch_size[1]
                # If there is anything in the mask patch, only then consider it
                if np.any(
                    mask[i : i + self._patch_size[0], j : j + self._patch_size[1]]
                ):
                    self._points.append([i, j])

        self._points = np.array(self._points)
        self._points[:, [0, 1]] = self._points[:, [1, 0]]
        self._mask = mask

    def __len__(self):
        return len(self._points)

    def __getitem__(self, idx):
        """
        This function is used to return the patch and its location.

        Args:
            idx (int): The index of the patch.

        Returns:
            (string, int, int): The patch, x and y locations.
        """
        x_loc, y_loc = self._points[idx]
        patch = self._os_image.read_region(
            (x_loc, y_loc),
            self._selected_level,
            (self._patch_size[0], self._patch_size[1]),
            as_array=True,
        )

        # this is to ensure that channels come at the beginning
        patch = patch.transpose([2, 0, 1])
        # this is to ensure that we always have a z-stack before applying any torchio transforms
        patch = np.expand_dims(patch, axis=-1)
        if self.transform is not None:
            patch = self.transform(patch)

        # remove z-stack
        patch = patch.squeeze(-1)

        return patch, (x_loc, y_loc)

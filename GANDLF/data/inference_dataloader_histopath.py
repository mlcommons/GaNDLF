#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 20:03:35 2019

@author: siddhesh
"""

import os

import numpy as np
import tiffslide
from GANDLF.data.patch_miner.opm.utils import get_patch_size_in_microns, tissue_mask
from skimage.transform import resize
from torch.utils.data.dataset import Dataset


def get_tissue_mask(image):
    """
    This function is used to generate tissue masks; works for patches as well

    Args:
        img_rgb (numpy.array): Input image.
        rgb_min (int, optional): The minimum threshold. Defaults to 50.

    Returns:
        numpy.array: The tissue mask.
    """
    try:
        resized_image = resize(image, (512, 512), anti_aliasing=True)
        mask = tissue_mask(resized_image)
        # upsample the mask to original size with nearest neighbor interpolation
        mask = resize(mask, (image.shape[0], image.shape[1]), order=0, mode="constant")
    except Exception as e:
        print("Entering fallback in histology inference loader because of: ", e)
        mask = np.ones(image.shape, dtype=np.ubyte)

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
        self._patch_size = get_patch_size_in_microns(wsi_path, self._patch_size)
        self._stride_size = stride_size
        if self._stride_size is None:
            self._stride_size = (
                (np.array(self._patch_size) / 2).astype(np.uint16).tolist()
            )
        self._stride_size = get_patch_size_in_microns(wsi_path, self._stride_size)
        self._selected_level = selected_level
        self._mask_level = mask_level
        self._os_image = tiffslide.open_slide(os.path.join(self._wsi_path))
        self._points = []
        self._basic_preprocessing()

    def _basic_preprocessing(self):
        mask = None
        height, width = self._os_image.level_dimensions[self._selected_level]
        try:
            mask_xdim, mask_ydim = self._os_image.level_dimensions[self._mask_level]
            mask = get_tissue_mask(
                self._os_image.read_region(
                    (0, 0),
                    self._mask_level,
                    (mask_xdim, mask_ydim),
                    as_array=True,
                )
            )

            if self._selected_level != self._mask_level:
                mask = resize(mask, (height, width))
            mask = (mask > 0).astype(np.ubyte)
        except Exception as e:
            print("Mask could not be initialized, using entire image:", e)
        # This is buggy because currently if mask_level is not equal to selected_level,
        # then this logic straight up does not work
        # You would have to scale the patch size appropriately for this to work correctly
        # Remove all the points which are closer to the boundary of the wsi
        # by accessing the WSI level properties with
        # self._os_image.level_dimensions[self._selected_level]
        # Logic as if point + self.patch_size > wsi_dimensions
        # The move the point by the wsi_dimensions - (patch_size + self.points)
        # This is because the patch is not going to be extracted if it is
        # outside the wsi
        for i in range(
            0,
            width - (self._patch_size[0] + self._stride_size[0]),
            self._stride_size[0],
        ):
            for j in range(
                0,
                height - (self._patch_size[1] + self._stride_size[1]),
                self._stride_size[1],
            ):
                # If point goes beyond the wsi in y_dim, then move so that we can extract the patch
                coord_width, coord_height = i, j
                if i + self._patch_size[0] > width:
                    coord_width = width - self._patch_size[0]
                # If point goes beyond the wsi in x_dim, then move so that we can extract the patch
                if j + self._patch_size[1] > height:
                    coord_height = height - self._patch_size[1]
                # If there is anything in the mask patch, only then consider it
                if mask is not None:
                    if np.any(
                        mask[
                            coord_width : coord_width + self._patch_size[0],
                            coord_height : coord_height + self._patch_size[1],
                        ]
                    ):
                        self._points.append([coord_width, coord_height])
                else:
                    self._points.append([coord_width, coord_height])

        self._points = np.array(self._points)
        self._points[:, [0, 1]] = self._points[:, [1, 0]]

    def get_patch_size(self):
        return self._patch_size

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

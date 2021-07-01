#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 20:03:35 2019

@author: siddhesh
"""

import os
import numpy as np
from torch.utils.data.dataset import Dataset

if os.name != "nt":
    """path inference is Linux-only because openslide for Windows works only for Python-3.8  whereas pickle5 works only for 3.6 and 3.7"""
    from openslide import OpenSlide
    from skimage.transform import resize
    from skimage.filters import threshold_otsu, median
    from skimage.morphology import binary_closing, disk, binary_erosion
    from scipy.ndimage import binary_fill_holes

    def tissue_mask_generation(img_rgb, rgb_min=50):
        """
        This function is used to generate tissue masks
        works for patches too i guess

        Args:
            img_rgb (numpy.array): Input image.
            rgb_min (int, optional): The minimum threshold. Defaults to 50.

        Returns:
            numpy.array: The tissue mask.
        """
        img_rgb = np.array(img_rgb)
        background_r = img_rgb[:, :, 0] > threshold_otsu(img_rgb[:, :, 0])
        background_g = img_rgb[:, :, 1] > threshold_otsu(img_rgb[:, :, 1])
        background_b = img_rgb[:, :, 2] > threshold_otsu(img_rgb[:, :, 2])
        tissue_rgb = np.logical_not(background_r & background_g & background_b)
        del background_b, background_g, background_r

        min_r = img_rgb[:, :, 0] > rgb_min
        min_g = img_rgb[:, :, 1] > rgb_min
        min_b = img_rgb[:, :, 2] > rgb_min
        tissue_mask = tissue_rgb & min_r & min_g & min_b
        del min_r, min_g, min_b

        close_kernel = np.ones((7, 7), dtype=np.uint8)
        image_close = binary_closing(np.array(tissue_mask), close_kernel)
        tissue_mask = binary_fill_holes(image_close)
        del image_close, close_kernel

        # Apply median filter
        tissue_mask = median(tissue_mask, disk(7))
        tissue_mask = np.array(tissue_mask).astype(np.uint8)
        tissue_mask = tissue_mask > 0

        return tissue_mask

    class InferTumorSegDataset(Dataset):
        def __init__(
            self, wsi_path, patch_size, stride_size, selected_level, mask_level
        ):
            self._wsi_path = wsi_path
            self._patch_size = patch_size
            self._stride_size = stride_size
            self._selected_level = selected_level
            self._mask_level = mask_level
            self._os_image = OpenSlide(os.path.join(self._wsi_path))
            self._points = []
            self._basic_preprocessing()

        def _basic_preprocessing(self):
            mask_xdim, mask_ydim = self._os_image.level_dimensions[self._mask_level]
            print(self._wsi_path, self._os_image)
            extracted_image = self._os_image.read_region(
                (0, 0), self._mask_level, (mask_xdim, mask_ydim)
            ).convert("RGB")
            mask = tissue_mask_generation(extracted_image)
            del extracted_image
            ydim, xdim = self._os_image.level_dimensions[self._selected_level]
            mask = resize(mask, (xdim, ydim))
            mask = (mask > 0).astype(np.uint8)
            for i in range(0, ydim - self._patch_size[0], self._stride_size[0]):
                for j in range(0, xdim - self._patch_size[1], self._stride_size[1]):
                    self._points.append([j, i])
            for i in range(len(self._points)):
                point = self._points[i]
                if not np.any(
                    mask[
                        point[0] : point[0] + self._patch_size[0],
                        point[1] : point[1] + self._patch_size[1],
                    ]
                ):
                    self._points[i] = [-1, -1]
            self._points = np.array(self._points)
            self._points = np.delete(
                self._points, np.argwhere(self._points == np.array([-1, -1])), 0
            )
            self._points[:, [0, 1]] = self._points[:, [1, 0]]
            self._mask = mask

        def __len__(self):
            return len(self._points)

        def __getitem__(self, idx):
            x_loc, y_loc = self._points[idx]
            patch = np.array(
                self._os_image.read_region(
                    (x_loc * 4, y_loc * 4),
                    self._selected_level,
                    (self._patch_size[0], self._patch_size[1]),
                ).convert("RGB")
            )
            patch = np.array(patch / 255)
            patch = patch.transpose([2, 0, 1])
            return patch, (x_loc, y_loc)

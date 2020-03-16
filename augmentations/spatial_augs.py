#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 20:20:11 2019

@author: siddhesh
"""

import numpy as np
import scipy
from augmentations.utils import crop_again

def augment_rot90(sample_data, sample_seg, num_rot=(1, 2, 3), axes=(0, 1, 2)):
    """
    :param sample_data:
    :param sample_seg:
    :param num_rot: rotate by 90 degrees how often? must be tuple -> nom rot randomly chosen from that tuple
    :param axes: around which axes will the rotation take place? two axes are chosen randomly from axes.
    :return:
    """
    num_rot = np.random.choice(num_rot)
    axes = np.random.choice(axes, size=2, replace=False)
    axes.sort()
    axes = [i + 1 for i in axes]
    sample_data = np.rot90(sample_data, num_rot, axes)
    if sample_seg is not None:
        sample_seg = np.rot90(sample_seg, num_rot, axes)
    return sample_data, sample_seg


def augment_mirroring(sample_data, sample_seg=None, axes=(0, 1, 2)):
    if (len(sample_data.shape) != 3) and (len(sample_data.shape) != 4):
        raise Exception(
            "Invalid dimension for sample_data and sample_seg. sample_data and sample_seg should be either "
            "[channels, x, y] or [channels, x, y, z]")
    if 0 in axes and np.random.uniform() < 0.5:
        sample_data[:, :] = sample_data[:, ::-1]
        if sample_seg is not None:
            sample_seg[:, :] = sample_seg[:, ::-1]
    if 1 in axes and np.random.uniform() < 0.5:
        sample_data[:, :, :] = sample_data[:, :, ::-1]
        if sample_seg is not None:
            sample_seg[:, :, :] = sample_seg[:, :, ::-1]
    if 2 in axes and len(sample_data.shape) == 4:
        if np.random.uniform() < 0.5:
            sample_data[:, :, :, :] = sample_data[:, :, :, ::-1]
            if sample_seg is not None:
                sample_seg[:, :, :, :] = sample_seg[:, :, :, ::-1]
    return sample_data, sample_seg

def augment_rotate_angle(sample_data, sample_seg, rotation_angle,
                         axes = (0, 1, 2)):
    """
    :param sample_data:
    :param sample_seg:
    :param num_rot: rotate by 90 degrees how often? must be tuple -> nom rot randomly chosen from that tuple
    :param axes: around which axes will the rotation take place? two axes are chosen randomly from axes.
    :return:
    """
    axes = np.random.choice(axes, size=2, replace=False)
    axes.sort()
    axes = [i + 1 for i in axes]
    t = sample_data.shape
    selected_rotation_angle = np.random.randint(rotation_angle)
    sample_data = scipy.ndimage.rotate(sample_data, selected_rotation_angle, axes)
    if sample_seg is not None:
        sample_seg = scipy.ndimage.rotate(sample_seg, selected_rotation_angle, axes)
    sample_data = crop_again(sample_data, t)
    sample_seg = crop_again(sample_seg, t)
    return sample_data, sample_seg
#
#def augment_zoom_noise(sample_data, zoom_factor):
#    zoomed_image = scipy.ndimage.zoom(sample_data, zoom_factor)
#    print("Zoom done")
#    unzoomed_image = scipy.ndimage.zoom(zoomed_image, (1/zoom_factor))
#    return unzoomed_image

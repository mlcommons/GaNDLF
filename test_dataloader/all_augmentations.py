from batchgenerators.augmentations.utils import create_zero_centered_coordinate_mesh, elastic_deform_coordinates, \
    interpolate_img, \
    rotate_coords_2d, rotate_coords_3d, scale_coords, resize_segmentation, resize_multichannel_image, \
    elastic_deform_coordinates_2
from batchgenerators.augmentations.crop_and_pad_augmentations import random_crop as random_crop_aug
from batchgenerators.augmentations.crop_and_pad_augmentations import center_crop as center_crop_aug
import numpy as np
from builtins import range
from torch.autograd import Variable
import torch
def augment_gamma(image):
    gamma = np.random.uniform(1,2)
    min_image = image.min()
    range_image = image.max() - min_image
    image = np.power(((image - min_image)/float(range_image + 1e-7)) , gamma)*range_image + min_image
    return image

def normalize(image_array):
    temp = image_array > 0
    temp_image_array = image_array[temp]    
    mu = np.mean(temp_image_array)
    sig = np.std(temp_image_array)
    image_array[temp] = (image_array[temp] - mu)/sig
    return image_array

def gaussian(img, is_training, mean, stddev):
    l,b,h =  img.shape
    noise = np.random.normal(mean, stddev, (l,b,h))
    noise = noise.reshape(l,b,h)
    img = img + noise 
    return img  






def augment_spatial_2(data, seg, patch_size, patch_center_dist_from_border= (64,64,64),
                    do_elastic_deform=True, deformation_scale=(0, 0.25),
                    do_rotation=True, angle_x=(0, 2 * np.pi), angle_y=(0, 2 * np.pi), angle_z=(0, 2 * np.pi),
                    do_scale=True, scale=(0.75, 1.25), border_mode_data='nearest', border_cval_data=0, order_data=3,
                    border_mode_seg='constant', border_cval_seg=0, order_seg=0, random_crop=True, p_el_per_sample=0.5,
                    p_scale_per_sample=0.5, p_rot_per_sample=0.5):
    """
    :param data:
    :param seg:
    :param patch_size:
    :param patch_center_dist_from_border:
    :param do_elastic_deform:
    :param magnitude: this determines how large the magnitude of the deformation is relative to the patch_size.
    0.125 = 12.5%% of the patch size (in each dimension).
    :param sigma: this determines the scale of the deformation. small values = local deformations,
    large values = large deformations.
    :param do_rotation:
    :param angle_x:
    :param angle_y:
    :param angle_z:
    :param do_scale:
    :param scale:
    :param border_mode_data:
    :param border_cval_data:
    :param order_data:
    :param border_mode_seg:
    :param border_cval_seg:
    :param order_seg:
    :param random_crop:
    :param p_el_per_sample:
    :param p_scale_per_sample:
    :param p_rot_per_sample:
    :param clip_to_safe_magnitude:
    :return:
    """
    dim = len(patch_size)
    seg_result = None
    if seg is not None:
        if dim == 2:
            seg_result = np.zeros((seg.shape[0], seg.shape[1], patch_size[0], patch_size[1]), dtype=np.float32)
        else:
            seg_result = np.zeros((seg.shape[0], seg.shape[1], patch_size[0], patch_size[1], patch_size[2]),
                                  dtype=np.float32)
           

    if dim == 2:
        data_result = np.zeros((data.shape[0], data.shape[1], patch_size[0], patch_size[1]), dtype=np.float32)
    else:
        data_result = np.zeros((data.shape[0], data.shape[1], patch_size[0], patch_size[1], patch_size[2]),
                               dtype=np.float32)

    if not isinstance(patch_center_dist_from_border, (list, tuple, np.ndarray)):
        patch_center_dist_from_border = dim * [patch_center_dist_from_border]

    for sample_id in range(data.shape[0]):
        coords = create_zero_centered_coordinate_mesh(patch_size)
        modified_coords = False

        if np.random.uniform() < p_el_per_sample and do_elastic_deform:
            mag = []
            sigmas = []

            # one scale per case, scale is in percent of patch_size
            def_scale = np.random.uniform(deformation_scale[0], deformation_scale[1])

            for d in range(len(data[sample_id].shape) - 1):
                # transform relative def_scale in pixels
                sigmas.append(def_scale * patch_size[d])

                # define max magnitude
                max_magnitude = sigmas[-1] * (3 / 4.)

                # the magnitude needs to depend on the scale, otherwise not much is going to happen most of the time.
                # we want the magnitude to be high, but not higher than max_magnitude (otherwise the deformations
                # become very ugly). Let's sample mag_real with a gaussian
                mag_real = np.random.normal(max_magnitude * (2 / 3), scale=max_magnitude / 3)

                # clip to make sure we stay reasonable
                mag_real = np.clip(mag_real, 0, max_magnitude)

                mag.append(mag_real)
            #print(np.round(sigmas, decimals=3), np.round(mag, decimals=3))
            coords = elastic_deform_coordinates_2(coords, sigmas, mag)
            modified_coords = True

        if np.random.uniform() < p_rot_per_sample and do_rotation:
            if angle_x[0] == angle_x[1]:
                a_x = angle_x[0]
            else:
                a_x = np.random.uniform(angle_x[0], angle_x[1])
            if dim == 3:
                if angle_y[0] == angle_y[1]:
                    a_y = angle_y[0]
                else:
                    a_y = np.random.uniform(angle_y[0], angle_y[1])
                if angle_z[0] == angle_z[1]:
                    a_z = angle_z[0]
                else:
                    a_z = np.random.uniform(angle_z[0], angle_z[1])
                coords = rotate_coords_3d(coords, a_x, a_y, a_z)
            else:
                coords = rotate_coords_2d(coords, a_x)
            modified_coords = True

        if np.random.uniform() < p_scale_per_sample and do_scale:
            if np.random.random() < 0.5 and scale[0] < 1:
                sc = np.random.uniform(scale[0], 1)
            else:
                sc = np.random.uniform(max(scale[0], 1), scale[1])
            coords = scale_coords(coords, sc)
            modified_coords = True

        # now find a nice center location
        if modified_coords:
            # recenter coordinates
            coords_mean = coords.mean(axis=tuple(range(1, len(coords.shape))), keepdims=True)
            coords -= coords_mean

            for d in range(dim):
                if random_crop:
                    ctr = np.random.uniform(patch_center_dist_from_border[d],
                                            data.shape[d + 2] - patch_center_dist_from_border[d])
                else:
                    ctr = int(np.round(data.shape[d + 2] / 2.))
                coords[d] += ctr
            for channel_id in range(data.shape[1]):
                data_result[sample_id, channel_id] = interpolate_img(data[sample_id, channel_id], coords, order_data,
                                                                     border_mode_data, cval=border_cval_data)
            if seg is not None:
                for channel_id in range(seg.shape[1]):
                    seg_result[sample_id, channel_id] = interpolate_img(seg[sample_id, channel_id], coords, order_seg,
                                                                        border_mode_seg, cval=border_cval_seg, is_seg=True)
        else:
            if seg is None:
                s = None
            else:
                s = seg[sample_id:sample_id + 1]
            if random_crop:
                margin = [patch_center_dist_from_border[d] - patch_size[d] // 2 for d in range(dim)]
                d, s = random_crop_aug(data[sample_id:sample_id + 1], s, patch_size, margin)
            else:
                d, s = center_crop_aug(data[sample_id:sample_id + 1], patch_size, s)
            data_result[sample_id] = d[0]
            if seg is not None:
                seg_result[sample_id] = s[0]
    return data_result, seg_result



def elastic_deformation(data, seg, patch_size, patch_center_dist_from_border, do_elastic_deform=True, deformation_scale=(0, 0.25), random_crop=True, p_el_per_sample=1):
 
    dim = len(patch_size)
    seg_result = None
    seg_result = np.zeros((seg.shape[0], seg.shape[1], patch_size[0], patch_size[1], patch_size[2]),
                                  dtype=np.float32)

    data_result = np.zeros((data.shape[0], data.shape[1], patch_size[0], patch_size[1], patch_size[2]),
                               dtype=np.float32)

    if not isinstance(patch_center_dist_from_border, (list, tuple, np.ndarray)):
        patch_center_dist_from_border = dim * [patch_center_dist_from_border]

    for sample_id in range(data.shape[0]):
        coords = create_zero_centered_coordinate_mesh(patch_size)
        modified_coords = False

        if np.random.uniform() < p_el_per_sample and do_elastic_deform:
            mag = []
            sigmas = []

            # one scale per case, scale is in percent of patch_size
            def_scale = np.random.uniform(deformation_scale[0], deformation_scale[1])

            for d in range(len(data[sample_id].shape) - 1):
                # transform relative def_scale in pixels
                sigmas.append(def_scale * patch_size[d])

                # define max magnitude
                max_magnitude = sigmas[-1] * (3 / 4.)

                # the magnitude needs to depend on the scale, otherwise not much is going to happen most of the time.
                # we want the magnitude to be high, but not higher than max_magnitude (otherwise the deformations
                # become very ugly). Let's sample mag_real with a gaussian
                mag_real = np.random.normal(max_magnitude * (2 / 3), scale=max_magnitude / 3)

                # clip to make sure we stay reasonable
                mag_real = np.clip(mag_real, 0, max_magnitude)

                mag.append(mag_real)
            #print(np.round(sigmas, decimals=3), np.round(mag, decimals=3))
            coords = elastic_deform_coordinates_2(coords, sigmas, mag)
            modified_coords = True

        if np.random.uniform() < p_rot_per_sample and do_rotation:
            if angle_x[0] == angle_x[1]:
                a_x = angle_x[0]
            else:
                a_x = np.random.uniform(angle_x[0], angle_x[1])
            if dim == 3:
                if angle_y[0] == angle_y[1]:
                    a_y = angle_y[0]
                else:
                    a_y = np.random.uniform(angle_y[0], angle_y[1])
                if angle_z[0] == angle_z[1]:
                    a_z = angle_z[0]
                else:
                    a_z = np.random.uniform(angle_z[0], angle_z[1])
                coords = rotate_coords_3d(coords, a_x, a_y, a_z)
            else:
                coords = rotate_coords_2d(coords, a_x)
            modified_coords = True

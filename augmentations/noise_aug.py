import random
import numpy as np
from scipy.ndimage import gaussian_filter
from augmentations.utils import get_range_val

def augment_rician_noise(data_sample, noise_variance=(0, 0.1)):
    variance = random.uniform(noise_variance[0], noise_variance[1])
    data_sample = np.sqrt(
        (data_sample + np.random.normal(0.0, variance, size=data_sample.shape)) ** 2 +
        np.random.normal(0.0, variance, size=data_sample.shape) ** 2)
    return data_sample

def augment_gaussian_noise(data_sample, noise_variance=(0, 0.1)):
    if noise_variance[0] == noise_variance[1]:
        variance = noise_variance[0]
    else:
        variance = random.uniform(noise_variance[0], noise_variance[1])
    data_sample = data_sample + np.random.normal(0.0, variance, size=data_sample.shape)
    return data_sample

def augment_gaussian_blur(data_sample, sigma_range, per_channel=True, p_per_channel=1):
    if not per_channel:
        sigma = get_range_val(sigma_range)
    for c in range(data_sample.shape[0]):
        if np.random.uniform() <= p_per_channel:
            if per_channel:
                sigma = get_range_val(sigma_range)
            print('Augmenting', c, sigma)
            t = gaussian_filter(data_sample[c], sigma*10, order=0)
            print(t.max(), t.min(), data_sample[c].max(), data_sample[c].min())
            data_sample[c] = t
    return data_sample

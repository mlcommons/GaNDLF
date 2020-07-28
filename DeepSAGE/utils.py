import numpy as np
def one_hot(segmask_array, largest_class):
    '''
    Encodes the mask voxels in a stacked numpy array for segmentation
    '''
    list = []
    tum_mask = np.array(list)
    
    # create a range from 0 to largest class and find if it is present in current mask
    # do not consider background
    for idx, j in enumerate(np.arange(largest_class)):
        if j != 0:
            mask_to_append = (segmask_array == j).astype(np.float32)
            if tum_mask.size == 0:
                tum_mask = mask_to_append
            else:
                tum_mask = np.concatenate((tum_mask, mask_to_append), axis=1)
    
    bag_mask = (segmask_array == 0).astype(np.float32)
    onehot_stack = np.concatenate((tum_mask, bag_mask), axis=1)
    return onehot_stack

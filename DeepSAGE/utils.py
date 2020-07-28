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
            tum_mask = np.concatenate((tum_mask, (segmask_array == j).astype(np.uint8)), axis=1)
    
    bag_mask = (segmask_array == 0).astype(np.uint8)
    onehot_stack = np.concatenate((tum_mask, bag_mask), axis=1)
    return onehot_stack

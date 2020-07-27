import numpy as np
def one_hot(segmask_array):
    '''
    Encodes the mask voxels in a stacked numpy array for segmentation
    '''
    # list = []
    # tum_mask = np.array(list)
    
    # # iterate over unique values in segmentation and stack them
    # for j in range(np.unique(segmask_array)):  # or range(len(theta))
        # tum_mask = [tum_mask, (segmask_array == j).astype(np.uint8)]
    tum_mask = (segmask_array>0.5).astype(np.uint8)
    bag_mask = (segmask_array == 0).astype(np.uint8)
    onehot_stack = [tum_mask, bag_mask]
    return np.array(onehot_stack)
   

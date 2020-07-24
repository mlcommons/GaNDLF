import numpy as np
def one_hot(segmask_array):
    tum_mask = (segmask_array>0.5).astype(np.uint8)
    bag_mask = (segmask_array == 0).astype(np.uint8)
    onehot_stack = [tum_mask, bag_mask]
    return np.array(onehot_stack)
   

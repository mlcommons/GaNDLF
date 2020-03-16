import numpy as np




def one_hot_nonoverlap(segmask_array):
    ed_mask = (segmask_array == 2).astype(np.uint8)
    enh_mask = (segmask_array == 4).astype(np.uint8)
    nec_mask = (segmask_array == 1).astype(np.uint8)
    bag_mask = (segmask_array == 0).astype(np.uint8)
    onehot_stack = [enh_mask, nec_mask, ed_mask, bag_mask]
    return np.array(onehot_stack)
    
def one_hot_2_overlap(segmask_array):
    wht_mask = (segmask_array >= 1).astype(np.uint8)
    tuc_mask = np.logical_or(segmask_array == 1, segmask_array == 4)
    tuc_mask = tuc_mask.astype(np.uint8)
    enh_mask = (segmask_array == 4).astype(np.uint8)
    onehot_stack = [wht_mask, tuc_mask, enh_mask]
    return np.array(onehot_stack)

import numpy as np
def one_hot(segmask_array, class_list):
    one_hot_stack = []
    for class_ in class_list:
        bin_mask = (segmask_array == int(class_))
        one_hot_stack = one_hot_stack.append(bin_mask)
    one_hot_stack = np.array(one_hot)
    return one_hot

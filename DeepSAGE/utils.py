import numpy as np
def one_hot(segmask_array, class_list):
    one_hot_stack = []
    segmask_array = segmask_array[0,0]
    for class_ in class_list:
        bin_mask = (segmask_array == int(class_))
        one_hot_stack.append(bin_mask)
    one_hot_stack = np.array(one_hot_stack)
    return one_hot_stack
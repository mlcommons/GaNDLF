import numpy as np
import SimpleITK as sitk

def one_hot(segmask_array, class_list):
    batch_size = segmask_array.shape[0]
    batch_stack = []
    for b in range(batch_size):
        one_hot_stack = []
        segmask_array_iter = segmask_array[b,0]
        for class_ in class_list:
            bin_mask = (segmask_array_iter == int(class_))
            one_hot_stack.append(bin_mask)
        one_hot_stack = torch.stack(one_hot_stack)
        batch_stack.append(one_hot_stack)
    batch_stack = torch.stack(batch_stack)    
    return batch_stack


def reverse_one_hot(predmask_array,class_list):
    idx_argmax  = np.argmax(predmask_array,axis=0)
    final_mask = 0
    for idx, class_ in enumerate(class_list):
        final_mask = final_mask +  (idx_argmax == idx)*class_
    return final_mask


def resize_image(input_image, output_size, interpolator = sitk.sitkLinear):
    '''
    This function resizes the input image based on the output size and interpolator
    '''
    inputSize = input_image.GetSize()
    inputSpacing = np.array(input_image.GetSpacing())
    outputSpacing = np.array(inputSpacing)
    for i in range(len(output_size)):
        outputSpacing[i] = inputSpacing[i] * (inputSize[i] / output_size[i])
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(output_size)
    resampler.SetOutputSpacing(outputSpacing)
    resampler.SetOutputOrigin(input_image.GetOrigin())
    resampler.SetOutputDirection(input_image.GetDirection())
    resampler.SetDefaultPixelValue(0)
    return resampler.Execute(input_image)

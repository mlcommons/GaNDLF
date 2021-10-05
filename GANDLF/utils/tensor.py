import os, sys
import numpy as np
import torch
import torch.nn as nn
import torchio


def one_hot(segmask_array, class_list):
    """
    This function creates a one-hot-encoded mask from the segmentation mask Tensor and specified class list

    Args:
        segmask_array (torch.Tensor): The segmentation mask Tensor.
        class_list (list): The list of classes based on which one-hot encoding needs to happen.

    Returns:
        torch.Tensor: The one-hot encoded torch.Tensor
    """
    batch_size = segmask_array.shape[0]
    # batch_stack = None
    def_shape = segmask_array.shape
    batch_stack = torch.zeros(def_shape[0], len(class_list), def_shape[2], def_shape[3], def_shape[4], dtype=torch.float32, device=segmask_array.device)
    for b in range(batch_size):
        # one_hot_stack = np.zeros([len(class_list), def_shape[2], def_shape[3], def_shape[4]])
        # since the input tensor is 5D, with [batch_size, modality, x, y, z], we do not need to consider the modality dimension for labels
        segmask_array_iter = segmask_array[b, 0, ...]
        bin_mask = (segmask_array_iter == 0)  # initialize bin_mask
        # this implementation allows users to combine logical operands

        class_idx = 0
        for _class in class_list:
            if isinstance(_class, str):
                if "||" in _class:  # special case
                    class_split = _class.split("||")
                    bin_mask = segmask_array_iter == int(class_split[0])
                    for i in range(1, len(class_split)):
                        bin_mask = bin_mask | (
                            segmask_array_iter == int(class_split[i])
                        )
                elif "|" in _class:  # special case
                    class_split = _class.split("|")
                    bin_mask = segmask_array_iter == int(class_split[0])
                    for i in range(1, len(class_split)):
                        bin_mask = bin_mask | (
                            segmask_array_iter == int(class_split[i])
                        )
                else:
                    # assume that it is a simple int
                    bin_mask = segmask_array_iter == int(_class)
            else:
                bin_mask = segmask_array_iter == int(_class)
            bin_mask = bin_mask.long()
            # we always ensure the append happens in dim 0, which is blank
            bin_mask = bin_mask.unsqueeze(0)

            batch_stack[b, class_idx, ...] = bin_mask
            class_idx += 1
                
        #     if one_hot_stack is None:
        #         one_hot_stack = bin_mask
        #     else:
        #         one_hot_stack = torch.cat((one_hot_stack, bin_mask))
                
        # if batch_stack is None:
        #     batch_stack = one_hot_stack
        #     # always ensure we are returning a tensor with batch_size encoded
        #     batch_stack = batch_stack.unsqueeze(0)
        # else:
        #     if one_hot_stack.shape != batch_stack.shape:
        #         one_hot_stack = one_hot_stack.unsqueeze(0)
        #     batch_stack = torch.cat((batch_stack, one_hot_stack))
    
    return batch_stack
    # return torch.from_numpy(batch_stack)


def reverse_one_hot(predmask_array, class_list):
    """
    This function creates a full segmentation mask Tensor from a one-hot-encoded mask and specified class list

    Args:
        predmask_array (torch.Tensor): The predicted segmentation mask Tensor.
        class_list (list): The list of classes based on which one-hot encoding needs to happen.

    Returns:
        numpy.array: The final mask as numpy array.
    """
    if isinstance(predmask_array, torch.Tensor):
        array_to_consider = predmask_array.cpu().numpy()
    else:
        array_to_consider = predmask_array
    idx_argmax = np.argmax(array_to_consider, axis=0)
    final_mask = 0
    special_cases_to_check = ["||"]
    special_case_detected = False
    max_current = 0

    for _class in class_list:
        for case in special_cases_to_check:
            if isinstance(_class, str):
                if case in _class:  # check if any of the special cases are present
                    special_case_detected = True
                    # if present, then split the sub-class
                    class_split = _class.split(case)
                    for i in class_split:  # find the max for computation later on
                        if int(i) > max_current:
                            max_current = int(i)

    if special_case_detected:
        start_idx = 0
        if (class_list[0] == 0) or (class_list[0] == "0"):
            start_idx = 1

        final_mask = np.asarray(predmask_array[start_idx, ...], dtype=int)
        start_idx += 1
        for i in range(start_idx, len(class_list)):
            # TODO: this should be replaced with information coming from the config that defines what the specific union of labels should be defined as
            # for example:
            # class_list = "[0,1||2]"
            # output_classes = [0,1]
            # this would make "1||2" be mapped to "1", and this mechanism can be extended for N possible combinations
            final_mask += np.asarray(predmask_array[i, ...], dtype=int)
    else:
        for idx, _class in enumerate(class_list):
            final_mask = final_mask + (idx_argmax == idx) * _class
    return final_mask


def send_model_to_device(model, amp, device, optimizer):
    """
    This function reads the environment variable(s) and send model to correct device

    Args:
        model (torch.nn.Module): The model that needs to be sent to specified device.
        amp (bool): Whether automatic mixed precision is to be used.
        device (str): Device type.
        optimizer (torch.optim): The optimizer for training.

    Returns:
        torch.nn.Module: The model after it has been sent to specified device
        bool: Whether automatic mixed precision is to be used or not.
        torch.device: Device type.
    """
    if device != "cpu":
        if os.environ.get("CUDA_VISIBLE_DEVICES") is None:
            sys.exit(
                "Please set the environment variable 'CUDA_VISIBLE_DEVICES' correctly before trying to run GANDLF on GPU"
            )

        dev = os.environ.get("CUDA_VISIBLE_DEVICES")
        # multi-gpu support
        # ###
        # # https://discuss.pytorch.org/t/cuda-visible-devices-make-gpu-disappear/21439/17?u=sarthakpati
        # ###
        if "," in dev:
            device = torch.device("cuda")
            model = nn.DataParallel(model, "[" + dev + "]")
        else:
            print("Device requested via CUDA_VISIBLE_DEVICES: ", dev)
            print("Total number of CUDA devices: ", torch.cuda.device_count())

            # if only a single visible device, it will be indexed as '0'
            if torch.cuda.device_count() == 1:
                dev = "0"

            dev_int = int(dev)
            print("Device finally used: ", dev)
            # device = torch.device('cuda:' + dev)
            device = torch.device("cuda")
            print("Sending model to aforementioned device")
            model = model.to(device)
            print(
                "Memory Total : ",
                round(
                    torch.cuda.get_device_properties(dev_int).total_memory / 1024 ** 3,
                    1,
                ),
                "GB, Allocated: ",
                round(torch.cuda.memory_allocated(dev_int) / 1024 ** 3, 1),
                "GB, Cached: ",
                round(torch.cuda.memory_reserved(dev_int) / 1024 ** 3, 1),
                "GB",
            )

        print(
            "Device - Current: %s Count: %d Name: %s Availability: %s"
            % (
                torch.cuda.current_device(),
                torch.cuda.device_count(),
                torch.cuda.get_device_name(device),
                torch.cuda.is_available(),
            )
        )

        if not (optimizer is None):
            # ensuring optimizer is in correct device - https://github.com/pytorch/pytorch/issues/8741
            optimizer.load_state_dict(optimizer.state_dict())

    else:
        dev = -1
        device = torch.device("cpu")
        model.cpu()
        amp = False
        print("Since Device is CPU, Mixed Precision Training is set to False")

    return model, amp, device


def get_class_imbalance_weights(training_data_loader, parameters):
    """
    This function calculates the penalty that is used for validation loss in multi-class problems

    Args:
        training_data_loader (torch.utils.data.DataLoader): The training data loader.
        parameters (dict): The parameters passed by the user yaml.

    Returns:
        dict: The penalty weights for different classes under consideration.
    """
    abs_dict = {}  # absolute counts for each class
    weights_dict = {}  # average for "weighted averaging"
    penalty_dict = None  # penalty for misclassification
    # basically, do this for segmentation/classification tasks

    if parameters["problem_type"] != "regression":
        penalty_dict = {}
        for i in range(0, len(parameters["model"]["class_list"])):
            abs_dict[i] = 0
            penalty_dict[i] = 0

    penalty_loader = training_data_loader

    # get the weights for use for dice loss
    total_counter = 0

    # For regression dice penalty need not be taken account
    # For classification this should be calculated on the basis of predicted labels and mask
    # iterate through full penalty data
    for _, (subject) in enumerate(penalty_loader):

        # segmentation needs masks to be one-hot encoded
        if parameters["problem_type"] == "segmentation":
            # accumulate dice weights for each label
            mask = subject["label"][torchio.DATA]
            one_hot_mask = one_hot(mask, parameters["model"]["class_list"])
            for i in range(0, len(parameters["model"]["class_list"])):
                currentNumber = torch.nonzero(
                    one_hot_mask[:, i, ...], as_tuple=False
                ).size(0)
                # class-specific non-zero voxels
                abs_dict[i] += currentNumber
                # total number of non-zero voxels to be considered
                total_counter += currentNumber

        # for classification, the value needs to be used directly
        elif parameters["problem_type"] == "classification":
            # accumulate weights for each label
            value_to_predict = subject["value_0"][0]
            for i in range(0, len(parameters["model"]["class_list"])):
                if value_to_predict == i:
                    abs_dict[i] += 1
                    # we only want to increase the counter for those subjects that are defined in the class_list
                    total_counter += 1

    # Normalize class weights
    weights_dict = {
        key: (val + sys.float_info.epsilon) / total_counter
        for key, val in abs_dict.items()
    }

    # get the raw penalty values
    penalty = {
        key: total_counter / (len(abs_dict) * (val + sys.float_info.epsilon))
        for key, val in abs_dict.items()
    }
    # normalize penalty to sum of 1
    penalty_sum = np.fromiter(penalty.values(), dtype=np.float64).sum()
    penalty_dict = {
        key: (val + sys.float_info.epsilon) / penalty_sum
        for key, val in penalty.items()
    }

    return penalty_dict, weights_dict

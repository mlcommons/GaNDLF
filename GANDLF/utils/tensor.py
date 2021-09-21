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
    batch_stack = []
    for b in range(batch_size):
        one_hot_stack = []
        segmask_array_iter = segmask_array[b, 0]
        bin_mask = segmask_array_iter == 0  # initialize bin_mask
        # this implementation allows users to combine logical operands
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
            one_hot_stack.append(bin_mask)
        one_hot_stack = torch.stack(one_hot_stack)
        batch_stack.append(one_hot_stack)
    batch_stack = torch.stack(batch_stack)
    return batch_stack


def reverse_one_hot(predmask_array, class_list):
    """
    This function creates a full segmentation mask Tensor from a one-hot-encoded mask and specified class list

    Args:
        predmask_array (torch.Tensor): The predicted segmentation mask Tensor.
        class_list (list): The list of classes based on which one-hot encoding needs to happen.

    Returns:
        torch.Tensor: The final mask torch.Tensor.
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

        final_mask = np.asarray(predmask_array[start_idx, :, :, :], dtype=int)
        start_idx += 1
        for i in range(start_idx, len(class_list)):
            final_mask += np.asarray(
                predmask_array[0, :, :, :], dtype=int
            )  # predmask_array[i,:,:,:].long()
            # temp_sum = torch.sum(output)
        # output_2 = (max_current - torch.sum(output)) % max_current
        # test_2 = 1
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
                    one_hot_mask[:, i, :, :, :], as_tuple=False
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
    weights_dict = {key: (val + sys.float_info.epsilon) / total_counter for key, val in abs_dict.items()}
    penalty_dict = {key: 1-val + sys.float_info.epsilon for key, val in weights_dict.items()}

    return penalty_dict, weights_dict

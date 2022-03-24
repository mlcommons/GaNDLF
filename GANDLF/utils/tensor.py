import os, sys
import numpy as np
import torch
import torch.nn as nn
import torchio
from tqdm import tqdm


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

    def_shape = segmask_array.shape
    if len(def_shape) == 4:
        # special case for sdnet
        batch_stack = torch.zeros(
            def_shape[0],
            len(class_list),
            def_shape[2],
            def_shape[3],
            dtype=torch.float32,
            device=segmask_array.device,
        )
    else:
        batch_stack = torch.zeros(
            def_shape[0],
            len(class_list),
            def_shape[2],
            def_shape[3],
            def_shape[4],
            dtype=torch.float32,
            device=segmask_array.device,
        )

    for b in range(batch_size):
        # since the input tensor is 5D, with [batch_size, modality, x, y, z], we do not need to consider the modality dimension for labels
        segmask_array_iter = segmask_array[b, 0, ...]
        bin_mask = segmask_array_iter == 0  # initialize bin_mask

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

    return batch_stack


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
    special_cases_to_check = ["||"]
    special_case_detected = False
    # max_current = 0

    for _class in class_list:
        for case in special_cases_to_check:
            if isinstance(_class, str):
                if case in _class:  # check if any of the special cases are present
                    special_case_detected = True

    final_mask = None
    if special_case_detected:
        # for this case, the output mask will be formed with the following logic:
        # for each class in class_list
        #   unless the class is 0, the output will be index * predmask_array at that class
        array_to_consider_bool = array_to_consider.astype(bool)
        for idx, i in enumerate(class_list):
            initialize_mask = False
            if isinstance(class_list[idx], str):
                for case in special_cases_to_check:
                    initialize_mask = False
                    if case in class_list[idx]:
                        initialize_mask = True
                    else:
                        if class_list[idx] != "0":
                            initialize_mask = True
            else:
                if class_list[idx] != 0:
                    initialize_mask = True

            if initialize_mask:
                if final_mask is None:
                    final_mask = idx * np.asarray(
                        array_to_consider_bool[idx, ...], dtype=int
                    )
                else:
                    final_mask[array_to_consider_bool[idx, ...]] = idx
    else:
        for i, _ in enumerate(class_list):
            if final_mask is None:
                final_mask = np.asarray(array_to_consider[i, ...], dtype=int) * int(i)
            else:
                final_mask += np.asarray(array_to_consider[i, ...], dtype=int) * int(i)
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
                    torch.cuda.get_device_properties(dev_int).total_memory / 1024**3,
                    1,
                ),
                "GB, Allocated: ",
                round(torch.cuda.memory_allocated(dev_int) / 1024**3, 1),
                "GB, Cached: ",
                round(torch.cuda.memory_reserved(dev_int) / 1024**3, 1),
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


def get_class_imbalance_weights_classification(training_df, params):
    """
    This function calculates the penalty used for loss functions in multi-class problems.
    It looks at the column "valuesToPredict" and identifies unique classes, fetches the class distribution
    and generates the required class weights, and then generates the weights needed for loss function which
    are inverse of the class weights generated divided by the total number of items in a single column

    Args:
        training_Df (pd.DataFrame): The training data frame.
        parameters (dict) : The parameters passed by the user yaml.

    Returns:
        dict: The penalty weights for different classes under consideration for classification.

    """
    class_count = training_df["ValueToPredict"].value_counts().to_dict()
    total_count = len(training_df)

    penalty_dict, weight_dict = {}, {}
    for i in range(params["model"]["num_classes"]):
        penalty_dict[i], weight_dict[i] = 0, 0

    for label in class_count.keys():
        weight_dict[label] = class_count[label] / total_count

    for label in class_count.keys():
        penalty_dict[label] = total_count / class_count[label]

    penalty_sum = np.fromiter(penalty_dict.values(), dtype=np.float64).sum()

    for label in class_count.keys():
        penalty_dict[label] = penalty_dict[label] / penalty_sum

    return penalty_dict, weight_dict


def get_class_imbalance_weights_segmentation(training_data_loader, parameters):
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
    for _, (subject) in enumerate(
        tqdm(penalty_loader, desc="Looping over training data for penalty calculation")
    ):

        # accumulate dice weights for each label
        mask = subject["label"][torchio.DATA]
        one_hot_mask = one_hot(mask, parameters["model"]["class_list"])
        for i in range(0, len(parameters["model"]["class_list"])):
            currentNumber = torch.nonzero(one_hot_mask[:, i, ...], as_tuple=False).size(
                0
            )
            # class-specific non-zero voxels
            abs_dict[i] += currentNumber
            # total number of non-zero voxels to be considered
            total_counter += currentNumber

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

    return weights_dict, penalty_dict


def get_linear_interpolation_mode(dimensionality):
    """
    Get linear interpolation mode.

    Args:
        dimensionality (int): The dimensions based on which interpolation mode is calculated

    Returns:
        str: Interpolation type to pass to interpolation function
    """

    mode = "nearest"
    if dimensionality == 2:
        mode = "bilinear"
    elif dimensionality == 3:
        mode = "trilinear"

    return mode

import os, sys, math
from datetime import datetime

os.environ[
    "TORCHIO_HIDE_CITATION_PROMPT"
] = "1"  # hides torchio citation request, see https://github.com/fepegar/torchio/issues/235
import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
import torch.nn as nn
import torchio
from GANDLF.models.modelBase import get_final_layer


def resample_image(
    img, spacing, size=None, interpolator=sitk.sitkLinear, outsideValue=0
):
    """
    Resample image to certain spacing and size.

    Args:
        img (SimpleITK.Image): The input image to resample.
        spacing (list): List of length 3 indicating the voxel spacing as [x, y, z].
        size (list, optional): List of length 3 indicating the number of voxels per dim [x, y, z], which will use compute the appropriate size based on the spacing. Defaults to [].
        interpolator (SimpleITK.InterpolatorEnum, optional): The interpolation type to use. Defaults to SimpleITK.sitkLinear.
        origin (list, optional): The location in physical space representing the [0,0,0] voxel in the input image.  Defaults to [0,0,0].
        outsideValue (int, optional): value used to pad are outside image.  Defaults to 0.

    Raises:
        Exception: Spacing/resolution mismatch.
        Exception: Size mismatch.

    Returns:
        SimpleITK.Image: The resampled input image.
    """
    if len(spacing) != img.GetDimension():
        raise Exception("len(spacing) != " + str(img.GetDimension()))

    # Set Size
    if size == None:
        inSpacing = img.GetSpacing()
        inSize = img.GetSize()
        size = [
            int(math.ceil(inSize[i] * (inSpacing[i] / spacing[i])))
            for i in range(img.GetDimension())
        ]
    else:
        if len(size) != img.GetDimension():
            raise Exception("len(size) != " + str(img.GetDimension()))

    # Resample input image
    return sitk.Resample(
        img,
        size,
        sitk.Transform(),
        interpolator,
        img.GetOrigin(),
        spacing,
        img.GetDirection(),
        outsideValue,
    )


def resize_image(input_image, output_size, interpolator=sitk.sitkLinear):
    """
    This function resizes the input image based on the output size and interpolator

    Args:
        input_image (SimpleITK.Image): The input image to resample.
        output_size (numpy.array): The output size to resample input_image to.
        interpolator (SimpleITK.InterpolatorEnum, optional): The interpolation type to use. Defaults to SimpleITK.sitkLinear.

    Returns:
        SimpleITK.Image: The resized input image.
    """
    inputSize = input_image.GetSize()
    inputSpacing = np.array(input_image.GetSpacing())
    outputSpacing = np.array(inputSpacing)

    if len(output_size) != len(inputSpacing):
        sys.exit(
            "The output size dimension is inconsistent with the input dataset, please check parameters."
        )

    for i, n in enumerate(output_size):
        outputSpacing[i] = inputSpacing[i] * (inputSize[i] / n)

    return resample_image(input_image, outputSpacing, interpolator=interpolator)


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
        for (
            _class
        ) in class_list:  # this implementation allows users to combine logical operands
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
                    class_split = _class.split(
                        case
                    )  # if present, then split the sub-class
                    for i in class_split:  # find the max for computation later on
                        if int(i) > max_current:
                            max_current = int(i)

    if special_case_detected:
        start_idx = 0
        if (class_list[0] == 0) or (class_list[0] == "0"):
            start_idx = 1

        final_mask = np.asarray(
            predmask_array[start_idx, :, :, :], dtype=int
        )  # predmask_array[0,:,:,:].long()
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


def checkPatchDivisibility(patch_size, number=16):
    """
    This function checks the divisibility of a numpy array or integer for architectural integrity

    Args:
        patch_size (numpy.array): The patch size for checking.
        number (int, optional): The number to check divisibility for. Defaults to 16.

    Returns:
        bool: If all elements of array are divisible or not, after taking 2D patches into account.
    """
    if isinstance(patch_size, int):
        patch_size_to_check = np.array(patch_size)
    else:
        patch_size_to_check = patch_size
    if (
        patch_size_to_check[-1] == 1
    ):  # for 2D, don't check divisibility of last dimension
        patch_size_to_check = patch_size_to_check[:-1]
    elif (
        patch_size_to_check[0] == 1
    ):  # for 2D, don't check divisibility of first dimension
        patch_size_to_check = patch_size_to_check[1:]
    if np.count_nonzero(np.remainder(patch_size_to_check, number)) > 0:
        return False

    # adding check to address https://github.com/CBICA/GaNDLF/issues/53
    # there is quite possibly a better way to do this
    unique = np.unique(patch_size_to_check)
    if (unique.shape[0] == 1) and (unique[0] <= number):
        return False
    return True


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


def fix_paths(cwd):
    """
    This function takes the current working directory of the script (which is required for VIPS) and sets up all the paths correctly

    Args:
        cwd (str): The current working directory.
    """
    if os.name == "nt":  # proceed for windows
        vipshome = os.path.join(cwd, "vips/vips-dev-8.10/bin")
        os.environ["PATH"] = vipshome + ";" + os.environ["PATH"]


def populate_header_in_parameters(parameters, headers):
    """
    This function populates the parameters with information from the header in a common manner

    Args:
        parameters (dict): The parameters passed by the user yaml.
        headers (dict): The CSV headers dictionary.

    Returns:
        dict: Combined parameter dictionary containing header information
    """
    # initialize common parameters based on headers
    parameters["headers"] = headers
    # ensure the number of output classes for model prediction is working correctly

    if len(headers["predictionHeaders"]) > 0:
        parameters["model"]["num_classes"] = len(headers["predictionHeaders"])
    is_regression, is_classification, is_segmentation = find_problem_type(
        parameters["headers"], get_final_layer(parameters["model"]["final_layer"])
    )

    # if the problem type is classification/segmentation, ensure the number of classes are picked from the configuration
    if not is_regression:
        parameters["model"]["num_classes"] = len(parameters["model"]["class_list"])

    # initialize number of channels for processing
    if not ("num_channels" in parameters["model"]):
        parameters["model"]["num_channels"] = len(headers["channelHeaders"])

    if is_regression:
        parameters["problem_type"] = "regression"
    elif is_classification:
        parameters["problem_type"] = "classification"
    elif is_segmentation:
        parameters["problem_type"] = "segmentation"

    return parameters


def find_problem_type(headersFromCSV, model_final_layer):
    """
    This function determines the type of problem at hand - regression, classification or segmentation

    Args:
        headersFromCSV (dict): The CSV headers dictionary.
        model_final_layer (model_final_layer): The final layer of the model. If None, the model is for regression.

    Returns:
        bool: If problem is regression.
        bool: If problem is classification.
        bool: If problem is segmentation.
    """
    # initialize problem type
    is_regression = False
    is_classification = False
    is_segmentation = False

    # check if regression/classification has been requested
    if len(headersFromCSV["predictionHeaders"]) > 0:
        if model_final_layer is None:
            is_regression = True
        else:
            is_classification = True
    else:
        is_segmentation = True

    return is_regression, is_classification, is_segmentation


def writeTrainingCSV(inputDir, channelsID, labelID, outputFile):
    """
    This function writes the CSV file based on the input directory, channelsID + labelsID strings

    Args:
        inputDir (str): The input directory.
        channelsID (str): The channel header(s) identifiers.
        labelID (str): The label header identifier.
        outputFile (str): The output files to write
    """
    channelsID_list = channelsID.split(",")  # split into list

    outputToWrite = "SubjectID,"
    for i, n in enumerate(channelsID_list):
        outputToWrite = outputToWrite + "Channel_" + str(i) + ","
    outputToWrite = outputToWrite + "Label"
    outputToWrite = outputToWrite + "\n"

    # iterate over all subject directories
    for dirs in os.listdir(inputDir):
        currentSubjectDir = os.path.join(inputDir, dirs)
        if os.path.isdir(currentSubjectDir):  # only consider folders
            filesInDir = os.listdir(
                currentSubjectDir
            )  # get all files in each directory
            maskFile = ""
            allImageFiles = ""
            for channel in channelsID_list:
                for i, n in enumerate(filesInDir):
                    currentFile = os.path.abspath(os.path.join(currentSubjectDir, n))
                    currentFile = currentFile.replace("\\", "/")
                    if channel in n:
                        allImageFiles += currentFile + ","
                    elif labelID in n:
                        maskFile = currentFile
            if allImageFiles:
                outputToWrite += dirs + "," + allImageFiles + maskFile + "\n"

    file = open(outputFile, "w")
    file.write(outputToWrite)
    file.close()


def parseTrainingCSV(inputTrainingCSVFile, train=True):
    """
    This function parses the input training CSV and returns a dictionary of headers and the full (randomized) data frame

    Args:
        inputTrainingCSVFile (str): The input data CSV file which contains all training data.
        train (bool, optional): Whether performing training. Defaults to True.

    Returns:
        pandas.DataFrame: The full dataset for computation.
        dict: The dictionary containing all relevant CSV headers.
    """
    ## read training dataset into data frame
    data_full = pd.read_csv(inputTrainingCSVFile)
    # shuffle the data - this is a useful level of randomization for the training process
    if train:
        data_full = data_full.sample(frac=1).reset_index(drop=True)

    # find actual header locations for input channel and label
    # the user might put the label first and the channels afterwards
    # or might do it completely randomly
    headers = {}
    headers["channelHeaders"] = []
    headers["predictionHeaders"] = []
    headers["labelHeader"] = None
    headers["subjectIDHeader"] = None

    for col in data_full.columns:
        # add appropriate headers to read here, as needed
        col_lower = col.lower()
        currentHeaderLoc = data_full.columns.get_loc(col)
        if (
            ("channel" in col_lower)
            or ("modality" in col_lower)
            or ("image" in col_lower)
        ):
            headers["channelHeaders"].append(currentHeaderLoc)
        elif "valuetopredict" in col_lower:
            headers["predictionHeaders"].append(currentHeaderLoc)
        elif (
            ("subject" in col_lower) or ("patient" in col_lower) or ("pid" in col_lower)
        ):
            headers["subjectIDHeader"] = currentHeaderLoc
        elif (
            ("label" in col_lower)
            or ("mask" in col_lower)
            or ("segmentation" in col_lower)
            or ("ground_truth" in col_lower)
            or ("groundtruth" in col_lower)
        ):
            if headers["labelHeader"] == None:
                headers["labelHeader"] = currentHeaderLoc
            else:
                print(
                    "WARNING: Multiple label headers found in training CSV, only the first one will be used",
                    file=sys.stderr,
                )

    return data_full, headers


def get_date_time():
    """
    Get a well-parsed date string

    Returns:
        str: The date in format YYYY/MM/DD::HH:MM:SS
    """
    now = datetime.now().strftime("%Y/%m/%d::%H:%M:%S")
    return now


def get_class_imbalance_weights(training_data_loader, parameters):
    """
    This function calculates the penalty that is used for validation loss in multi-class problems

    Args:
        training_data_loader (torch.utils.data.DataLoader): The training data loader.
        parameters (dict): The parameters passed by the user yaml.

    Returns:
        dict: The penalty weights for different classes under consideration.
    """
    weights_dict = {}  # average for "weighted averaging"
    penalty_dict = None  # penalty for misclassification
    # basically, do this for segmentation/classification tasks
    if parameters["problem_type"] is not "regression":
        penalty_dict = {}
        for i in range(0, len(parameters["model"]["class_list"])):
            weights_dict[i] = 0
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
                weights_dict[i] += currentNumber
                # total number of non-zero voxels to be considered
                total_counter += currentNumber

        # for classification, the value needs to be used directly
        elif parameters["problem_type"] == "classification":
            # accumulate weights for each label
            value_to_predict = subject["value_0"][0]
            for i in range(0, len(parameters["model"]["class_list"])):
                if value_to_predict == i:
                    weights_dict[i] += 1
                    # we only want to increase the counter for those subjects that are defined in the class_list
                    total_counter += 1

    # get the penalty values - weights_dict contains the overall number for each class in the penalty data
    for i in range(0, len(parameters["model"]["class_list"])):
        penalty = total_counter  # start with the assumption that all the non-zero voxels (segmentation) or activate labels (classification) make up the penalty
        for j in range(0, len(parameters["model"]["class_list"])):
            if i != j:  # for differing classes, subtract the current weight
                penalty -= weights_dict[j]
        
        # finally, the "penalty" variable contains the total number of voxels/activations that are not part of the current class
        # this is to be used to weight the loss function
        penalty_dict[i] = penalty / total_counter

    return penalty_dict


def get_filename_extension_sanitized(filename):
    """
    This function returns the extension of the filename with leading and trailing characters removed.
    Args:
        filename (str): The filename to be processed.
    Returns:
        str: The filename with extension removed.
    """
    _, ext = os.path.splitext(filename)
    # if .gz or .nii file is detected, always return .nii.gz
    if (ext == ".gz") or (ext == ".nii"):
        ext = ".nii.gz"
    return ext


def populate_channel_keys_in_params(data_loader, parameters):
    """
    Function to read channel key information from specified data loader

    Args:
        data_loader (torch.DataLoader): The data loader to query key information from.
        parameters (dict): The parameters passed by the user yaml.

    Returns:
        dict: Updated parameters that include key information
    """
    batch = next(
        iter(data_loader)
    )  # using train_loader makes this slower as train loader contains full augmentations
    all_keys = list(batch.keys())
    channel_keys = []
    value_keys = []
    print("All Keys : ", all_keys)
    for item in all_keys:
        if item.isnumeric():
            channel_keys.append(item)
        elif "value" in item:
            value_keys.append(item)
    parameters["channel_keys"] = channel_keys
    if value_keys:
        parameters["value_keys"] = value_keys

    return parameters


def perform_sanity_check_on_subject(subject, parameters):
    """
    This function performs sanity check on the subject to ensure presence of consistent header information WITHOUT loading images into memory.

    Args:
        subject (torchio.Subject): The input subject.
        parameters (dict): The parameters passed by the user yaml.

    Returns:
        bool: True if everything is okay.

    Raises:
        ValueError: Dimension mismatch in the images.
        ValueError: Origin mismatch in the images.
        ValueError: Orientation mismatch in the images.
    """
    # read the first image and save that for comparison
    file_reader_base = None

    import copy

    list_for_comparison = copy.deepcopy(parameters["headers"]["channelHeaders"])
    if "labelHeader" in parameters["headers"]:
        list_for_comparison.append("label")

    for key in list_for_comparison:
        if file_reader_base is None:
            file_reader_base = sitk.ImageFileReader()
            file_reader_base.SetFileName(subject[str(key)]["path"])
            file_reader_base.ReadImageInformation()
        else:
            # in this case, file_reader_base is ready
            file_reader_current = sitk.ImageFileReader()
            file_reader_current.SetFileName(subject[str(key)]["path"])
            file_reader_current.ReadImageInformation()

            if file_reader_base.GetDimension() != file_reader_current.GetDimension():
                raise ValueError(
                    "Dimensions for Subject '"
                    + subject["subject_id"]
                    + "' are not consistent."
                )

            if file_reader_base.GetOrigin() != file_reader_current.GetOrigin():
                raise ValueError(
                    "Origin for Subject '"
                    + subject["subject_id"]
                    + "' are not consistent."
                )

            if file_reader_base.GetDirection() != file_reader_current.GetDirection():
                raise ValueError(
                    "Orientation for Subject '"
                    + subject["subject_id"]
                    + "' are not consistent."
                )

            if file_reader_base.GetSpacing() != file_reader_current.GetSpacing():
                raise ValueError(
                    "Spacing for Subject '"
                    + subject["subject_id"]
                    + "' are not consistent."
                )

    return True

from skimage.exposure.exposure import _output_dtype
from GANDLF.inference_loop import inference_loop
import os


def InferenceManager(dataframe, outputDir, parameters, device):
    """
    This function takes in a dataframe, with some other parameters and performs the inference
    """
    # get the indeces for kfold splitting
    inferenceData_full = dataframe

    # # initialize parameters for inference
    if not ("weights" in parameters):
        parameters["weights"] = None  # no need for loss weights for inference
    if not ("class_weights" in parameters):
        parameters["class_weights"] = None  # no need for class weights for inference

    n_folds = parameters["nested_training"]["validation"]

    fold_dirs = []
    if n_folds > 1:
        for dir in sorted(os.listdir(outputDir)):
            if dir.isdigit():
                fold_dirs.append(os.path.join(outputDir, dir, ""))
    else:
        fold_dirs = [outputDir]

    for fold_dir in fold_dirs:

        inference_loop(
            inferenceDataFromPickle=inferenceData_full,
            outputDir=fold_dir,
            device=device,
            parameters=parameters,
        )

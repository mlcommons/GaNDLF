from GANDLF.compute import inference_loop
import os
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd


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
        directories = sorted(os.listdir(outputDir))
        for d in directories:
            if d.isdigit():
                fold_dirs.append(os.path.join(outputDir, d, ""))
    else:
        fold_dirs = [outputDir]

    # this is for the case where inference is happening using a single model
    if len(fold_dirs) == 0:
        fold_dirs = [outputDir]

    probs_list = []

    is_classification = parameters["problem_type"] == "classification"

    for fold_dir in fold_dirs:
        parameters["current_fold_dir"] = fold_dir
        inference_loop(
            inferenceDataFromPickle=inferenceData_full,
            outputDir=fold_dir,
            device=device,
            parameters=parameters,
        )

        if parameters["problem_type"] == "classification":
            logits_dir = os.path.join(fold_dir, "logits.csv")
            is_logits_dir_exist = os.path.isfile(logits_dir)

            if is_classification and is_logits_dir_exist:
                # fold_logits = np.genfromtxt(logits_dir, delimiter=",")
                fold_logits = pd.read_csv(logits_dir)[
                    parameters["model"]["class_list"]
                ].values
                fold_logits = torch.from_numpy(fold_logits)
                fold_probs = F.softmax(fold_logits, dim=1)
                probs_list.append(fold_probs)

    if probs_list and is_classification:
        columns = ["SubjectID", "PredictedClass"] + parameters["model"]["class_list"]
        # columns = ["SubjectID",] + parameters["model"]["class_list"]
        averaged_probs_df = pd.DataFrame(columns=columns)
        averaged_probs_df.SubjectID = dataframe[0]

        probs_list = torch.stack(probs_list)
        averaged_probs = torch.mean(probs_list, 0).numpy()
        averaged_probs_df[parameters["model"]["class_list"]] = averaged_probs
        averaged_probs_df.PredictedClass = [
            parameters["model"]["class_list"][a] for a in averaged_probs.argmax(1)
        ]
        averaged_probs_df.to_csv(
            os.path.join(outputDir, "averaged_probabilities.csv"), index=False, sep=","
        )
        # np.savetxt(
        #     os.path.join(outputDir, "averaged_probabilities.csv"),
        #     averaged_probs,
        #     delimiter=",",
        # )

import os
from typing import Optional
from pathlib import Path
import pandas as pd
import torch
import torch.nn.functional as F

from GANDLF.compute import inference_loop
from GANDLF.utils import get_unique_timestamp


def InferenceManager(
    dataframe: pd.DataFrame,
    modelDir: str,
    parameters: dict,
    device: str,
    outputDir: Optional[str] = None,
) -> None:
    """
    This function is used to perform inference on a model using a dataframe.

    Args:
        dataframe (pd.DataFrame): The dataframe containing the data to be used for inference.
        modelDir (str): The path to the model directory.
        parameters (dict): The parameters to be used for inference.
        device (str): The device to be used for inference.
        outputDir (Optional[str], optional): The output directory for the inference results. Defaults to None.
    """
    # get the indeces for kfold splitting
    inferenceData_full = dataframe

    # if outputDir is not provided, create a new directory with a unique timestamp
    if outputDir is None:
        outputDir = os.path.join(modelDir, get_unique_timestamp())
        print(
            "Output directory not provided, creating a new directory with a unique timestamp: ",
            outputDir,
        )
    Path(outputDir).mkdir(parents=True, exist_ok=True)

    parameters["output_dir"] = outputDir

    # initialize parameters for inference
    for key in ["penalty_weights", "sampling_weights", "class_weights"]:
        parameters[key] = parameters.get(key, None)

    n_folds = parameters["nested_training"]["validation"]

    modelDir_split = [modelDir]
    if "," in modelDir:
        modelDir_split = modelDir.split(",")

    averaged_probs_list = []
    for current_modelDir in modelDir_split:
        fold_dirs = []
        if n_folds > 1:
            directories = sorted(os.listdir(current_modelDir))
            for d in directories:
                if d.isdigit():
                    fold_dirs.append(os.path.join(current_modelDir, d, ""))
        else:
            fold_dirs = [current_modelDir]

        # this is for the case where inference is happening using a single model
        if len(fold_dirs) == 0:
            fold_dirs = [current_modelDir]

        probs_list = []
        class_list = None
        is_classification = parameters["problem_type"] == "classification"

        # initialize model type for processing: if not defined, default to torch
        if not ("type" in parameters["model"]):
            parameters["model"]["type"] = "torch"

        for fold_dir in fold_dirs:
            parameters["current_fold_dir"] = fold_dir
            inference_loop(
                inferenceDataFromPickle=inferenceData_full,
                modelDir=fold_dir,
                device=device,
                parameters=parameters,
                outputDir=outputDir,
            )

            if is_classification:
                logits_dir = os.path.join(fold_dir, "logits.csv")
                is_logits_dir_exist = os.path.isfile(logits_dir)

                if is_logits_dir_exist:
                    # fold_logits = np.genfromtxt(logits_dir, delimiter=",")
                    class_list = [str(c) for c in parameters["model"]["class_list"]]
                    fold_logits = pd.read_csv(logits_dir)[class_list].values
                    fold_logits = torch.from_numpy(fold_logits)
                    fold_probs = F.softmax(fold_logits, dim=1)
                    probs_list.append(fold_probs)

        if is_classification and (n_folds > 1):
            probs_list = torch.stack(probs_list)
            averaged_probs_list.append(torch.mean(probs_list, 0))

    # this logic should be changed if we want to do multi-fold inference for histo images
    if (parameters["modality"] == "rad") and averaged_probs_list and is_classification:
        columns = ["SubjectID", "PredictedClass"] + parameters["model"]["class_list"]
        averaged_probs_df = pd.DataFrame(columns=columns)
        averaged_probs_df.SubjectID = dataframe[0]

        averaged_probs_across_models = torch.stack(averaged_probs_list)
        averaged_probs_across_models = torch.mean(
            averaged_probs_across_models, 0
        ).numpy()
        averaged_probs_df[class_list] = averaged_probs_across_models
        averaged_probs_df.PredictedClass = [
            class_list[a] for a in averaged_probs_across_models.argmax(1)
        ]
        filepath_to_save = os.path.join(outputDir, "final_preds_and_avg_probs.csv")
        if os.path.isfile(filepath_to_save):
            filepath_to_save = os.path.join(
                outputDir,
                "final_preds_and_avg_probs" + get_unique_timestamp() + ".csv",
            )
        averaged_probs_df.to_csv(filepath_to_save, index=False)

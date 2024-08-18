import os
from pathlib import Path
from typing import Optional

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
        parameters.setdefault(key, None)

    n_folds = parameters["nested_training"]["validation"]
    modelDir_split = modelDir.split(",") if "," in modelDir else [modelDir]

    averaged_probs_list = []
    for current_modelDir in modelDir_split:
        fold_dirs = (
            [
                os.path.join(current_modelDir, d, "")
                for d in sorted(os.listdir(current_modelDir))
                if d.isdigit()
            ]
            if n_folds > 1
            else [current_modelDir]
        )

        probs_list = []
        is_classification = parameters["problem_type"] == "classification"
        parameters["model"].setdefault("type", "torch")
        class_list = (
            [str(c) for c in parameters["model"]["class_list"]]
            if is_classification
            else None
        )

        for fold_dir in fold_dirs:
            parameters["current_fold_dir"] = fold_dir
            inference_loop(
                inferenceDataFromPickle=dataframe,
                modelDir=fold_dir,
                device=device,
                parameters=parameters,
                outputDir=outputDir,
            )

            if is_classification:
                logits_path = os.path.join(fold_dir, "logits.csv")
                if os.path.isfile(logits_path):
                    fold_logits = pd.read_csv(logits_path)[class_list].values
                    fold_logits = torch.from_numpy(fold_logits)
                    fold_probs = F.softmax(fold_logits, dim=1)
                    probs_list.append(fold_probs)

        if is_classification and probs_list:
            probs_list = torch.stack(probs_list)
            averaged_probs_list.append(torch.mean(probs_list, 0))

    # this logic should be changed if we want to do multi-fold inference for histo images
    if averaged_probs_list and is_classification:
        averaged_probs_df = pd.DataFrame(
            columns=["SubjectID", "PredictedClass"] + class_list
        )
        averaged_probs_df["SubjectID"] = dataframe.iloc[:, 0]

        averaged_probs_across_models = torch.mean(
            torch.stack(averaged_probs_list), 0
        ).numpy()
        averaged_probs_df[class_list] = averaged_probs_across_models
        averaged_probs_df["PredictedClass"] = [
            class_list[idx] for idx in averaged_probs_across_models.argmax(axis=1)
        ]

        filepath_to_save = os.path.join(outputDir, "final_preds_and_avg_probs.csv")
        if os.path.isfile(filepath_to_save):
            filepath_to_save = os.path.join(
                outputDir,
                "final_preds_and_avg_probs_" + get_unique_timestamp() + ".csv",
            )

        averaged_probs_df.to_csv(filepath_to_save, index=False)

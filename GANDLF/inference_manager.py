import os
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from GANDLF.utils import get_unique_timestamp
import lightning.pytorch as pl
from GANDLF.models.lightning_module import GandlfLightningModule
from GANDLF.data.lightning_datamodule import GandlfInferenceDatamodule


def InferenceManager(
    dataframe: pd.DataFrame,
    modelDir: str,
    parameters: dict,
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
        outputDir = os.path.join(modelDir, get_unique_timestamp(), "output_inference")
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

    # This should be handled by config parser
    accelerator = parameters.get("accelerator", "auto")
    allowed_accelerators = ["cpu", "gpu", "auto"]
    # codacy ignore Generic/ReDoS: This is not a SQL query, it's an error message.
    assert (
        accelerator in allowed_accelerators
    ), f"Invalid accelerator selected: {accelerator}. Please select from {allowed_accelerators}"
    strategy = parameters.get("strategy", "auto")
    allowed_strategies = ["auto", "ddp"]
    # codacy ignore Generic/ReDoS: This is not a SQL query, it's an error message.
    assert (
        strategy in allowed_strategies
    ), f"Invalid strategy selected: {strategy}. Please select from {allowed_strategies}"
    precision = parameters.get("precision", "32")
    allowed_precisions = [
        "64",
        "64-true",
        "32",
        "32-true",
        "16",
        "16-mixed",
        "bf16",
        "bf16-mixed",
    ]
    # codacy ignore Generic/ReDoS: This is not a SQL query, it's an error message.
    assert (
        precision in allowed_precisions
    ), f"Invalid precision selected: {precision}. Please select from {allowed_precisions}"

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

        is_classification = parameters["problem_type"] == "classification"
        parameters["model"].setdefault("type", "torch")
        class_list = (
            [str(c) for c in parameters["model"]["class_list"]]
            if is_classification
            else None
        )
        probs_list = None
        for fold_dir in fold_dirs:
            trainer = pl.Trainer(
                accelerator=accelerator,
                strategy=strategy,
                fast_dev_run=False,
                devices=parameters.get("devices", "auto"),
                num_nodes=parameters.get("num_nodes", 1),
                precision=precision,
                gradient_clip_algorithm=parameters["clip_mode"],
                gradient_clip_val=parameters["clip_grad"],
                max_epochs=parameters["num_epochs"],
                sync_batchnorm=False,
                enable_checkpointing=False,
                logger=False,
                num_sanity_val_steps=0,
            )
            datamodule = GandlfInferenceDatamodule(dataframe, parameters)
            parameters = datamodule.updated_parameters_dict
            lightning_module = GandlfLightningModule(parameters, output_dir=fold_dir)

            if parameters.get("auto_batch_size_find", False):
                print(
                    "Auto batch size find is not supported in inference. Dataloader batch size is always 1."
                )

            trainer.predict(lightning_module, datamodule=datamodule)
            if is_classification:
                prob_values_for_all_subjects_in_fold = list(
                    lightning_module.subject_classification_class_probabilities.values()
                )
                if prob_values_for_all_subjects_in_fold:
                    probs_list = torch.stack(
                        prob_values_for_all_subjects_in_fold, dim=1
                    )

        if is_classification and probs_list is not None:
            averaged_probs_list.append(torch.mean(probs_list, 0))

    # this logic should be changed if we want to do multi-fold inference for histo images
    if averaged_probs_list and is_classification:
        averaged_probs_df = pd.DataFrame(
            columns=["SubjectID", "PredictedClass"] + class_list
        )
        averaged_probs_df["SubjectID"] = dataframe.iloc[:, 0]

        averaged_probs_across_models = (
            torch.mean(torch.stack(averaged_probs_list), 0).cpu().numpy()
        )
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

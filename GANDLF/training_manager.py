import os
import yaml

# codacy ignore python-use-of-pickle: Pickle usage is safe in this context (local data only).
import pickle
import shutil
import pandas as pd
from pathlib import Path
from warnings import warn

import lightning.pytorch as pl
from lightning.pytorch.profilers import PyTorchProfiler
from lightning.pytorch.tuner import Tuner as LightningTuner

from GANDLF.utils import get_dataframe, split_data
from GANDLF.models.lightning_module import GandlfLightningModule
from GANDLF.data.lightning_datamodule import GandlfTrainingDatamodule

from typing import Optional


def TrainingManager(
    dataframe: pd.DataFrame,
    outputDir: str,
    parameters: dict,
    resume: bool,
    reset: bool,
    profile: Optional[bool] = False,
) -> None:
    """
    This is the training manager that ties all the training functionality together

    Args:
        dataframe (pandas.DataFrame): The full data from CSV.
        outputDir (str): The main output directory.
        parameters (dict): The parameters dictionary.
        resume (bool): Whether the previous run will be resumed or not.
        reset (bool): Whether the previous run will be reset or not.
        profile(bool): Whether we want the profile activity or not. Defaults to False.

    """

    if "output_dir" not in parameters:
        parameters["output_dir"] = outputDir
    if reset:
        shutil.rmtree(outputDir)
        Path(outputDir).mkdir(parents=True, exist_ok=True)

    # save the current model configuration as a sanity check
    currentModelConfigPickle = os.path.join(outputDir, "parameters.pkl")
    if (not os.path.exists(currentModelConfigPickle)) or reset or resume:
        with open(currentModelConfigPickle, "wb") as handle:
            pickle.dump(parameters, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        if os.path.exists(currentModelConfigPickle):
            print(
                "Using previously saved parameter file",
                currentModelConfigPickle,
                flush=True,
            )
            parameters = pickle.load(open(currentModelConfigPickle, "rb"))

    dataframe_split = split_data(dataframe, parameters)

    last_indeces, _, _, _ = dataframe_split[-1]

    # check the last indeces to see if single fold training is requested
    singleFoldTesting = True if last_indeces[0] == 0 else False
    singleFoldValidation = True if last_indeces[1] == 0 else False

    for (
        testing_and_valid_indeces,
        trainingData,
        validationData,
        testingData,
    ) in dataframe_split:
        # the output of the current fold is only needed if multi-fold training is happening
        currentTestingOutputFolder = outputDir
        if not singleFoldTesting:
            currentTestingOutputFolder = os.path.join(
                outputDir, "testing_" + str(testing_and_valid_indeces[0])
            )
            Path(currentTestingOutputFolder).mkdir(parents=True, exist_ok=True)

        currentValidationOutputFolder = currentTestingOutputFolder
        if not singleFoldValidation:
            currentValidationOutputFolder = os.path.join(
                currentTestingOutputFolder, str(testing_and_valid_indeces[1])
            )
            Path(currentValidationOutputFolder).mkdir(parents=True, exist_ok=True)

        # initialize the dataframes and save them to disk
        data_dict = {
            "training": trainingData,
            "validation": validationData,
            "testing": testingData,
        }
        data_dict_files = {}
        for data_type, data in data_dict.items():
            data_dict_files[data_type] = None
            if data is not None:
                currentDataPickle = os.path.join(
                    currentValidationOutputFolder, "data_" + data_type + ".pkl"
                )
                data_dict_files[data_type] = currentDataPickle
                if (not os.path.exists(currentDataPickle)) or reset or resume:
                    data.to_pickle(currentDataPickle)
                    data.to_csv(currentDataPickle.replace(".pkl", ".csv"), index=False)
                else:
                    # read the data from the pickle if present
                    data_dict[data_type] = get_dataframe(currentDataPickle)

        # Dataloader initialization - should be extracted somewhere else (preferably abstracted away)
        datamodule = GandlfTrainingDatamodule(data_dict_files, parameters)
        parameters = datamodule.updated_parameters_dict

        # This entire section should be handled in config parser

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

        warn(
            f"Configured to use {accelerator} with {strategy} for training, but current development configuration will force single-device only training."
        )
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
            profiler=PyTorchProfiler(sort_by="cpu_time_total", row_limit=10)
            if profile
            else None,
        )

        lightning_module = GandlfLightningModule(
            parameters, output_dir=currentValidationOutputFolder
        )

        if parameters.get("auto_batch_size_find", False):
            LightningTuner(trainer).scale_batch_size(
                lightning_module, datamodule=datamodule
            )

        if parameters.get("auto_lr_find", False):
            LightningTuner(trainer).lr_find(lightning_module, datamodule=datamodule)

        trainer.fit(lightning_module, datamodule=datamodule)

        testing_data = data_dict_files.get("testing", None)
        if testing_data:
            trainer.test(lightning_module, datamodule=datamodule)


def TrainingManager_split(
    dataframe_train: pd.DataFrame,
    dataframe_validation: pd.DataFrame,
    dataframe_testing: pd.DataFrame,
    outputDir: str,
    parameters: dict,
    resume: bool,
    reset: bool,
    profile: Optional[bool] = False,
):
    """
    This is the training manager that ties all the training functionality together

    Args:
        dataframe_train (pd.DataFrame): The training data from CSV.
        dataframe_validation (pd.DataFrame): The validation data from CSV.
        dataframe_testing (pd.DataFrame): The testing data from CSV.
        outputDir (str): The main output directory.
        parameters (dict): The parameters dictionary.
        resume (bool): Whether the previous run will be resumed or not.
        reset (bool): Whether the previous run will be reset or not.
        profile(bool): Whether the we want the profile activity or not. Defaults to False.

    """
    currentModelConfigPickle = os.path.join(outputDir, "parameters.pkl")
    currentModelConfigYaml = os.path.join(outputDir, "config.yaml")

    if (not os.path.exists(currentModelConfigPickle)) or reset or resume:
        with open(currentModelConfigPickle, "wb") as handle:
            pickle.dump(parameters, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        if os.path.exists(currentModelConfigPickle):
            print(
                "Using previously saved parameter file",
                currentModelConfigPickle,
                flush=True,
            )
            parameters = pickle.load(open(currentModelConfigPickle, "rb"))

    if (not os.path.exists(currentModelConfigYaml)) or reset or resume:
        with open(currentModelConfigYaml, "w") as handle:
            yaml.dump(parameters, handle, default_flow_style=False)

    data_dict_files = {
        "training": dataframe_train,
        "validation": dataframe_validation,
        "testing": dataframe_testing,
    }

    datamodule = GandlfTrainingDatamodule(data_dict_files, parameters)
    parameters = datamodule.updated_parameters_dict

    # This entire section should be handled in config parser

    accelerator = parameters.get("accelerator", "auto")
    allowed_accelerators = ["cpu", "gpu", "auto"]
    assert (
        accelerator in allowed_accelerators
    ), f"Invalid accelerator selected: {accelerator}. Please select from {allowed_accelerators}"
    strategy = parameters.get("strategy", "auto")
    allowed_strategies = ["auto", "ddp"]
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
    assert (
        precision in allowed_precisions
    ), f"Invalid precision selected: {precision}. Please select from {allowed_precisions}"

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
        profiler=PyTorchProfiler(sort_by="cpu_time_total", row_limit=10)
        if profile
        else None,
    )
    lightning_module = GandlfLightningModule(parameters, output_dir=outputDir)

    if parameters.get("auto_batch_size_find", False):
        LightningTuner(trainer).scale_batch_size(
            lightning_module, datamodule=datamodule
        )

    if parameters.get("auto_lr_find", False):
        LightningTuner(trainer).lr_find(lightning_module, datamodule=datamodule)

    trainer.fit(lightning_module, datamodule=datamodule)

    if dataframe_testing is not None:
        trainer.test(lightning_module, datamodule=datamodule)
